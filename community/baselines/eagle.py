#!/usr/bin/env python3
"""EAGLE next-snapshot (q,k,t) community query and held-out evaluation."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from datasets.community_eval_builder import set_metrics
from datasets.baseline_eval import (baseline_training_split, evaluation_start_t,
                                    load_test_samples, non_empty_samples,
                                    query_set_sha256, require_fit_boundary)
from datasets.baseline_split import training_boundaries
from datasets.dataset_builder import DEFAULT_TEST_RATIO, build_snapshots, load_time_slice_manifest
from methods.eagle import DEFAULT_ROOT, load_predictor, load_runtime
from community.baselines.baseline_graph import PredictedGraph


VALID_KS = (3, 4, 5, 6, 7)
DEFAULT_SLICES = Path(__file__).resolve().parents[2] / "data/mooc/time_slices/step_43200_window_86400"


class EagleCommunityPredictor:
    """Zebra's historical-community/edge candidate protocol, indexed incrementally."""

    def __init__(self, snapshots, scorer, *, threshold, pair_batch_size=65536):
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if scorer.config.branch == "time" and not 0 <= threshold <= 1:
            raise ValueError("Time threshold must be in [0, 1]")
        if pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
        self.snapshots = snapshots
        self.scorer = scorer
        self.threshold = float(threshold)
        self.pair_batch_size = pair_batch_size
        self.prepared_t = None
        self._communities = {k: defaultdict(set) for k in VALID_KS}
        self._neighbors = defaultdict(set)

    def prepare_time(self, t):
        if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < len(self.snapshots) - 1:
            raise IndexError("prediction requires t and t+1")
        if self.prepared_t is not None and t < self.prepared_t:
            raise ValueError("history cannot move backwards")
        self.scorer.prepare_time(t)
        start = 0 if self.prepared_t is None else self.prepared_t + 1
        for index in range(start, t + 1):
            snapshot = self.snapshots[index]
            for k in VALID_KS:
                for component in snapshot.get("k_core_comps", {}).get(k, {}).get("components", ()):
                    nodes = frozenset(int(node) for node in component)
                    for node in nodes:
                        self._communities[k][node].update(nodes)
            for u, v, *_ in snapshot["edge_list"]:
                u, v = int(u), int(v)
                if u != v:
                    left, right = min(u, v), max(u, v)
                    self._neighbors[left].add(right)
            self.prepared_t = index

    def candidate(self, q, k, t):
        if k not in VALID_KS:
            raise ValueError("k must be 3..7")
        if t != self.prepared_t:
            raise ValueError("prepare requested time first")
        return frozenset(self._communities[k].get(q, ()))

    def query(self, q, k, t):
        candidate = self.candidate(q, k, t)
        nodes = np.asarray(sorted(candidate), dtype=np.int64)
        positions = {int(node): i for i, node in enumerate(nodes)}
        # Read only edges observed through t; no all-snapshot edge index.
        pairs = [(u, v) for u in nodes for v in sorted(self._neighbors.get(int(u), ()))
                 if v in positions]
        selected = []
        for start in range(0, len(pairs), self.pair_batch_size):
            batch = np.asarray(pairs[start:start + self.pair_batch_size], dtype=np.int64)
            scores = self.scorer.score_edges(batch, target_t=t + 1,
                                              batch_size=self.pair_batch_size)
            selected.extend(tuple(edge) for edge in batch[scores > self.threshold])
        if selected:
            indices = np.asarray([(positions[u], positions[v]) for u, v in selected], dtype=np.int64)
            row = np.concatenate((indices[:, 0], indices[:, 1]))
            col = np.concatenate((indices[:, 1], indices[:, 0]))
            adjacency = sparse.csr_matrix((np.ones(len(row), dtype=np.bool_), (row, col)),
                                          shape=(len(nodes), len(nodes)))
        else:
            adjacency = sparse.csr_matrix((len(nodes), len(nodes)), dtype=np.bool_)
        graph = PredictedGraph(nodes, adjacency)
        return graph.community(candidate, q, k), {
            "candidate_size": len(nodes), "scored_edge_count": len(pairs),
            "selected_edge_count": graph.edge_count,
        }


def load_eagle_run(slices_dir, checkpoint, *, config_path, device="cpu", eagle_root=DEFAULT_ROOT):
    slices_dir = Path(slices_dir).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint else None
    config_path = Path(config_path).resolve()
    config = load_runtime(eagle_root).SnapshotInferenceConfig(**json.loads(config_path.read_text()))
    if config.branch != "structure" and checkpoint is None:
        raise ValueError("Time/Hybrid require a checkpoint")
    run_path = config_path.parent / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text())
        if (Path(run["slices_dir"]).resolve() != slices_dir or
                run["fit_end_t"] != config.fit_end_t or
                run["protocol"] != "eagle_snapshot_v1" or
                (checkpoint is not None and Path(run["checkpoint"]).resolve() != checkpoint)):
            raise ValueError("EAGLE run provenance does not match dataset/checkpoint/config")
        recorded_manifest = run.get("source_manifest_sha256")
        if recorded_manifest is not None:
            manifest_path = slices_dir / "metadata.json"
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != recorded_manifest:
                raise ValueError("EAGLE training manifest does not match current slices")
    snapshots = build_snapshots(slices_dir)[0]
    if config.fit_end_t is not None and config.fit_end_t >= len(snapshots) - 1:
        raise ValueError("checkpoint fitting reaches past available predictions")
    scorer = load_predictor(snapshots, config=config, checkpoint=checkpoint,
                            device=device, eagle_root=eagle_root)
    return snapshots, scorer


def evaluate(predictor, *, slices_dir, start_t=None, max_samples=None, seed=42):
    snapshots = predictor.snapshots
    manifest = load_time_slice_manifest(slices_dir)
    split_t = int(len(snapshots) * (1 - DEFAULT_TEST_RATIO))
    start_t = evaluation_start_t(len(snapshots)) if start_t is None else start_t
    if not split_t <= start_t < len(snapshots) - 1:
        raise ValueError("start_t must be in held-out range")
    if predictor.scorer.config.fit_end_t is not None and start_t < predictor.scorer.config.fit_end_t:
        raise ValueError("evaluation precedes model fitting boundary")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    # Truth is read only for scoring, never handed to the predictor.
    samples = load_test_samples(slices_dir, len(snapshots))
    samples = non_empty_samples(sample for sample in samples if sample["t"] >= start_t)
    if max_samples is not None:
        samples = samples[:max_samples]
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["t"]].append(sample)
    rows = defaultdict(list)
    for t, group in sorted(grouped.items()):
        predictor.prepare_time(t)
        for sample in group:
            started = time.perf_counter()
            community, stats = predictor.query(sample["query"], sample["k"], t)
            elapsed = time.perf_counter() - started
            rows[sample["k"]].append({
                **set_metrics(community, sample["community"]), **stats,
                "query_s": elapsed, "size_ratio": len(community) / len(sample["community"])
                if sample["community"] else 0.0,
            })
    fields = ("precision", "recall", "f1", "jaccard", "candidate_size",
              "scored_edge_count", "selected_edge_count", "query_s", "size_ratio")
    per_k = {k: {**{field: float(np.mean([row[field] for row in values])) for field in fields},
                  "samples": len(values)} for k, values in sorted(rows.items())}
    return {
        "dataset": manifest["dataset"], "start_t": start_t,
        "sample_scope": "shared_community_eval_non_empty_only",
        "fit_end_t": predictor.scorer.config.fit_end_t, "threshold": predictor.threshold,
        "candidate_history": "union_of_q_historical_k_core_communities_through_t",
        "edge_candidates": "historical_undirected_edges_within_candidate_through_t",
        "samples": len(samples), "per_k": per_k,
        "macro": {field: float(np.mean([row[field] for row in per_k.values()]))
                  for field in fields} if per_k else {},
        "query_set_sha256": query_set_sha256(samples),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("query", "eval"):
        sub = commands.add_parser(name)
        sub.add_argument("--slices-dir", default=str(DEFAULT_SLICES))
        sub.add_argument("--eagle-root", default=str(DEFAULT_ROOT))
        sub.add_argument("--checkpoint", help="trained EAGLE Time checkpoint")
        sub.add_argument("--config", required=True, help="inference_config.json from the training run")
        sub.add_argument("--threshold", type=float, required=True,
                         help="explicit decision threshold (calibrate on validation data)")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--pair-batch-size", type=int, default=65536)
    query = commands.choices["query"]
    query.add_argument("q", type=int)
    query.add_argument("k", type=int, choices=VALID_KS)
    query.add_argument("t", type=int)
    evaluation = commands.choices["eval"]
    evaluation.add_argument("--start-t", type=int)
    evaluation.add_argument("--max-samples", type=int)
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    snapshots, scorer = load_eagle_run(args.slices_dir, args.checkpoint,
                                      config_path=args.config, device=args.device,
                                      eagle_root=args.eagle_root)
    predictor = EagleCommunityPredictor(snapshots, scorer, threshold=args.threshold,
                                         pair_batch_size=args.pair_batch_size)
    if args.command == "query":
        predictor.prepare_time(args.t)
        community, stats = predictor.query(args.q, args.k, args.t)
        result = {"q": args.q, "k": args.k, "t": args.t, "target_t": args.t + 1,
                  "community": sorted(community), **stats, "threshold": args.threshold,
                  "scorer": scorer.metadata}
    else:
        require_fit_boundary(scorer.config.fit_end_t, len(snapshots))
        run_path = Path(args.config).parent / "run.json"
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        train_end_t, fit_end_t = training_boundaries(len(snapshots))
        if (run.get("split_rule") not in (
                "snapshot_55_15_30_v1", "snapshot_55_15_30_v1; equal times stay together")
                or run.get("train_end_t") != train_end_t
                or run.get("test_start_t") != fit_end_t):
            raise ValueError("EAGLE checkpoint lacks verified 7:3 snapshot training split")
        result = evaluate(predictor, slices_dir=args.slices_dir, start_t=args.start_t,
                          max_samples=args.max_samples)
        result["training_split"] = baseline_training_split(len(snapshots))
    text = json.dumps(result, indent=2)
    if args.command == "eval" and args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
