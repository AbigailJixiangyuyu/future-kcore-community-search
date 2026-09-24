#!/usr/bin/env python3
"""Predict next-snapshot communities from checkpoint-backed TFWaveFormer links."""

import argparse
from collections import defaultdict
import json
import time
from pathlib import Path

import numpy as np

from datasets.community_eval_builder import set_metrics
from datasets.baseline_eval import (evaluation_start_t, load_test_samples,
                                    non_empty_samples, query_set_sha256)
from datasets.dataset_builder import build_snapshots, load_time_slice_manifest
from methods.tfwaveformer import DEFAULT_ROOT, SnapshotEdges, load_runtime
from community.baselines.baseline_graph import component_after_peeling


KS = (3, 4, 5, 6, 7)
DEFAULT_SLICES = Path(__file__).resolve().parents[2] / "data/mooc/time_slices/step_43200_window_86400"


class TFWaveFormerCommunityPredictor:
    """Incrementally index history; never inspect target snapshot during query."""

    def __init__(self, snapshots, link_predictor, *, threshold=0.5, batch_size=128):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.snapshots = snapshots
        self.link_predictor = link_predictor
        self.threshold = float(threshold)
        self.batch_size = batch_size
        self.prepared_t = -1
        self.communities = []
        self.adjacency = defaultdict(set)

    def prepare_time(self, t):
        if not isinstance(t, int) or not 0 <= t < len(self.snapshots) - 1:
            raise IndexError("prediction requires t and a next-snapshot slot")
        if t < self.prepared_t:
            raise ValueError("cannot move history backwards")
        # Verify fitted history before exposing newly indexed candidates.
        self.link_predictor.prepare_time(t)
        for step in range(self.prepared_t + 1, t + 1):
            snapshot = self.snapshots[step]
            self.communities.append(snapshot["k_core_comps"])
            for u, v, *_ in snapshot["edge_list"]:
                u, v = int(u), int(v)
                if u != v:
                    self.adjacency[u].add(v)
                    self.adjacency[v].add(u)
        self.prepared_t = t

    def candidate(self, q, k):
        nodes = set()
        for by_k in self.communities:
            info = by_k.get(k)
            if info is not None and q in info["node_set"]:
                for component in info["components"]:
                    if q in component:
                        nodes.update(component)
                        break
        return frozenset(nodes)

    def predict(self, q, k, t):
        if k not in KS:
            raise ValueError("k must be in 3..7")
        if t != self.prepared_t:
            raise ValueError("prepare_time(t) before querying")
        started = time.perf_counter()
        candidate = self.candidate(q, k)
        candidate_s = time.perf_counter() - started
        accepted = []
        batch = []
        pair_count = 0

        def score_batch():
            if batch:
                scores = self.link_predictor.score_edges(
                    np.asarray(batch, dtype=np.int64), target_t=t + 1,
                    batch_size=self.batch_size,
                )
                accepted.extend(pair for pair, score in zip(batch, scores)
                                if score >= self.threshold)

        for u in sorted(candidate):
            for v in sorted(self.adjacency.get(u, ()) & candidate):
                if u < v:
                    batch.append((u, v))
                    pair_count += 1
                    if len(batch) == self.batch_size:
                        score_batch()
                        batch = []
        score_batch()
        score_s = time.perf_counter() - started - candidate_s
        community = component_after_peeling(candidate, accepted, q, k)
        elapsed = time.perf_counter() - started
        return {
            "q": int(q), "k": k, "t": t, "target_t": t + 1,
            "community": sorted(community), "community_size": len(community),
            "candidate_size": len(candidate), "pair_count": pair_count,
            "predicted_edge_count": len(accepted),
            "candidate_search_s": candidate_s, "edge_prediction_s": score_s,
            "community_search_s": elapsed - candidate_s - score_s,
            "query_s": elapsed,
        }


def evaluate(predictor, samples, *, start_t, max_samples=None):
    """Offline metrics: only this function may access next-snapshot truth."""
    if not 0 <= start_t < len(predictor.snapshots) - 1:
        raise ValueError("start_t must have a next snapshot")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    selected = non_empty_samples(
        s for s in samples if s["t"] >= start_t and s["k"] in KS)
    if max_samples is not None:
        selected = selected[:max_samples]
    rows = defaultdict(list)
    per_sample = []
    for sample in sorted(selected, key=lambda s: s["t"]):
        t = sample["t"]
        predictor.prepare_time(t)
        result = predictor.predict(sample["query"], sample["k"], t)
        metrics = set_metrics(result["community"], sample["community"])
        record = {**result, **metrics, "truth_size": len(sample["community"])}
        per_sample.append(record)
        rows[sample["k"]].append(metrics)
    names = ("precision", "recall", "f1", "jaccard")
    per_k = {str(k): dict(
        {name: float(np.mean([row[name] for row in group])) for name in names},
        samples=len(group),
    ) for k, group in sorted(rows.items())}
    return {
        "samples": len(per_sample), "start_t": start_t,
        "sample_scope": "shared_community_eval_non_empty_only",
        "query_set_sha256": query_set_sha256(selected),
        "candidate_protocol": "historical_q_k_community_union_and_historical_edges",
        "threshold": predictor.threshold, "per_k": per_k,
        "macro": {name: float(np.mean([row[name] for row in per_k.values()]))
                  for name in names} if per_k else {},
        "records": per_sample,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("query", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES)
        sub.add_argument("--checkpoint", type=Path, required=True,
                         help="trusted snapshot checkpoint, not legacy event checkpoint")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--tfwaveformer-root", type=Path, default=DEFAULT_ROOT)
        sub.add_argument("--threshold", type=float, default=0.5,
                         help="explicit link-score threshold; 0.5 is not calibrated")
        sub.add_argument("--batch-size", type=int, default=128)
        sub.add_argument("--output", type=Path, help="new JSON output path")
        if command == "query":
            sub.add_argument("q", type=int)
            sub.add_argument("k", type=int, choices=KS)
            sub.add_argument("t", type=int)
        else:
            sub.add_argument("--start-t", type=int)
            sub.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("--output already exists")
    snapshots, _, _, _ = build_snapshots(args.slices_dir)
    runtime = load_runtime(args.tfwaveformer_root)
    link = runtime.TFWaveFormerSnapshotPredictor(
        SnapshotEdges(snapshots), checkpoint=args.checkpoint, device=args.device,
    )
    predictor = TFWaveFormerCommunityPredictor(
        snapshots, link, threshold=args.threshold, batch_size=args.batch_size,
    )
    if args.command == "query":
        predictor.prepare_time(args.t)
        result = predictor.predict(args.q, args.k, args.t)
    else:
        manifest = load_time_slice_manifest(args.slices_dir)
        start_t = (evaluation_start_t(len(snapshots))
                   if args.start_t is None else args.start_t)
        samples = load_test_samples(args.slices_dir, len(snapshots))
        result = evaluate(predictor, samples, start_t=start_t,
                          max_samples=args.max_samples)
        result["dataset"] = manifest["dataset"]
    result.update(checkpoint=str(args.checkpoint.resolve()),
                  slices_dir=str(args.slices_dir.resolve()),
                  threshold_calibrated=False,
                  candidate_protocol="historical_q_k_community_union_and_historical_edges")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
