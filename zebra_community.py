#!/usr/bin/env python3
"""Predict next-snapshot k-core communities from Zebra link scores."""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import networkit as nk
import numpy as np
from scipy import sparse
import torch

from datasets.community_eval_builder import (
    sample_qk_coreness_weighted,
    set_metrics,
)
from datasets.dataset_builder import build_snapshots


ROOT = Path(__file__).resolve().parent
DEFAULT_ZEBRA_ROOT = ROOT.parent / "Zebra"
DEFAULT_SLICES = (
    ROOT / "data/mooc/time_slices/step_43200_window_86400"
)
DEFAULT_ZEBRA_DATASET = "mooc-snapshot"
DEFAULT_CHECKPOINT = (
    DEFAULT_ZEBRA_ROOT
    / "saved_checkpoints/mooc-snapshot-50-0.0001-streaming-"
    "[0.1, 0.1]-[0.5, 0.95]-20.pth"
)
DEFAULT_THRESHOLD = 0.5
DEFAULT_PAIR_BATCH_SIZE = 65536
CACHE_VERSION = 1

# Networkit 11.0.1 still looks up this NumPy alias when bulk-loading COO edges.
if "ulong" not in np.__dict__:
    np.ulong = np.uint64


def historical_community_union(snapshots, q, k, t):
    """Union q's connected k-core communities in snapshots 0 through t."""
    if not isinstance(t, int) or t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    if k <= 0:
        raise ValueError("k must be positive")
    candidate = set()
    for snapshot in snapshots[:t + 1]:
        k_info = snapshot.get("k_core_comps", {}).get(k)
        if k_info is None or q not in k_info["node_set"]:
            continue
        for component in k_info["components"]:
            if q in component:
                candidate.update(component)
                break
    return frozenset(candidate)


def _upper_triangle_batches(node_count, batch_size):
    left_parts = []
    right_parts = []
    buffered = 0
    for left in range(node_count - 1):
        right_start = left + 1
        while right_start < node_count:
            take = min(batch_size - buffered, node_count - right_start)
            left_parts.append(np.full(take, left, dtype=np.int64))
            right_parts.append(np.arange(
                right_start, right_start + take, dtype=np.int64
            ))
            buffered += take
            right_start += take
            if buffered == batch_size:
                yield np.concatenate(left_parts), np.concatenate(right_parts)
                left_parts = []
                right_parts = []
                buffered = 0
    if buffered:
        yield np.concatenate(left_parts), np.concatenate(right_parts)


@dataclass
class PredictedGraph:
    nodes: np.ndarray
    adjacency: sparse.csr_matrix

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        if self.adjacency.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("predicted adjacency shape does not match nodes")
        self.node_positions = {
            int(node): position for position, node in enumerate(self.nodes)
        }
        self._community_cache = {}

    @property
    def edge_count(self):
        return int(self.adjacency.nnz // 2)

    def community(self, candidate_nodes, q, k):
        """Return q's connected component in the induced predicted k-core."""
        candidate_nodes = sorted(set(candidate_nodes))
        if q not in candidate_nodes or k <= 0:
            return frozenset()
        cache_key = (tuple(candidate_nodes), k)
        cached = self._community_cache.get(cache_key)
        if cached is not None:
            return cached.get(q, frozenset())
        try:
            positions = np.fromiter(
                (self.node_positions[node] for node in candidate_nodes),
                dtype=np.int64,
                count=len(candidate_nodes),
            )
        except KeyError as error:
            raise ValueError(
                "candidate node is absent from the predicted graph"
            ) from error

        induced = sparse.triu(
            self.adjacency[positions][:, positions], k=1, format="coo"
        )
        graph = nk.GraphFromCoo(
            (
                induced.row.astype(np.int64, copy=False),
                induced.col.astype(np.int64, copy=False),
            ),
            n=len(candidate_nodes),
        )
        core_scores = nk.centrality.CoreDecomposition(graph).run().scores()

        communities = {}
        unseen = {
            position for position, score in enumerate(core_scores) if score >= k
        }
        while unseen:
            start = next(iter(unseen))
            visited = {start}
            unseen.remove(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in graph.iterNeighbors(node):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
            component = frozenset(
                candidate_nodes[position] for position in visited
            )
            for node in component:
                communities[node] = component
        self._community_cache[cache_key] = communities
        return communities.get(q, frozenset())


class ZebraCommunityPredictor:
    """Bridge coreness snapshots and a checkpoint-backed Zebra predictor."""

    def __init__(self, snapshots, zebra_predictor, node_mapping_path,
                 snapshot_mapping_path, threshold=DEFAULT_THRESHOLD,
                 pair_batch_size=DEFAULT_PAIR_BATCH_SIZE, cache_dir=None):
        self.snapshots = snapshots
        self.zebra = zebra_predictor
        self.threshold = float(threshold)
        self.pair_batch_size = int(pair_batch_size)
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
        node_mapping_path = Path(node_mapping_path)
        snapshot_mapping_path = Path(snapshot_mapping_path)
        self.original_to_zebra = self._load_node_mapping(node_mapping_path)
        self.time_to_zebra = self._load_snapshot_mapping(snapshot_mapping_path)
        self._validate_mappings()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_hash = self._hash_file(self.zebra.checkpoint_path)
        self.config_hash = hashlib.sha256(json.dumps(
            getattr(self.zebra, "config", {}), sort_keys=True
        ).encode("utf-8")).hexdigest()
        self.mapping_hash = hashlib.sha256(
            (
                self._hash_file(node_mapping_path)
                + self._hash_file(snapshot_mapping_path)
            ).encode("ascii")
        ).hexdigest()

    @staticmethod
    def _load_node_mapping(path):
        with Path(path).open(newline="") as mapping_file:
            reader = csv.DictReader(mapping_file)
            if reader.fieldnames != ["original_id", "zebra_id"]:
                raise ValueError("invalid Zebra node mapping header")
            return {
                int(row["original_id"]): int(row["zebra_id"])
                for row in reader
            }

    @staticmethod
    def _load_snapshot_mapping(path):
        with Path(path).open(newline="") as mapping_file:
            reader = csv.DictReader(mapping_file)
            required = {"zebra_ts", "slice_index"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("invalid Zebra snapshot mapping header")
            return {
                int(row["slice_index"]): int(row["zebra_ts"])
                for row in reader
            }

    def _validate_mappings(self):
        expected_times = set(range(len(self.snapshots)))
        if set(self.time_to_zebra) != expected_times:
            raise ValueError("snapshot mapping does not match coreness snapshots")
        mapped_times = [self.time_to_zebra[t] for t in range(len(self.snapshots))]
        if mapped_times != list(range(1, len(self.snapshots) + 1)):
            raise ValueError("Zebra snapshot timestamps must be consecutive from 1")
        snapshot_nodes = {
            node
            for snapshot in self.snapshots
            for node in snapshot.get("core_dict", {})
        }
        missing = snapshot_nodes - set(self.original_to_zebra)
        if missing:
            raise ValueError(
                "node mapping is missing {} snapshot nodes".format(len(missing))
            )

    @staticmethod
    def _hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @property
    def test_start_t(self):
        test_time = float(np.quantile(self.zebra.graph_df.ts, 0.85))
        eligible = [
            t for t, zebra_ts in self.time_to_zebra.items()
            if zebra_ts <= test_time
        ]
        if not eligible:
            raise ValueError("Zebra test boundary precedes all snapshots")
        return max(eligible)

    def candidate(self, q, k, t):
        return historical_community_union(self.snapshots, q, k, t)

    def _cache_path(self, original_nodes, t):
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(np.int64(CACHE_VERSION).tobytes())
        digest.update(self.checkpoint_hash.encode("ascii"))
        digest.update(self.config_hash.encode("ascii"))
        digest.update(self.mapping_hash.encode("ascii"))
        digest.update(np.float64(self.threshold).tobytes())
        digest.update(np.int64(t).tobytes())
        digest.update(np.asarray(original_nodes, dtype=np.int64).tobytes())
        return self.cache_dir / "{}.npz".format(digest.hexdigest())

    def predict_graph(self, original_nodes, t):
        """Predict all unordered edges over nodes using history through t."""
        if t < 0 or t >= len(self.snapshots) - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        original_nodes = np.asarray(sorted(set(original_nodes)), dtype=np.int64)
        cache_path = self._cache_path(original_nodes, t)
        if cache_path is not None and cache_path.is_file():
            return PredictedGraph(
                original_nodes,
                sparse.load_npz(str(cache_path)).tocsr(),
            )
        if len(original_nodes) == 0:
            adjacency = sparse.csr_matrix((0, 0), dtype=np.bool_)
            return PredictedGraph(original_nodes, adjacency)

        try:
            zebra_nodes = np.fromiter(
                (self.original_to_zebra[int(node)] for node in original_nodes),
                dtype=np.int32,
                count=len(original_nodes),
            )
        except KeyError as error:
            raise ValueError("candidate node is absent from node_mapping.csv") from error

        observed_timestamp = self.time_to_zebra[t]
        query_timestamp = self.time_to_zebra[t + 1]
        self.zebra.replay_until(observed_timestamp)
        embeddings = self.zebra.encode_nodes(zebra_nodes, query_timestamp)

        edge_left = []
        edge_right = []
        for left, right in _upper_triangle_batches(
            len(original_nodes), self.pair_batch_size
        ):
            left_index = torch.from_numpy(left).long().to(self.zebra.device)
            right_index = torch.from_numpy(right).long().to(self.zebra.device)
            probabilities = self.zebra.score_undirected_embeddings(
                embeddings[left_index], embeddings[right_index]
            )
            selected = probabilities.gt(self.threshold).cpu().numpy()
            if selected.any():
                edge_left.append(left[selected].astype(np.int32, copy=False))
                edge_right.append(right[selected].astype(np.int32, copy=False))

        if edge_left:
            left = np.concatenate(edge_left)
            right = np.concatenate(edge_right)
            rows = np.concatenate((left, right))
            columns = np.concatenate((right, left))
            adjacency = sparse.csr_matrix(
                (
                    np.ones(len(rows), dtype=np.bool_),
                    (rows, columns),
                ),
                shape=(len(original_nodes), len(original_nodes)),
            )
        else:
            adjacency = sparse.csr_matrix(
                (len(original_nodes), len(original_nodes)), dtype=np.bool_
            )
        adjacency.eliminate_zeros()
        if cache_path is not None:
            temporary_path = cache_path.with_suffix(".tmp.npz")
            sparse.save_npz(str(temporary_path), adjacency, compressed=True)
            temporary_path.replace(cache_path)
        return PredictedGraph(original_nodes, adjacency)

    def predict(self, q, k, t):
        candidate = self.candidate(q, k, t)
        graph = self.predict_graph(candidate, t)
        return graph.community(candidate, q, k)


def _load_zebra_predictor(zebra_root, dataset, checkpoint, device):
    zebra_root = str(Path(zebra_root).resolve())
    if zebra_root not in sys.path:
        sys.path.insert(0, zebra_root)
    from inference import ZebraLinkPredictor
    return ZebraLinkPredictor(dataset, checkpoint, device=device)


def _build_predictor(args):
    slices_dir = Path(args.slices_dir)
    snapshots, total_nodes, _kmax, _hmax = build_snapshots(slices_dir)
    zebra_data_dir = Path(args.zebra_root) / "data" / args.zebra_dataset
    zebra = _load_zebra_predictor(
        args.zebra_root, args.zebra_dataset, args.checkpoint, args.device
    )
    predictor = ZebraCommunityPredictor(
        snapshots,
        zebra,
        zebra_data_dir / "node_mapping.csv",
        zebra_data_dir / "snapshot_mapping.csv",
        threshold=args.threshold,
        pair_batch_size=args.pair_batch_size,
        cache_dir=args.cache_dir,
    )
    return predictor, total_nodes


def _query(args):
    predictor, _ = _build_predictor(args)
    candidate = predictor.candidate(args.q, args.k, args.t)
    started = time.time()
    graph = predictor.predict_graph(candidate, args.t)
    community = graph.community(candidate, args.q, args.k)
    result = {
        "q": args.q,
        "k": args.k,
        "t": args.t,
        "candidate_size": len(candidate),
        "predicted_edge_count": graph.edge_count,
        "community_size": len(community),
        "community": sorted(community),
        "elapsed_s": time.time() - started,
    }
    print(json.dumps(result, ensure_ascii=True))


def _aggregate(rows):
    result = {}
    for k, values in sorted(rows.items()):
        result[k] = {
            name: float(np.mean([row[name] for row in values]))
            for name in (
                "precision", "recall", "f1", "jaccard",
                "size_ratio", "pred_ratio", "elapsed_s",
            )
        }
        result[k]["samples"] = len(values)
    return result


def _evaluate(args):
    predictor, total_nodes = _build_predictor(args)
    start_t = predictor.test_start_t if args.start_t is None else args.start_t
    samples = sample_qk_coreness_weighted(
        predictor.snapshots,
        int(len(predictor.snapshots) * 0.7),
        [3, 4, 5, 6, 7],
        dataset_name="mooc",
        cache_dir=Path(args.slices_dir) / "sample_cache",
    )
    samples = [
        sample for sample in samples
        if sample["t"] >= start_t and sample["community"]
    ]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    samples_by_t = defaultdict(list)
    for sample in samples:
        sample["candidate"] = predictor.candidate(
            sample["query"], sample["k"], sample["t"]
        )
        samples_by_t[sample["t"]].append(sample)

    rows = defaultdict(list)
    wall_start = time.time()
    for t, time_samples in sorted(samples_by_t.items()):
        union_nodes = set().union(
            *(sample["candidate"] for sample in time_samples)
        )
        graph_start = time.time()
        graph = predictor.predict_graph(union_nodes, t)
        graph_elapsed = time.time() - graph_start
        print(
            "t={} samples={} nodes={} edges={} graph_s={:.3f}".format(
                t, len(time_samples), len(union_nodes), graph.edge_count,
                graph_elapsed,
            ),
            flush=True,
        )
        for sample in time_samples:
            started = time.time()
            prediction = graph.community(
                sample["candidate"], sample["query"], sample["k"]
            )
            metrics = set_metrics(prediction, sample["community"])
            rows[sample["k"]].append({
                **metrics,
                "size_ratio": len(prediction) / len(sample["community"]),
                "pred_ratio": len(prediction) / total_nodes * 100,
                "elapsed_s": time.time() - started,
            })

    per_k = _aggregate(rows)
    macro = {}
    if per_k:
        for name in (
            "precision", "recall", "f1", "jaccard",
            "size_ratio", "pred_ratio", "elapsed_s",
        ):
            macro[name] = float(np.mean([
                result[name] for result in per_k.values()
            ]))
    result = {
        "dataset": "mooc",
        "start_t": start_t,
        "threshold": predictor.threshold,
        "samples": len(samples),
        "per_k": per_k,
        "macro": macro,
        "wall_s": time.time() - wall_start,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def _add_common_arguments(parser):
    parser.add_argument("--slices-dir", default=str(DEFAULT_SLICES))
    parser.add_argument("--zebra-root", default=str(DEFAULT_ZEBRA_ROOT))
    parser.add_argument("--zebra-dataset", default=DEFAULT_ZEBRA_DATASET)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--pair-batch-size", type=int, default=DEFAULT_PAIR_BATCH_SIZE
    )
    parser.add_argument(
        "--cache-dir", default=str(ROOT / ".zebra_cache")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Zebra-driven next-snapshot k-core community prediction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    query_parser = subparsers.add_parser("query")
    _add_common_arguments(query_parser)
    query_parser.add_argument("q", type=int)
    query_parser.add_argument("k", type=int)
    query_parser.add_argument("t", type=int)
    query_parser.set_defaults(handler=_query)

    eval_parser = subparsers.add_parser("eval")
    _add_common_arguments(eval_parser)
    eval_parser.add_argument("--start-t", type=int, default=None)
    eval_parser.add_argument("--max-samples", type=int, default=None)
    eval_parser.add_argument("--output", default=None)
    eval_parser.set_defaults(handler=_evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
