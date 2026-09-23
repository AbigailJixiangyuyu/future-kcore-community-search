#!/usr/bin/env python3
"""Predict next-snapshot k-core communities from Zebra link scores."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkit as nk
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F

from datasets.community_eval_builder import (
    sample_qk_coreness_weighted,
    set_metrics,
)
from datasets.dataset_builder import build_snapshots, load_time_slice_manifest
from datasets.baseline_eval import (baseline_training_split, evaluation_start_t,
                                    load_test_samples, non_empty_samples,
                                    query_set_sha256, require_fit_boundary)
from methods.zebra_history_cache import ZebraHistoryCache


ROOT = Path(__file__).resolve().parent
DEFAULT_ZEBRA_ROOT = ROOT / "third_party" / "Zebra"
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
CACHE_VERSION = 2
EMBEDDING_CACHE_VERSION = 1
PROGRESSIVE_RESULT_VERSION = 2
PROGRESSIVE_KS = (7, 6, 5, 4, 3)
COMMUNITY_METRICS = (
    "precision",
    "recall",
    "f1",
    "jaccard",
    "size_ratio",
    "pred_ratio",
    "elapsed_s",
)
TIMING_SCHEMA = {
    "timing_version": 2,
    "timing_clock": "perf_counter",
    "elapsed_scope": "candidate_search_encoding_edge_prediction_and_community",
    "prediction_total_scope": "prepare_plus_query_excluding_load_and_evaluation",
    "wall_scope": "entry_to_result_ready_excluding_serialization_and_output",
    "prediction_disk_cache": False,
    "cross_query_cache": False,
    "initial_state_policy": "load_exact_boundary_cache_else_replay_and_save",
}


class TimedZebraSession:
    """Independent queries over shared, prepared historical state."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.t = None

    def now(self):
        device = torch.device(self.predictor.zebra.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def prepare(self, t):
        predictor = self.predictor
        if not 0 <= t < len(predictor.snapshots) - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        started = self.now()
        reused = t == self.t
        initial = self.t is None
        cache_hit = False
        cache = getattr(predictor, "history_cache", None)
        cache_load_s = 0.0
        cache_save_s = 0.0
        if not reused:
            # Persist only the first requested boundary, not every incremental slice.
            if initial and cache is not None:
                cache_started = self.now()
                cache_hit = cache.load(t)
                cache_load_s = self.now() - cache_started
            if not cache_hit:
                predictor.zebra.replay_until(predictor.time_to_zebra[t])
                if initial and cache is not None:
                    cache_started = self.now()
                    cache.save(t)
                    cache_save_s = self.now() - cache_started
            self.t = t
        elapsed = self.now() - started
        return {
            "t": t, "prepare_s": elapsed,
            "state_update_s": 0.0 if reused else elapsed,
            "context_cache_hit": reused, "initial_prepare": initial,
            "state_cache_enabled": cache is not None,
            "state_cache_hit": cache_hit,
            "state_cache_load_s": cache_load_s,
            "state_cache_save_s": cache_save_s,
        }

    def query(self, q, k, t):
        if t != self.t:
            raise ValueError("prepare the requested time before querying")
        predictor = self.predictor
        started = self.now()
        candidate = predictor.candidate(q, k, t)
        nodes = np.asarray(sorted(candidate), dtype=np.int64)
        candidate_end = self.now()
        if len(nodes):
            embeddings = predictor.zebra.encode_nodes(
                predictor._mapped_nodes(nodes), predictor.time_to_zebra[t + 1]
            )
        encode_end = self.now()
        if len(nodes):
            graph = predictor._decode_graph(nodes, embeddings, t)
        else:
            graph = PredictedGraph(nodes, sparse.csr_matrix((0, 0), dtype=np.bool_))
        graph_end = self.now()
        community = graph.community(candidate, q, k)
        ended = self.now()
        return community, graph, {
            "candidate_size": len(candidate),
            "newly_encoded_node_count": len(nodes),
            "candidate_search_s": candidate_end - started,
            "node_encoding_s": encode_end - candidate_end,
            "edge_prediction_s": graph_end - encode_end,
            "community_search_s": ended - graph_end,
            "query_s": ended - started,
            "elapsed_s": ended - started,
        }

# Networkit 11.0.1 still looks up this NumPy alias when bulk-loading COO edges.
if "ulong" not in np.__dict__:
    np.ulong = np.uint64


def historical_community_union(snapshots, q, k, t):
    """Union q's connected k-core communities in snapshots 0..t."""
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


class HistoricalEdgeIndex:
    """Index first appearance; never expose an edge first seen after t."""

    def __init__(self, snapshots):
        self.snapshot_count = len(snapshots)
        self.adjacency = defaultdict(dict)
        digest = hashlib.sha256()
        for t, snapshot in enumerate(snapshots):
            for u, v, *_ in snapshot["edge_list"]:
                u, v = sorted((int(u), int(v)))
                if u != v and v not in self.adjacency[u]:
                    self.adjacency[u][v] = t
        for u in sorted(self.adjacency):
            for v, first_t in sorted(self.adjacency[u].items()):
                digest.update(np.asarray((u, v, first_t), dtype=np.int64).tobytes())
        self.signature = digest.hexdigest()

    def batches(self, original_nodes, t, batch_size=DEFAULT_PAIR_BATCH_SIZE):
        if not 0 <= t < self.snapshot_count - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        positions = {int(node): i for i, node in enumerate(original_nodes)}
        if len(positions) != len(original_nodes):
            raise ValueError("candidate nodes must be unique")
        left, right = [], []
        for u, i in positions.items():
            for v, first_t in self.adjacency.get(u, {}).items():
                if first_t <= t and v in positions:
                    left.append(i)
                    right.append(positions[v])
                    if len(left) == batch_size:
                        yield np.asarray(left), np.asarray(right)
                        left, right = [], []
        if left:
            yield np.asarray(left), np.asarray(right)

    def count(self, original_nodes, t):
        return sum(len(left) for left, _ in self.batches(original_nodes, t))


class _ProjectedUndirectedDecoder:
    """Exactly reuse the node-wise terms of Zebra's first decoder layer."""

    def __init__(self, zebra, embeddings):
        with torch.no_grad():
            affinity = zebra.model.affinity_score
            embedding_dim = embeddings.shape[1]
            if affinity.fc1.in_features != 2 * embedding_dim:
                raise ValueError("Zebra decoder input does not match embeddings")
            first_weight = affinity.fc1.weight
            self.left_projection = F.linear(
                embeddings,
                first_weight[:, :embedding_dim],
                affinity.fc1.bias,
            )
            self.right_projection = F.linear(
                embeddings,
                first_weight[:, embedding_dim:],
            )
        self.output_layer = affinity.fc2

    def _directed(self, left, right):
        hidden = F.relu(
            self.left_projection[left] + self.right_projection[right]
        )
        return self.output_layer(hidden).squeeze(dim=1).sigmoid()

    def score(self, left, right):
        with torch.no_grad():
            forward = self._directed(left, right)
            reverse = self._directed(right, left)
            return (forward + reverse) / 2


def _build_projected_decoder(zebra, embeddings):
    model = getattr(zebra, "model", None)
    affinity = getattr(model, "affinity_score", None)
    if affinity is None or not hasattr(affinity, "fc1") or not hasattr(
        affinity, "fc2"
    ):
        return None
    return _ProjectedUndirectedDecoder(zebra, embeddings)


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(
        "Object of type {} is not JSON serializable".format(
            value.__class__.__name__
        )
    )


def _atomic_write_json(path, payload):
    _atomic_write_text(
        path,
        json.dumps(
            payload, default=_json_default, indent=2, sort_keys=True
        ) + "\n",
    )


def _atomic_save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as output:
        np.save(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(path)


def _sample_identifier(sample):
    return "{}:{}:{}".format(
        int(sample["t"]), int(sample["k"]), int(sample["query"])
    )


def _prepare_progressive_samples(samples, snapshots, start_t, end_t,
                                 ks=PROGRESSIVE_KS, edge_index=None):
    edge_index = edge_index or HistoricalEdgeIndex(snapshots)
    k_order = {k: position for position, k in enumerate(ks)}
    prepared = []
    identifiers = set()
    for sample in samples:
        t = int(sample["t"])
        k = int(sample["k"])
        if not sample["community"] or t < start_t or t > end_t or k not in k_order:
            continue
        candidate = historical_community_union(
            snapshots, int(sample["query"]), k, t
        )
        identifier = _sample_identifier(sample)
        if identifier in identifiers:
            raise ValueError("duplicate progressive sample: {}".format(identifier))
        identifiers.add(identifier)
        node_count = len(candidate)
        prepared.append({
            **sample,
            "sample_id": identifier,
            "candidate": candidate,
            "candidate_size": node_count,
            "pair_count": edge_index.count(sorted(candidate), t),
            "all_pair_count": node_count * (node_count - 1) // 2,
        })
    return sorted(
        prepared,
        key=lambda sample: (
            k_order[sample["k"]],
            sample["candidate_size"],
            sample["t"],
            sample["query"],
        ),
    )


def _progressive_run_signature(predictor, samples, start_t, end_t):
    digest = hashlib.sha256()
    digest.update(np.int64(PROGRESSIVE_RESULT_VERSION).tobytes())
    digest.update(np.int64(EMBEDDING_CACHE_VERSION).tobytes())
    digest.update(predictor.checkpoint_hash.encode("ascii"))
    digest.update(predictor.config_hash.encode("ascii"))
    digest.update(predictor.mapping_hash.encode("ascii"))
    digest.update(np.float32(predictor.threshold).tobytes())
    digest.update(json.dumps(predictor.candidate_metadata,
                             sort_keys=True).encode("ascii"))
    digest.update(predictor.edge_index.signature.encode("ascii"))
    digest.update(np.int64(start_t).tobytes())
    digest.update(np.int64(end_t).tobytes())
    for sample in samples:
        digest.update(sample["sample_id"].encode("ascii"))
        digest.update(np.asarray(
            sorted(sample["candidate"]), dtype=np.int64
        ).tobytes())
    return digest.hexdigest()


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
        self.edge_index = HistoricalEdgeIndex(snapshots)
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

    @property
    def candidate_metadata(self):
        return {
            "candidate_history": "all_snapshots_through_t",
            "edge_candidates": "historical_edges_only",
        }

    def candidate(self, q, k, t):
        return historical_community_union(
            self.snapshots, q, k, t
        )

    def _cache_path(self, original_nodes, t):
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(np.int64(CACHE_VERSION).tobytes())
        digest.update(self.edge_index.signature.encode("ascii"))
        digest.update(self.checkpoint_hash.encode("ascii"))
        digest.update(self.config_hash.encode("ascii"))
        digest.update(self.mapping_hash.encode("ascii"))
        digest.update(np.float32(self.threshold).tobytes())
        digest.update(np.int64(t).tobytes())
        digest.update(np.asarray(original_nodes, dtype=np.int64).tobytes())
        return self.cache_dir / "{}.npz".format(digest.hexdigest())

    def _mapped_nodes(self, original_nodes):
        try:
            return np.fromiter(
                (self.original_to_zebra[int(node)] for node in original_nodes),
                dtype=np.int32,
                count=len(original_nodes),
            )
        except KeyError as error:
            raise ValueError(
                "candidate node is absent from node_mapping.csv"
            ) from error

    def encode_original_nodes(self, original_nodes, t):
        if t < 0 or t >= len(self.snapshots) - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        original_nodes = np.asarray(
            sorted(set(original_nodes)), dtype=np.int64
        )
        if len(original_nodes) == 0:
            hidden_dim = self.zebra.config["node_dim"] * (
                len(self.zebra.config["alpha_list"]) + 1
            )
            return original_nodes, np.empty(
                (0, hidden_dim), dtype=np.float32
            )
        zebra_nodes = self._mapped_nodes(original_nodes)
        self.zebra.replay_until(self.time_to_zebra[t])
        embeddings = self.zebra.encode_nodes(
            zebra_nodes, self.time_to_zebra[t + 1]
        )
        return original_nodes, embeddings.detach().cpu().numpy().astype(
            np.float32, copy=False
        )

    def predict_graph_from_embeddings(self, original_nodes, embeddings, t,
                                      progress_callback=None):
        original_nodes = np.asarray(original_nodes, dtype=np.int64)
        embeddings = torch.as_tensor(
            np.asarray(embeddings), dtype=torch.float32, device=self.zebra.device
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(original_nodes):
            raise ValueError("cached embeddings do not match candidate nodes")
        return self._decode_graph(
            original_nodes, embeddings, t, progress_callback=progress_callback
        )

    def _decode_graph(self, original_nodes, embeddings, t,
                      progress_callback=None, edge_decisions=None):
        total_pairs = self.edge_index.count(original_nodes, t)
        if total_pairs == 0:
            if progress_callback is not None:
                progress_callback(0, 0)
            adjacency = sparse.csr_matrix(
                (len(original_nodes), len(original_nodes)), dtype=np.bool_
            )
            return PredictedGraph(original_nodes, adjacency)

        projected_decoder = None
        decoder_initialized = False
        edge_left = []
        edge_right = []
        completed_pairs = 0
        for left, right in self.edge_index.batches(
            original_nodes, t, self.pair_batch_size
        ):
            keys = [
                (int(original_nodes[u]), int(original_nodes[v]))
                for u, v in zip(left, right)
            ] if edge_decisions is not None else None
            missing = np.asarray([
                i for i, key in enumerate(keys) if key not in edge_decisions
            ], dtype=np.int64) if keys is not None else np.arange(len(left))
            selected = np.zeros(len(left), dtype=np.bool_)
            if len(missing):
                if not decoder_initialized:
                    projected_decoder = _build_projected_decoder(self.zebra, embeddings)
                    decoder_initialized = True
                left_index = torch.from_numpy(left[missing]).long().to(self.zebra.device)
                right_index = torch.from_numpy(right[missing]).long().to(self.zebra.device)
                if projected_decoder is None:
                    probabilities = self.zebra.score_undirected_embeddings(
                        embeddings[left_index], embeddings[right_index]
                    )
                else:
                    probabilities = projected_decoder.score(left_index, right_index)
                selected[missing] = probabilities.gt(self.threshold).cpu().numpy()
                if keys is not None:
                    for i in missing:
                        edge_decisions[keys[i]] = bool(selected[i])
            if keys is not None:
                selected = np.asarray([edge_decisions[key] for key in keys], dtype=np.bool_)
            if selected.any():
                edge_left.append(left[selected].astype(np.int32, copy=False))
                edge_right.append(right[selected].astype(np.int32, copy=False))
            completed_pairs += len(left)
            if progress_callback is not None:
                progress_callback(completed_pairs, total_pairs)

        if completed_pairs != total_pairs:
            raise RuntimeError(
                "decoded {} of {} unordered node pairs".format(
                    completed_pairs, total_pairs
                )
            )
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
        return PredictedGraph(original_nodes, adjacency)

    def predict_graph(self, original_nodes, t):
        """Score only historical edges over candidate nodes, through t."""
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
        zebra_nodes = self._mapped_nodes(original_nodes)
        self.zebra.replay_until(self.time_to_zebra[t])
        embeddings = self.zebra.encode_nodes(
            zebra_nodes, self.time_to_zebra[t + 1]
        )
        graph = self._decode_graph(original_nodes, embeddings, t)
        if cache_path is not None:
            temporary_path = cache_path.with_suffix(".tmp.npz")
            sparse.save_npz(
                str(temporary_path), graph.adjacency, compressed=True
            )
            temporary_path.replace(cache_path)
        return graph

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
    state_cache_dir = getattr(args, "state_cache_dir", None)
    if state_cache_dir is None:
        state_cache_dir = slices_dir / "model_cache" / "zebra_state"
    predictor.history_cache = (
        ZebraHistoryCache(predictor, state_cache_dir, args.zebra_root)
        if state_cache_dir else None
    )
    return predictor, total_nodes


def _query(args):
    wall_start = time.perf_counter()
    predictor, _ = _build_predictor(args)
    session = TimedZebraSession(predictor)
    load_s = session.now() - wall_start
    preparation = session.prepare(args.t)
    community, graph, timing = session.query(args.q, args.k, args.t)
    result = {
        "q": args.q,
        "k": args.k,
        "t": args.t,
        **TIMING_SCHEMA,
        **preparation,
        **timing,
        "load_s": load_s,
        "prediction_total_s": preparation["prepare_s"] + timing["query_s"],
        **predictor.candidate_metadata,
        "predicted_edge_count": graph.edge_count,
        "community_size": len(community),
        "community": sorted(community),
    }
    result["wall_s"] = session.now() - wall_start
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


def _sample_result_path(work_dir, sample):
    return Path(work_dir) / "samples" / "t{:06d}_k{}_q{}.json".format(
        int(sample["t"]), int(sample["k"]), int(sample["query"])
    )


def _load_completed_records(work_dir, samples, run_signature):
    records = {}
    for sample in samples:
        result_path = _sample_result_path(work_dir, sample)
        if not result_path.is_file():
            continue
        try:
            record = json.loads(result_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            record.get("run_signature") != run_signature
            or record.get("sample_id") != sample["sample_id"]
            or int(record.get("pair_count", -1)) != sample["pair_count"]
        ):
            continue
        records[sample["sample_id"]] = record
    return records


def _aggregate_completed_records(records):
    result = {}
    for name in COMMUNITY_METRICS:
        result[name] = float(np.mean([record[name] for record in records], dtype=np.float32))
    result["samples"] = len(records)
    result["predicted_edge_count"] = int(sum(
        record["predicted_edge_count"] for record in records
    ))
    result["scored_pair_count"] = int(sum(
        record["pair_count"] for record in records
    ))
    return result


def _build_progress_payload(dataset_name, predictor, samples, records,
                            run_signature, start_t, end_t, started_at,
                            current=None, phase="scoring"):
    current = current or {}
    per_k = {}
    completed_metrics = []
    for k in PROGRESSIVE_KS:
        k_samples = [sample for sample in samples if sample["k"] == k]
        k_records = [
            records[sample["sample_id"]]
            for sample in k_samples
            if sample["sample_id"] in records
        ]
        complete = len(k_records) == len(k_samples)
        status = "complete" if complete else "pending"
        if current.get("k") == k and not complete:
            status = "running"
        metrics = (
            _aggregate_completed_records(k_records)
            if complete and k_samples
            else {name: "INF" for name in COMMUNITY_METRICS}
        )
        if complete and k_samples:
            completed_metrics.append(metrics)
        current_pairs = (
            int(current.get("sample_pairs_completed", 0))
            if (
                current.get("k") == k
                and current.get("sample_id") not in records
            ) else 0
        )
        per_k[str(k)] = {
            "status": status,
            "samples_total": len(k_samples),
            "samples_completed": len(k_records),
            "pairs_total": int(sum(
                sample["pair_count"] for sample in k_samples
            )),
            "pairs_completed": int(sum(
                record["pair_count"] for record in k_records
            )) + current_pairs,
            "metrics": metrics,
        }

    macro = {name: "INF" for name in COMMUNITY_METRICS}
    if len(completed_metrics) == len(PROGRESSIVE_KS):
        macro = {
            name: float(np.mean([
                metrics[name] for metrics in completed_metrics
            ], dtype=np.float32))
            for name in COMMUNITY_METRICS
        }
    return {
        "version": PROGRESSIVE_RESULT_VERSION,
        "timing_version": 1,
        "timing_comparison_eligible": False,
        "timing_note": "Legacy cached-embedding workflow; use eval for online timing.",
        "dataset": dataset_name,
        "run_signature": run_signature,
        "checkpoint": str(Path(predictor.zebra.checkpoint_path).resolve()),
        "threshold": predictor.threshold,
        **predictor.candidate_metadata,
        "pair_batch_size": predictor.pair_batch_size,
        "sample_scope": {
            "start_t": start_t,
            "end_t": end_t,
            "non_empty_only": True,
            "validation_leakage_warning": start_t < predictor.test_start_t,
            "strict_zebra_test_start_t": predictor.test_start_t,
            "samples": len(samples),
        },
        "execution_order": {
            "k": list(PROGRESSIVE_KS),
            "within_k": "candidate_size_ascending",
            "all_unordered_pairs": True,
        },
        "phase": phase,
        "started_at": started_at,
        "updated_at": _now(),
        "samples_completed": len(records),
        "current": current,
        "per_k": per_k,
        "macro": macro,
    }


def _format_metric(value):
    if value == "INF":
        return "INF"
    return "{:.6f}".format(float(value))


def _write_comparison_markdown(path, zebra_payload, hybrid_result_path=None):
    hybrid = None
    if hybrid_result_path and Path(hybrid_result_path).is_file():
        try:
            hybrid = json.loads(Path(hybrid_result_path).read_text())
        except (OSError, ValueError):
            hybrid = None
    lines = [
        "# WikiTalk Community Prediction Evaluation",
        "",
        "- Samples: non-empty ground-truth communities, t={}..{}.".format(
            zebra_payload["sample_scope"]["start_t"],
            zebra_payload["sample_scope"]["end_t"],
        ),
        "- Zebra order: k=7,6,5,4,3; candidate size ascending.",
        "- `INF` means that the complete k-level evaluation has not finished.",
        "- The selected range overlaps Zebra validation data before t={}.".format(
            zebra_payload["sample_scope"]["strict_zebra_test_start_t"]
        ),
        "",
        "| Method | k | Status | Samples | Precision | Recall | F1 | Jaccard |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("Hybrid", "Zebra"):
        for k in PROGRESSIVE_KS:
            if method == "Zebra":
                row = zebra_payload["per_k"][str(k)]
                metrics = row["metrics"]
                status = row["status"]
                samples = "{}/{}".format(
                    row["samples_completed"], row["samples_total"]
                )
            elif hybrid and str(k) in hybrid.get("per_k", {}):
                metrics = hybrid["per_k"][str(k)]
                status = "complete"
                samples = str(metrics["samples"])
            elif hybrid and k in hybrid.get("per_k", {}):
                metrics = hybrid["per_k"][k]
                status = "complete"
                samples = str(metrics["samples"])
            else:
                metrics = {name: "INF" for name in COMMUNITY_METRICS}
                status = "running" if method == "Hybrid" else "pending"
                samples = "0"
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    method,
                    k,
                    status,
                    samples,
                    _format_metric(metrics["precision"]),
                    _format_metric(metrics["recall"]),
                    _format_metric(metrics["f1"]),
                    _format_metric(metrics["jaccard"]),
                )
            )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _embedding_paths(embedding_dir, t):
    prefix = Path(embedding_dir) / "t{:06d}".format(t)
    return (
        prefix.with_name(prefix.name + "_nodes.npy"),
        prefix.with_name(prefix.name + "_embeddings.npy"),
    )


def _valid_embedding_cache(nodes_path, embeddings_path, expected_nodes):
    if not nodes_path.is_file() or not embeddings_path.is_file():
        return False
    try:
        nodes = np.load(nodes_path, mmap_mode="r", allow_pickle=False)
        embeddings = np.load(
            embeddings_path, mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError):
        return False
    return (
        nodes.dtype == np.int64
        and nodes.shape == expected_nodes.shape
        and np.array_equal(nodes, expected_nodes)
        and embeddings.dtype == np.float32
        and embeddings.ndim == 2
        and embeddings.shape[0] == len(nodes)
    )


def _prepare_embedding_cache(predictor, samples, embedding_dir,
                             status_callback=None):
    nodes_by_time = defaultdict(set)
    for sample in samples:
        nodes_by_time[sample["t"]].update(sample["candidate"])
    expected = {
        t: np.asarray(sorted(nodes), dtype=np.int64)
        for t, nodes in nodes_by_time.items()
    }
    missing_times = []
    for t, nodes in sorted(expected.items()):
        nodes_path, embeddings_path = _embedding_paths(embedding_dir, t)
        if not _valid_embedding_cache(nodes_path, embeddings_path, nodes):
            missing_times.append(t)
    if not missing_times:
        return expected

    missing = set(missing_times)
    for index, (t, nodes) in enumerate(sorted(expected.items()), start=1):
        if status_callback:
            status_callback({
                "embedding_time": t,
                "embedding_times_completed": index - 1,
                "embedding_times_total": len(expected),
                "embedding_nodes": len(nodes),
            })
        predictor.zebra.replay_until(predictor.time_to_zebra[t])
        if t not in missing:
            continue
        zebra_nodes = predictor._mapped_nodes(nodes)
        embeddings = predictor.zebra.encode_nodes(
            zebra_nodes, predictor.time_to_zebra[t + 1]
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        nodes_path, embeddings_path = _embedding_paths(embedding_dir, t)
        _atomic_save_npy(nodes_path, nodes)
        _atomic_save_npy(embeddings_path, embeddings)
    return expected


def _load_candidate_embeddings(embedding_dir, sample):
    nodes_path, embeddings_path = _embedding_paths(
        embedding_dir, sample["t"]
    )
    all_nodes = np.load(nodes_path, mmap_mode="r", allow_pickle=False)
    all_embeddings = np.load(
        embeddings_path, mmap_mode="r", allow_pickle=False
    )
    candidate_nodes = np.asarray(
        sorted(sample["candidate"]), dtype=np.int64
    )
    positions = np.searchsorted(all_nodes, candidate_nodes)
    if (
        len(positions)
        and (
            positions[-1] >= len(all_nodes)
            or not np.array_equal(all_nodes[positions], candidate_nodes)
        )
    ):
        raise ValueError("candidate nodes are missing from embedding cache")
    return candidate_nodes, np.asarray(all_embeddings[positions])


def _progressive_evaluate(args):
    predictor, total_nodes = _build_predictor(args)
    manifest = load_time_slice_manifest(args.slices_dir)
    dataset_name = manifest["dataset"]
    split_t = int(len(predictor.snapshots) * 0.7)
    end_t = (
        len(predictor.snapshots) - 2
        if args.end_t is None else args.end_t
    )
    raw_samples = sample_qk_coreness_weighted(
        predictor.snapshots,
        split_t,
        [3, 4, 5, 6, 7],
        dataset_name=dataset_name,
        cache_dir=Path(args.slices_dir) / "sample_cache",
    )
    samples = _prepare_progressive_samples(
        raw_samples, predictor.snapshots, args.start_t, end_t,
        edge_index=predictor.edge_index,
    )
    if not samples:
        raise ValueError("no non-empty community samples in the selected range")

    work_dir = Path(args.work_dir)
    progress_path = Path(args.output)
    comparison_path = Path(args.comparison_output)
    work_dir.mkdir(parents=True, exist_ok=True)
    run_signature = _progressive_run_signature(
        predictor, samples, args.start_t, end_t
    )
    run_metadata_path = work_dir / "run.json"
    if run_metadata_path.is_file():
        existing = json.loads(run_metadata_path.read_text())
        if existing.get("run_signature") != run_signature:
            raise ValueError(
                "work directory belongs to a different progressive run"
            )
        if not args.resume:
            raise FileExistsError(
                "progressive run already exists; pass --resume to continue"
            )
    else:
        _atomic_write_json(run_metadata_path, {
            "run_signature": run_signature,
            "created_at": _now(),
            "dataset": dataset_name,
        })

    embedding_dir = Path(args.embedding_cache_dir) / run_signature[:16]
    embedding_dir.mkdir(parents=True, exist_ok=True)
    records = _load_completed_records(work_dir, samples, run_signature)
    started_at = _now()
    if progress_path.is_file():
        try:
            previous_progress = json.loads(progress_path.read_text())
            if previous_progress.get("run_signature") == run_signature:
                started_at = previous_progress.get("started_at", started_at)
        except (OSError, ValueError):
            pass
    last_status_update = [0.0]

    def publish(current=None, phase="scoring", force=False):
        now = time.time()
        if not force and now - last_status_update[0] < args.status_interval:
            return
        payload = _build_progress_payload(
            dataset_name,
            predictor,
            samples,
            records,
            run_signature,
            args.start_t,
            end_t,
            started_at,
            current=current,
            phase=phase,
        )
        _atomic_write_json(progress_path, payload)
        _write_comparison_markdown(
            comparison_path, payload, args.hybrid_result
        )
        last_status_update[0] = now

    publish(phase="embedding_cache", force=True)

    def embedding_status(current):
        publish(current=current, phase="embedding_cache")

    _prepare_embedding_cache(
        predictor, samples, embedding_dir, embedding_status
    )
    publish(phase="scoring", force=True)

    for sample_index, sample in enumerate(samples, start=1):
        if sample["sample_id"] in records:
            continue
        sample_started = time.time()
        current = {
            "sample_index": sample_index,
            "sample_total": len(samples),
            "sample_id": sample["sample_id"],
            "q": int(sample["query"]),
            "k": int(sample["k"]),
            "t": int(sample["t"]),
            "candidate_size": sample["candidate_size"],
            "sample_pair_count": sample["pair_count"],
            "sample_pairs_completed": 0,
        }

        def pair_status(completed_pairs, total_pairs):
            current["sample_pairs_completed"] = int(completed_pairs)
            current["sample_pair_count"] = int(total_pairs)
            publish(current=current, phase="scoring")

        candidate_nodes, embeddings = _load_candidate_embeddings(
            embedding_dir, sample
        )
        graph = predictor.predict_graph_from_embeddings(
            candidate_nodes, embeddings, int(sample["t"]),
            progress_callback=pair_status
        )
        prediction = graph.community(
            candidate_nodes, int(sample["query"]), int(sample["k"])
        )
        metrics = set_metrics(prediction, sample["community"])
        record = {
            "run_signature": run_signature,
            "sample_id": sample["sample_id"],
            "q": int(sample["query"]),
            "k": int(sample["k"]),
            "t": int(sample["t"]),
            "candidate_size": sample["candidate_size"],
            "pair_count": sample["pair_count"],
            "predicted_edge_count": graph.edge_count,
            "prediction_size": len(prediction),
            "truth_size": len(sample["community"]),
            "prediction": sorted(prediction),
            **metrics,
            "size_ratio": len(prediction) / len(sample["community"]),
            "pred_ratio": len(prediction) / total_nodes * 100,
            "elapsed_s": time.time() - sample_started,
            "completed_at": _now(),
        }
        _atomic_write_json(_sample_result_path(work_dir, sample), record)
        records[sample["sample_id"]] = record
        current["sample_pairs_completed"] = sample["pair_count"]
        publish(phase="scoring", force=True)
        print(
            "completed {}/{} sample={} nodes={} pairs={} edges={} elapsed_s={:.3f}".format(
                len(records),
                len(samples),
                sample["sample_id"],
                sample["candidate_size"],
                sample["pair_count"],
                graph.edge_count,
                record["elapsed_s"],
            ),
            flush=True,
        )

    publish(phase="complete", force=True)


def _progressive_status(args):
    progress_path = Path(args.progress)
    if not progress_path.is_file():
        raise FileNotFoundError(
            "progress file not found: {}".format(progress_path)
        )
    payload = json.loads(progress_path.read_text())
    print(json.dumps(payload, indent=2, sort_keys=True))


def _verified_snapshot_split(predictor, dataset):
    split_path = Path(str(predictor.zebra.checkpoint_path) + ".split.json")
    if not split_path.is_file():
        raise ValueError("Zebra checkpoint has no 7:3 snapshot split metadata; retrain")
    split = json.loads(split_path.read_text())
    if (split.get("split_rule") != "snapshot_55_15_30_v1"
            or split.get("snapshot_count") != len(predictor.snapshots)
            or split.get("dataset") != dataset
            or split.get("checkpoint_sha256") != predictor.checkpoint_hash):
        raise ValueError("Zebra checkpoint snapshot split does not match slices")
    return require_fit_boundary(split.get("fit_end_t"), len(predictor.snapshots))


def _evaluate(args):
    wall_start = time.perf_counter()
    predictor, total_nodes = _build_predictor(args)
    session = TimedZebraSession(predictor)
    load_s = session.now() - wall_start
    sample_started = time.perf_counter()
    manifest = load_time_slice_manifest(args.slices_dir)
    dataset_name = manifest["dataset"]
    split_t = _verified_snapshot_split(predictor, args.zebra_dataset)
    start_t = evaluation_start_t(len(predictor.snapshots)) if args.start_t is None else args.start_t
    if not split_t <= start_t < len(predictor.snapshots) - 1:
        raise ValueError("evaluation start outside the shared held-out range")
    samples = load_test_samples(args.slices_dir, len(predictor.snapshots))
    samples = non_empty_samples(sample for sample in samples if sample["t"] >= start_t)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    samples_by_t = defaultdict(list)
    for sample in samples:
        samples_by_t[sample["t"]].append(sample)

    rows = defaultdict(list)
    sample_prepare_s = time.perf_counter() - sample_started
    preparations = []
    query_total = 0.0
    metric_s = 0.0
    candidate_sizes = []
    for t, time_samples in sorted(samples_by_t.items()):
        preparation = session.prepare(t)
        preparations.append(preparation)
        time_query_s = 0.0
        for sample in time_samples:
            prediction, graph, timing = session.query(
                sample["query"], sample["k"], t
            )
            time_query_s += timing["query_s"]
            candidate_sizes.append(timing["candidate_size"])
            metric_started = time.perf_counter()
            metrics = set_metrics(prediction, sample["community"])
            rows[sample["k"]].append({
                **metrics,
                **timing,
                "size_ratio": len(prediction) / len(sample["community"])
                if sample["community"] else 0.0,
                "pred_ratio": len(prediction) / total_nodes * 100,
            })
            metric_s += time.perf_counter() - metric_started
        query_total += time_query_s
        print(
            "t={} samples={} prepare_s={:.3f} query_s={:.3f}".format(
                t, len(time_samples), preparation["prepare_s"], time_query_s
            ), flush=True,
        )

    per_k = _aggregate(rows)
    timing_names = (
        "query_s", "candidate_search_s", "node_encoding_s",
        "edge_prediction_s", "community_search_s",
    )
    for k, values in rows.items():
        per_k[k].update({
            name: float(np.mean([row[name] for row in values], dtype=np.float32))
            for name in timing_names
        })
    macro = {}
    if per_k:
        for name in (
            "precision", "recall", "f1", "jaccard",
            "size_ratio", "pred_ratio", "elapsed_s",
        ) + timing_names:
            macro[name] = float(np.mean([
                result[name] for result in per_k.values()
            ], dtype=np.float32))
    result = {
        **TIMING_SCHEMA,
        "dataset": dataset_name,
        "start_t": start_t,
        "training_split": baseline_training_split(len(predictor.snapshots)),
        "threshold": predictor.threshold,
        **predictor.candidate_metadata,
        "samples": len(samples),
        "sample_scope": "shared_community_eval_non_empty_only",
        "candidate_size_mean": float(np.mean(candidate_sizes, dtype=np.float32)) if samples else 0.0,
        "candidate_size_max": max(candidate_sizes, default=0),
        "query_set_sha256": query_set_sha256(samples),
        "per_k": per_k,
        "macro": macro,
        "load_s": load_s,
        "sample_prepare_s": sample_prepare_s,
        "metric_s": metric_s,
        "prepare_s": sum(row["prepare_s"] for row in preparations),
        "query_s": query_total,
        "elapsed_s": query_total,
        "prepared_time_count": len(preparations),
        "per_time_preparation": preparations,
        "initial_prepare_s": preparations[0]["prepare_s"] if preparations else 0.0,
    }
    result["prediction_total_s"] = result["prepare_s"] + query_total
    result["prepare_mean_s"] = (
        result["prepare_s"] / len(preparations) if preparations else None
    )
    result["incremental_prepare_mean_s"] = (
        sum(row["prepare_s"] for row in preparations[1:]) / (len(preparations) - 1)
        if len(preparations) > 1 else None
    )
    result["query_mean_s"] = query_total / len(samples) if samples else None
    result["amortized_prediction_s"] = (
        result["prediction_total_s"] / len(samples) if samples else None
    )
    result["wall_s"] = session.now() - wall_start
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
    parser.add_argument(
        "--state-cache-dir", default=None,
        help="Initial history-state cache for query/eval; default: "
             "<slices-dir>/model_cache/zebra_state. Pass an empty string to disable.",
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

    progressive_parser = subparsers.add_parser("progressive-eval")
    _add_common_arguments(progressive_parser)
    progressive_parser.add_argument("--start-t", type=int, default=343)
    progressive_parser.add_argument("--end-t", type=int, default=None)
    progressive_parser.add_argument(
        "--work-dir", default=str(ROOT / "results/wiki_talk/zebra_work")
    )
    progressive_parser.add_argument(
        "--embedding-cache-dir",
        default=str(ROOT / ".zebra_embedding_cache"),
    )
    progressive_parser.add_argument(
        "--output",
        default=str(
            ROOT / "results/wiki_talk/zebra_community_eval_progress.json"
        ),
    )
    progressive_parser.add_argument(
        "--comparison-output",
        default=str(
            ROOT / "results/wiki_talk/community_eval_comparison.md"
        ),
    )
    progressive_parser.add_argument(
        "--hybrid-result",
        default=str(
            ROOT / "results/wiki_talk/hybrid_community_eval.json"
        ),
    )
    progressive_parser.add_argument(
        "--status-interval", type=float, default=30.0
    )
    progressive_parser.add_argument("--resume", action="store_true")
    progressive_parser.set_defaults(handler=_progressive_evaluate)

    status_parser = subparsers.add_parser("progressive-status")
    status_parser.add_argument(
        "--progress",
        default=str(
            ROOT / "results/wiki_talk/zebra_community_eval_progress.json"
        ),
    )
    status_parser.set_defaults(handler=_progressive_status)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
