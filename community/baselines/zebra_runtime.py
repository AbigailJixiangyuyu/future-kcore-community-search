"""Zebra snapshot state, link scoring, and timed community queries."""

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F

from community.baselines.baseline_graph import PredictedGraph, historical_community_union


DEFAULT_THRESHOLD = 0.5
DEFAULT_PAIR_BATCH_SIZE = 65536
CACHE_VERSION = 2


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
