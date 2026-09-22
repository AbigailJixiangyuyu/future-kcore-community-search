#!/usr/bin/env python3
"""Predict next-snapshot communities with on-demand coreness BFS."""

import argparse
import hashlib
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import torch

from datasets.community_eval_builder import (
    sample_qk_coreness_weighted,
    set_metrics,
)
from datasets.coreness_prediction_builder import (
    CurrentSnapshotStructureFeatureTable,
    STRUCTURE_TIME_REFERENCE,
    TimeSliceStructureFeatureTable,
    prepare_inference_feature_table,
)
from datasets.dataset_builder import load_time_slice_manifest
from datasets.indexed_slices import logical_slice_identity
from datasets.snapshot_store import open_snapshot_store, SnapshotStore
from methods.hybrid_coreness import (
    LayeredCommunityResult,
    layered_threshold_bfs,
    load_hybrid_coreness_model,
    predict_coreness_indexed_map,
)
from methods.t_ppr import TemporalPPR
from methods.coreness_edge_generation import generate_predicted_edges
from methods.tcs_representation import TCSStreamingIndex


ROOT = Path(__file__).resolve().parent
DEFAULT_SLICES = ROOT / "data/mooc/time_slices/step_43200_window_86400"
DEFAULT_BATCH_SIZE = 512
EVALUATION_KS = (3, 4, 5, 6, 7)
STATE_CACHE_VERSION = 1
TIMING_SCHEMA = {
    "cross_query_cache": True,
    "cross_query_coreness_cache": True,
    "cross_query_tppr_score_cache": True,
    "timing_version": 2,
    "timing_clock": "perf_counter",
    "elapsed_scope": "bfs_and_inference_plus_edge_generation",
    "prediction_total_scope": "prepare_plus_query_excluding_load_and_evaluation",
    "wall_scope": "entry_to_result_ready_excluding_serialization_and_output",
}


def prepare_timed_context(predictor, t):
    """Charge preparation only to the call that performs it, never cache hits."""
    previous = getattr(predictor, "_context", None)
    started = time.perf_counter()
    context = predictor.prepare_time(t)
    prepare_s = time.perf_counter() - started
    reused = context is previous
    return context, {
        "prepare_s": prepare_s,
        "state_update_s": 0.0 if reused else context.state_update_s,
        "feature_materialize_s": 0.0 if reused else context.feature_materialize_s,
        "context_cache_hit": reused,
    }


@dataclass
class HybridTimeContext:
    """Read-only graph and feature state shared by queries at one time."""

    time: int
    adjacency: dict
    feature_table: dict
    structure_table: Union[TimeSliceStructureFeatureTable, CurrentSnapshotStructureFeatureTable]
    coreness_cache: dict
    state_update_s: float
    feature_materialize_s: float
    edge_neighbor_scores: dict = field(default_factory=dict)

    def release(self):
        """Drop data that is valid only for this query time."""
        self.feature_table.clear()
        self.structure_table.clear()
        self.coreness_cache.clear()
        self.edge_neighbor_scores.clear()


class HybridCommunityPredictor:
    """Run threshold-driven community BFS backed by a coreness model."""

    def __init__(self, snapshots, model, checkpoint, hmax, batch_size=512,
                 device=None, state_cache_dir=None, state_cache_identity=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.snapshots = snapshots
        self.model = model
        self.checkpoint = checkpoint
        self.hmax = int(hmax)
        if isinstance(snapshots, SnapshotStore):
            snapshots.set_core_lookback(model.core_lookback)
        self.batch_size = int(batch_size)
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.feature_config = checkpoint.get("feature_config", {})
        self._validate_config()

        self.t_ppr = TemporalPPR(
            snapshots,
            alpha=self.feature_config["t_ppr_alpha"],
            beta=self.feature_config["t_ppr_beta"],
            streaming_only=True,
        )
        self.t_ppr_index = self.t_ppr.streaming_index(
            top_l=self.feature_config["top_l"],
            # Community inference keeps exactly the Top-L state it consumes.
            # Older checkpoints may record a wider training-time state here;
            # model weights do not depend on that internal approximation width.
            internal_top_k=self.feature_config["top_l"],
            min_score=self.feature_config["min_probability"],
        )
        self.tcs_index = TCSStreamingIndex(
            snapshots, kmax=self.model.kmax
        )
        self._adjacency = {}
        self._current_time = -1
        self._context = None
        self.state_cache_dir = (
            Path(state_cache_dir) if state_cache_dir is not None else None
        )
        self.state_cache_identity = state_cache_identity

    def _validate_config(self):
        if self.feature_config.get("structure_time_reference") != STRUCTURE_TIME_REFERENCE:
            raise ValueError(
                "checkpoint uses historical or unspecified structure times; "
                "retrain with current_observed_snapshot structures"
            )
        required = {
            "top_l",
            "t_ppr_internal_top_k",
            "order",
            "t_ppr_alpha",
            "t_ppr_beta",
            "min_probability",
        }
        missing = required - set(self.feature_config)
        if missing:
            raise ValueError(
                "checkpoint feature_config is missing: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        if self.model.hmax != self.hmax:
            raise ValueError(
                "checkpoint hmax {} does not match snapshot hmax {}".format(
                    self.model.hmax, self.hmax
                )
            )
        if self.model.order != int(self.feature_config["order"]):
            raise ValueError("checkpoint model and feature order do not match")

    def _state_cache_metadata(self, t):
        return {
            "version": STATE_CACHE_VERSION,
            "numeric_dtype": "float32",
            "identity": self.state_cache_identity,
            "time": int(t),
            "snapshot_count": len(self.snapshots),
            "kmax": int(self.model.kmax),
            "tcs_alpha": float(self.tcs_index.alpha),
            "top_l": int(self.t_ppr_index.top_l),
            "internal_top_k": int(self.t_ppr_index.internal_top_k),
            "t_ppr_alpha": float(self.t_ppr.alpha),
            "t_ppr_beta": float(self.t_ppr.beta),
            "min_score": float(self.t_ppr_index.min_score),
            "node_index_size": len(self.t_ppr_index._nodes),
        }

    def _cache_metadata_matches(self, metadata, target_time):
        if not isinstance(metadata, dict):
            return False
        cache_time = metadata.get("time")
        if not isinstance(cache_time, int) or cache_time > target_time:
            return False
        expected = self._state_cache_metadata(cache_time)
        return metadata == expected

    def _default_state_cache_path(self, t):
        if self.state_cache_dir is None:
            raise RuntimeError("state caching is disabled")
        return self.state_cache_dir / "hybrid_state_t{:06d}.npz".format(t)

    @staticmethod
    def _adjacency_arrays(adjacency):
        nodes = np.asarray(sorted(adjacency), dtype=np.int64)
        degrees = np.fromiter(
            (len(adjacency[int(node)]) for node in nodes),
            dtype=np.int64,
            count=len(nodes),
        )
        offsets = np.empty(len(nodes) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(degrees, out=offsets[1:])
        neighbors = np.fromiter(
            (
                neighbor
                for node in nodes
                for neighbor in sorted(adjacency[int(node)])
            ),
            dtype=np.int64,
            count=int(offsets[-1]),
        )
        return nodes, offsets, neighbors

    def save_state_cache(self, t=None, output_path=None):
        """Persist the causal streaming state without materialized features."""
        if t is None:
            t = self._current_time
        if t != self._current_time or t < 0:
            raise ValueError("state cache time must equal the current state time")
        output_path = (
            Path(output_path)
            if output_path is not None
            else self._default_state_cache_path(t)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        adjacency_nodes, adjacency_offsets, adjacency_neighbors = (
            self._adjacency_arrays(self._adjacency)
        )
        t_ppr_rows = np.flatnonzero(
            (self.t_ppr_index._state_lengths > 0)
            | (self.t_ppr_index._norms != 0.0)
        ).astype(np.int64, copy=False)
        tcs_rows = np.flatnonzero(
            self.tcs_index._last_times >= 0
        ).astype(np.int64, copy=False)
        sparse_nodes = np.asarray(
            sorted(self.tcs_index._sparse_states), dtype=np.int64
        )
        if len(sparse_nodes):
            sparse_penalties = np.stack([
                self.tcs_index._sparse_states[int(node)][0]
                for node in sparse_nodes
            ])
            sparse_times = np.asarray([
                self.tcs_index._sparse_states[int(node)][1]
                for node in sparse_nodes
            ], dtype=np.int32)
        else:
            sparse_penalties = np.empty(
                (0, self.model.kmax), dtype=np.float32
            )
            sparse_times = np.empty(0, dtype=np.int32)

        temporary_path = output_path.with_name(output_path.name + ".tmp")
        try:
            with temporary_path.open("wb") as cache_file:
                np.savez(
                    cache_file,
                    metadata=np.asarray(json.dumps(
                        self._state_cache_metadata(t), sort_keys=True
                    )),
                    adjacency_nodes=adjacency_nodes,
                    adjacency_offsets=adjacency_offsets,
                    adjacency_neighbors=adjacency_neighbors,
                    t_ppr_rows=t_ppr_rows,
                    t_ppr_nodes=self.t_ppr_index._state_nodes[t_ppr_rows],
                    t_ppr_times=self.t_ppr_index._state_times[t_ppr_rows],
                    t_ppr_scores=self.t_ppr_index._state_scores[t_ppr_rows],
                    t_ppr_lengths=self.t_ppr_index._state_lengths[t_ppr_rows],
                    t_ppr_norms=self.t_ppr_index._norms[t_ppr_rows],
                    tcs_rows=tcs_rows,
                    tcs_penalties=self.tcs_index._penalties[tcs_rows],
                    tcs_times=self.tcs_index._last_times[tcs_rows],
                    tcs_weight_sum=np.asarray(self.tcs_index.weight_sum, dtype=np.float32),
                    tcs_sparse_nodes=sparse_nodes,
                    tcs_sparse_penalties=sparse_penalties,
                    tcs_sparse_times=sparse_times,
                )
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return output_path

    @staticmethod
    def _read_cache_metadata(cache_path):
        with np.load(str(cache_path), allow_pickle=False) as cached:
            return json.loads(str(cached["metadata"].item()))

    def _restore_state_cache(self, cache_path, metadata=None):
        """Restore one validated cache into freshly initialized indexes."""
        with np.load(str(cache_path), allow_pickle=False) as cached:
            if metadata is None:
                metadata = json.loads(str(cached["metadata"].item()))
            cache_time = int(metadata["time"])
            if not self._cache_metadata_matches(metadata, cache_time):
                raise ValueError("Hybrid state cache is incompatible")

            adjacency_nodes = cached["adjacency_nodes"]
            adjacency_offsets = cached["adjacency_offsets"]
            adjacency_neighbors = cached["adjacency_neighbors"]
            if len(adjacency_offsets) != len(adjacency_nodes) + 1:
                raise ValueError("Hybrid state cache adjacency is invalid")
            self._adjacency = {
                int(node): set(adjacency_neighbors[
                    adjacency_offsets[position]:adjacency_offsets[position + 1]
                ].tolist())
                for position, node in enumerate(adjacency_nodes)
            }

            t_ppr_rows = cached["t_ppr_rows"]
            if len(t_ppr_rows) and int(t_ppr_rows.max()) >= len(
                self.t_ppr_index._nodes
            ):
                raise ValueError("Hybrid state cache T-PPR rows are invalid")
            self.t_ppr_index._state_nodes[t_ppr_rows] = cached["t_ppr_nodes"]
            self.t_ppr_index._state_times[t_ppr_rows] = cached["t_ppr_times"]
            self.t_ppr_index._state_scores[t_ppr_rows] = cached["t_ppr_scores"]
            self.t_ppr_index._state_lengths[t_ppr_rows] = cached[
                "t_ppr_lengths"
            ]
            self.t_ppr_index._norms[t_ppr_rows] = cached["t_ppr_norms"]
            self.t_ppr_index.current_time = cache_time

            tcs_rows = cached["tcs_rows"]
            if len(tcs_rows):
                self.tcs_index._ensure_capacity(int(tcs_rows.max()))
                self.tcs_index._penalties[tcs_rows] = cached["tcs_penalties"]
                self.tcs_index._last_times[tcs_rows] = cached["tcs_times"]
            self.tcs_index._sparse_states = {
                int(node): (penalties.copy(), int(sparse_time))
                for node, penalties, sparse_time in zip(
                    cached["tcs_sparse_nodes"],
                    cached["tcs_sparse_penalties"],
                    cached["tcs_sparse_times"],
                )
            }
            self.tcs_index.weight_sum = np.float32(cached["tcs_weight_sum"].item())
            self.tcs_index.current_time = cache_time
        self._current_time = cache_time
        return cache_time

    def _restore_best_state_cache(self, target_time):
        if self.state_cache_dir is None or not self.state_cache_dir.exists():
            return None
        candidates = []
        for cache_path in self.state_cache_dir.glob("hybrid_state_t*.npz"):
            try:
                metadata = self._read_cache_metadata(cache_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if self._cache_metadata_matches(metadata, target_time):
                candidates.append((metadata["time"], cache_path, metadata))
        if not candidates:
            return None
        _, cache_path, metadata = max(candidates, key=lambda item: item[0])
        cache_time = self._restore_state_cache(cache_path, metadata)
        print(
            "[hybrid_state] Loaded t={} from {}".format(
                cache_time, cache_path
            ),
            flush=True,
        )
        return cache_path

    def advance_state(self, t, use_cache=True, auto_save=True):
        """Restore or compute state; persist the first requested time on a miss.

        Later times advance in memory without producing one cache per snapshot.
        Explicit cache-building commands disable auto_save to control the output.
        """
        if not isinstance(t, int):
            raise TypeError("t must be an integer snapshot index")
        if t < 0 or t >= len(self.snapshots) - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        if t < self._current_time:
            raise ValueError("predictor time cannot move backwards")

        started = time.perf_counter()
        first_cached_advance = (
            use_cache and self._current_time < 0
            and self.state_cache_dir is not None
        )
        # Invalidate before mutating the shared graph or streaming indexes.
        # Explicit advance_state() calls must not leave old features usable.
        if self._context is not None and t != self._context.time:
            self._context.release()
            self._context = None
        if isinstance(self.snapshots, SnapshotStore):
            self.snapshots.retain_for_prediction(t, self.model.core_lookback)
        if use_cache and self._current_time < 0:
            self._restore_best_state_cache(t)
        target_cache_hit = self._current_time == t
        for snapshot_time in range(self._current_time + 1, t + 1):
            snapshot = self.snapshots[snapshot_time]
            for node in snapshot.get("core_dict", {}):
                self._adjacency.setdefault(node, set())
            for edge in snapshot["edge_list"]:
                u, v = edge[0], edge[1]
                if u == v:
                    continue
                self._adjacency.setdefault(u, set()).add(v)
                self._adjacency.setdefault(v, set()).add(u)

            self.t_ppr_index.advance_to(snapshot_time)
            self.tcs_index.advance_to(snapshot_time)
        self._current_time = t
        if first_cached_advance and auto_save and not target_cache_hit:
            cache_path = self.save_state_cache()
            print(
                "[hybrid_state] Saved t={} to {}".format(t, cache_path),
                flush=True,
            )
        return time.perf_counter() - started

    def prepare_time(self, t):
        """Advance state and materialize shared features for query time ``t``."""
        # Validate even on cache hits; advance_state also invalidates old context.
        state_update_s = self.advance_state(t)
        if self._context is not None and self._context.time == t:
            return self._context
        if self._context is not None:
            self._context.release()
            self._context = None

        feature_started = time.perf_counter()
        feature_table, structure_table = prepare_inference_feature_table(
            self.snapshots,
            self._adjacency,
            t=t,
            kmax=self.model.kmax,
            hmax=self.hmax,
            t_ppr_index=self.t_ppr_index,
            tcs_index=self.tcs_index,
            order=self.model.order,
            core_lookback=self.model.core_lookback,
        )
        feature_materialize_s = time.perf_counter() - feature_started
        if isinstance(self.snapshots, SnapshotStore):
            self.snapshots.release_payloads()
        self._context = HybridTimeContext(
            time=t,
            adjacency=self._adjacency,
            feature_table=feature_table,
            structure_table=structure_table,
            coreness_cache={},
            state_update_s=state_update_s,
            feature_materialize_s=feature_materialize_s,
        )
        return self._context

    def generate_edges(self, t, nodes, k, context=None):
        """Generate edges only within the explicitly selected BFS node set."""
        context = context or self.prepare_time(t)
        if context.time != t or context is not self._context:
            raise ValueError("context is stale or does not match the query time")
        if self.t_ppr_index.current_time != t:
            raise ValueError("T-PPR state does not match the query time")
        selected = set(nodes)
        if not selected.issubset(context.adjacency):
            raise ValueError("selected nodes must belong to historical graph")
        if not selected.issubset(context.coreness_cache):
            raise ValueError("selected nodes must have cached BFS predictions")
        missing = sorted(selected - context.edge_neighbor_scores.keys())
        if missing:
            context.edge_neighbor_scores.update(
                self.t_ppr_index.top_neighbor_scores(missing)
            )
        return generate_predicted_edges(
            selected,
            context.coreness_cache,
            None,
            k,
            scores_for_node=context.edge_neighbor_scores.__getitem__,
        )

    def predict(self, q, k, t, context=None):
        """Reuse current-time coreness predictions and merged T-PPR rows."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if k > self.model.kmax:
            raise ValueError("k exceeds the checkpoint coreness range")
        context = context or self.prepare_time(t)
        if context.time != t:
            raise ValueError("context time does not match the query")
        if context is not self._context:
            raise ValueError("context is stale; prepare the query time again")
        if q not in context.adjacency:
            return LayeredCommunityResult(frozenset(), {}, 0, 0)

        def predict_batch(nodes):
            # layered_threshold_bfs already supplies sorted, unique node IDs.
            missing = [
                node for node in nodes if node not in context.coreness_cache
            ]
            if missing:
                context.coreness_cache.update(predict_coreness_indexed_map(
                    self.model,
                    context.feature_table,
                    context.structure_table,
                    missing,
                    batch_size=self.batch_size,
                    device=self.device,
                ))
            return {node: context.coreness_cache[node] for node in nodes}

        cached_before = len(context.coreness_cache)
        result = layered_threshold_bfs(
            q, k, context.adjacency, predict_batch
        )
        return result.with_new_prediction_count(
            len(context.coreness_cache) - cached_before
        )


def _resolve_device(device_name):
    if device_name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return device_name


def _resolve_checkpoint_path(slices_dir, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint)
    return Path(slices_dir) / "model_cache" / "hybrid_coreness.pt"


def _state_cache_identity(slices_dir):
    """Fingerprint data/feature semantics, not the physical storage layout.

    The versioned prefix deliberately rejects old metadata-bytes-only keys:
    they did not certify window content and cannot safely be auto-aliased.
    """
    slices_dir = Path(slices_dir)
    manifest = load_time_slice_manifest(slices_dir)
    metadata = json.loads(
        (slices_dir / "snapshot_cache" / "metadata.json").read_text()
    )
    payload = {
        "version": "hybrid_state_inputs_v2",
        "windows": logical_slice_identity(slices_dir, manifest),
        "snapshot_features": metadata,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "v2:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_predictor(args):
    slices_dir = Path(args.slices_dir)
    snapshots, total_nodes, kmax, hmax = open_snapshot_store(slices_dir)
    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint_path(slices_dir, args.checkpoint)
    model, checkpoint = load_hybrid_coreness_model(
        checkpoint_path, device=device
    )
    if checkpoint.get("feature_config", {}).get("numeric_dtype") != "float32":
        warnings.warn(
            "This checkpoint predates float32 T-PPR/TCS computation. It remains "
            "loadable, but predictions must be re-evaluated with the new inputs."
        )
    if model.kmax != kmax:
        raise ValueError(
            "checkpoint kmax {} does not match snapshot kmax {}".format(
                model.kmax, kmax
            )
        )
    predictor = HybridCommunityPredictor(
        snapshots,
        model,
        checkpoint,
        hmax=hmax,
        batch_size=args.batch_size,
        device=device,
        state_cache_dir=slices_dir / "model_cache" / "hybrid_state",
        state_cache_identity=_state_cache_identity(slices_dir),
    )
    return predictor, total_nodes


def _query(args):
    wall_start = time.perf_counter()
    predictor, _ = _build_predictor(args)
    load_s = time.perf_counter() - wall_start
    context, preparation = prepare_timed_context(predictor, args.t)
    started = time.perf_counter()
    result = predictor.predict(args.q, args.k, args.t, context=context)
    query_s = time.perf_counter() - started
    payload = {
        **TIMING_SCHEMA,
        **preparation,
        "prediction_scope": "nodes_only",
        "load_s": load_s,
        "bfs_selection_s": query_s,
        "edge_generation_s": 0.0,
        "query_s": query_s,
        "prediction_total_s": preparation["prepare_s"] + query_s,
        "q": args.q,
        "k": args.k,
        "t": args.t,
        "community_size": len(result.community),
        "examined_node_count": result.predicted_node_count,
        "predicted_node_count": result.predicted_node_count,
        "newly_predicted_node_count": result.newly_predicted_node_count,
        "reused_prediction_count": result.reused_prediction_count,
        "accepted_node_count": result.accepted_node_count,
        "rejected_node_count": (
            result.predicted_node_count - result.accepted_node_count
        ),
        "bfs_layers": result.bfs_layers,
        "community": sorted(result.community),
        "elapsed_s": query_s,
    }
    payload["wall_s"] = time.perf_counter() - wall_start
    print(json.dumps(payload, ensure_ascii=True))


def _edge_prediction_metrics(predicted_edges, target_snapshot, true_community):
    """Compare final generated edges with the target community's induced edges."""
    community = set(true_community)
    truth = {
        (min(u, v), max(u, v))
        for u, v, *_ in target_snapshot["edge_list"]
        if u != v and u in community and v in community
    }
    prediction = {
        (min(u, v), max(u, v))
        for u, v in predicted_edges if u != v
    }
    return {
        f"edge_{name}": value
        for name, value in set_metrics(prediction, truth).items()
    }


def _aggregate(rows):
    result = {}
    metric_names = (
        "precision",
        "recall",
        "f1",
        "jaccard",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "edge_jaccard",
        "size_ratio",
        "pred_ratio",
        "predicted_node_count",
        "bfs_layers",
        "elapsed_s",
        "query_s",
        "bfs_selection_s",
        "edge_generation_s",
    )
    for k, values in sorted(rows.items()):
        result[k] = {
            name: float(np.mean([row[name] for row in values], dtype=np.float32))
            for name in metric_names
        }
        result[k]["samples"] = len(values)
        lowered_valid = [
            row for row in values if row["case3_lowered_node_ratio"] is not None
        ]
        result[k]["core_ratio_valid_samples"] = len(lowered_valid)
        result[k]["core_ratio_empty_samples"] = len(values) - len(lowered_valid)
        result[k]["case3_lowered_node_count"] = sum(
            row["case3_lowered_node_count"] for row in values
        )
        result[k]["case3_lowered_node_ratio"] = (
            float(np.mean([row["case3_lowered_node_ratio"] for row in lowered_valid], dtype=np.float32))
            if lowered_valid else None
        )
        valid = [row for row in values if row["edge_case1_ratio"] is not None]
        result[k]["edge_ratio_valid_samples"] = len(valid)
        result[k]["edge_ratio_empty_samples"] = len(values) - len(valid)
        for i in (1, 2, 3):
            result[k][f"edge_case{i}_count"] = sum(
                row[f"edge_case{i}_count"] for row in values
            )
            result[k][f"edge_case{i}_ratio"] = (
                float(np.mean([row[f"edge_case{i}_ratio"] for row in valid], dtype=np.float32))
                if valid else None
            )
    return result


def _edge_aggregate_summary(per_k):
    """Equal-weight valid K means, with separate cumulative audit counts."""
    summary = {
        name: sum(row[name] for row in per_k.values())
        for name in (
            "edge_ratio_valid_samples", "edge_ratio_empty_samples",
            "core_ratio_valid_samples", "core_ratio_empty_samples",
            "case3_lowered_node_count",
        )
    }
    lowered_valid = [row for row in per_k.values()
                     if row["core_ratio_valid_samples"] > 0]
    summary["case3_lowered_node_ratio"] = (
        float(np.mean([row["case3_lowered_node_ratio"] for row in lowered_valid], dtype=np.float32))
        if lowered_valid else None
    )
    valid = [row for row in per_k.values()
             if row["edge_ratio_valid_samples"] > 0]
    for i in (1, 2, 3):
        summary[f"edge_case{i}_count"] = sum(
            row[f"edge_case{i}_count"] for row in per_k.values()
        )
        summary[f"edge_case{i}_ratio"] = (
            float(np.mean([row[f"edge_case{i}_ratio"] for row in valid], dtype=np.float32))
            if valid else None
        )
    return summary


def _build_state_cache(args):
    predictor, _ = _build_predictor(args)
    cache_time = (
        int(len(predictor.snapshots) * 0.7)
        if args.time is None else args.time
    )
    started = time.perf_counter()
    state_update_s = predictor.advance_state(
        cache_time, use_cache=not args.rebuild, auto_save=False
    )
    output_path = predictor.save_state_cache(
        cache_time, output_path=args.output
    )
    payload = {
        "time": cache_time,
        "path": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "state_update_s": state_update_s,
        "wall_s": time.perf_counter() - started,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _evaluate(args, *, predictor_builder=None, metadata=None):
    wall_start = time.perf_counter()
    predictor, total_nodes = (predictor_builder or _build_predictor)(args)
    load_s = time.perf_counter() - wall_start
    sample_started = time.perf_counter()
    manifest = load_time_slice_manifest(args.slices_dir)
    dataset_name = manifest["dataset"]
    split_t = int(len(predictor.snapshots) * 0.7)
    start_t = split_t if args.start_t is None else args.start_t
    samples = sample_qk_coreness_weighted(
        predictor.snapshots,
        split_t,
        EVALUATION_KS,
        dataset_name=dataset_name,
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
        samples_by_t[sample["t"]].append(sample)

    rows = defaultdict(list)
    sample_prepare_s = time.perf_counter() - sample_started
    total_examined_nodes = 0
    total_unique_predictions = 0
    total_state_update_s = 0.0
    total_feature_materialize_s = 0.0
    total_query_s = 0.0
    total_edge_generation_s = 0.0
    total_prepare_s = 0.0
    total_bfs_s = 0.0
    metric_s = 0.0
    for t, time_samples in sorted(samples_by_t.items()):
        context, preparation = prepare_timed_context(predictor, t)
        total_prepare_s += preparation["prepare_s"]
        query_s = 0.0
        predicted_at_time = 0
        new_predictions_at_time = 0
        for sample in time_samples:
            started = time.perf_counter()
            prediction = predictor.predict(
                sample["query"], sample["k"], sample["t"], context=context
            )
            edge_started = time.perf_counter()
            bfs_selection_s = edge_started - started
            total_bfs_s += bfs_selection_s
            generated = predictor.generate_edges(
                t, prediction.community, sample["k"], context=context
            )
            edge_generation_s = time.perf_counter() - edge_started
            total_edge_generation_s += edge_generation_s
            elapsed = bfs_selection_s + edge_generation_s
            query_s += elapsed
            predicted_at_time += prediction.predicted_node_count
            new_predictions_at_time += prediction.newly_predicted_node_count
            metric_started = time.perf_counter()
            metrics = set_metrics(prediction.community, sample["community"])
            edge_metrics = _edge_prediction_metrics(
                generated.edges, predictor.snapshots[t + 1], sample["community"]
            )
            rows[sample["k"]].append({
                **metrics,
                **edge_metrics,
                "size_ratio": len(prediction.community) / len(sample["community"]),
                "pred_ratio": len(prediction.community) / total_nodes * 100,
                "predicted_node_count": prediction.predicted_node_count,
                "bfs_layers": prediction.bfs_layers,
                "elapsed_s": elapsed,
                "query_s": elapsed,
                "bfs_selection_s": bfs_selection_s,
                "edge_generation_s": edge_generation_s,
                **generated.edge_metrics(),
                **generated.core_reduction_metrics(),
            })
            metric_s += time.perf_counter() - metric_started
        print(
            "t={} samples={} examined_nodes={} new_predictions={} "
            "cache_hits={} nodes={} structures={} state_update_s={:.3f} "
            "feature_materialize_s={:.3f} query_s={:.3f}".format(
                t,
                len(time_samples),
                predicted_at_time,
                new_predictions_at_time,
                predicted_at_time - new_predictions_at_time,
                len(context.feature_table["nodes"]),
                len(context.structure_table),
                context.state_update_s,
                context.feature_materialize_s,
                query_s,
            ),
            flush=True,
        )
        total_examined_nodes += predicted_at_time
        total_unique_predictions += new_predictions_at_time
        total_state_update_s += preparation["state_update_s"]
        total_feature_materialize_s += preparation["feature_materialize_s"]
        total_query_s += query_s
        if isinstance(predictor.snapshots, SnapshotStore):
            # Future truth is evaluation-only; do not retain it between slices.
            predictor.snapshots.retain_for_prediction(
                t, predictor.model.core_lookback
            )
            predictor.snapshots.release_payloads()

    per_k = _aggregate(rows)
    macro = {}
    if per_k:
        for name in (
            "precision",
            "recall",
            "f1",
            "jaccard",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "edge_jaccard",
            "size_ratio",
            "pred_ratio",
            "predicted_node_count",
            "bfs_layers",
            "elapsed_s",
            "query_s",
            "bfs_selection_s",
            "edge_generation_s",
        ):
            macro[name] = float(np.mean([
                result[name] for result in per_k.values()
            ], dtype=np.float32))
    edge_summary = _edge_aggregate_summary(per_k)
    macro.update({name: value for name, value in edge_summary.items()
                  if name.endswith("_ratio")})
    payload = {
        **(metadata or {}),
        **TIMING_SCHEMA,
        "prediction_scope": "nodes_and_edges",
        "cache_policy": "shared_time_features_coreness_and_tppr_scores",
        "load_s": load_s,
        "sample_prepare_s": sample_prepare_s,
        "metric_s": metric_s,
        "prepare_s": total_prepare_s,
        "bfs_selection_s": total_bfs_s,
        "prediction_total_s": total_prepare_s + total_query_s,
        "elapsed_s": total_query_s,
        "amortized_prediction_s": (
            (total_prepare_s + total_query_s) / len(samples) if samples else None
        ),
        **{name: value for name, value in edge_summary.items()
           if not name.endswith("_ratio")},
        "edge_attribution": "final_edge_latest_creation_case",
        "edge_ratio_aggregation": "query_mean_then_valid_k_mean_excluding_edgeless",
        "core_ratio_aggregation": "query_mean_then_valid_k_mean_excluding_empty_communities",
        "core_ratio_denominator": "all_bfs_community_nodes",
        "dataset": dataset_name,
        "checkpoint": str(
            _resolve_checkpoint_path(args.slices_dir, args.checkpoint).resolve()
        ),
        "start_t": start_t,
        "samples": len(samples),
        "query_set_sha256": hashlib.sha256(json.dumps(sorted(
            (int(s["query"]), int(s["k"]), int(s["t"])) for s in samples
        )).encode("ascii")).hexdigest(),
        "examined_node_count": total_examined_nodes,
        "unique_predicted_node_count": total_unique_predictions,
        "reused_prediction_count": (
            total_examined_nodes - total_unique_predictions
        ),
        "state_update_s": total_state_update_s,
        "feature_materialize_s": total_feature_materialize_s,
        "query_s": total_query_s,
        "edge_generation_s": total_edge_generation_s,
        "per_k": per_k,
        "macro": macro,
    }
    payload["wall_s"] = time.perf_counter() - wall_start
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def _add_common_arguments(parser):
    parser.add_argument("--slices-dir", default=str(DEFAULT_SLICES))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="defaults to <slices-dir>/model_cache/hybrid_coreness.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )


def main():
    parser = argparse.ArgumentParser(
        description="On-demand hybrid coreness community prediction"
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

    cache_parser = subparsers.add_parser("build-state-cache")
    _add_common_arguments(cache_parser)
    cache_parser.add_argument("--time", type=int, default=None)
    cache_parser.add_argument("--output", default=None)
    cache_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore an existing earlier state cache while building",
    )
    cache_parser.set_defaults(handler=_build_state_cache)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
