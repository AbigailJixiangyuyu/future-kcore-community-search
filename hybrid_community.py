#!/usr/bin/env python3
"""Predict next-snapshot communities with on-demand coreness BFS."""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from datasets.community_eval_builder import (
    sample_qk_coreness_weighted,
    set_metrics,
)
from datasets.coreness_prediction_builder import (
    prepare_inference_feature_arrays,
)
from datasets.dataset_builder import build_snapshots, load_time_slice_manifest
from methods.hybrid_coreness import (
    layered_threshold_bfs,
    load_hybrid_coreness_model,
    predict_coreness_map,
)
from methods.t_ppr import TemporalPPR


ROOT = Path(__file__).resolve().parent
DEFAULT_SLICES = ROOT / "data/mooc/time_slices/step_43200_window_86400"
DEFAULT_CHECKPOINT = (
    DEFAULT_SLICES / "model_cache/hybrid_coreness_v5_h64.pt"
)
DEFAULT_BATCH_SIZE = 512
EVALUATION_KS = (3, 4, 5, 6, 7)


@dataclass
class HybridTimeContext:
    """Read-only graph and feature state shared by queries at one time."""

    time: int
    adjacency: dict
    t_ppr_index: object
    structure_cache: dict
    tcs_cache: dict
    influence_cache: dict
    coreness_cache: dict


class HybridCommunityPredictor:
    """Run threshold-driven community BFS backed by a coreness model."""

    def __init__(self, snapshots, model, checkpoint, hmax, batch_size=512,
                 device=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.snapshots = snapshots
        self.model = model
        self.checkpoint = checkpoint
        self.hmax = int(hmax)
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
        )
        self.t_ppr_index = self.t_ppr.streaming_index(
            top_l=self.feature_config["top_l"],
            internal_top_k=self.feature_config["t_ppr_internal_top_k"],
            min_score=self.feature_config["min_probability"],
        )
        self._adjacency = {}
        self._current_time = -1
        self._context = None

    def _validate_config(self):
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

    def prepare_time(self, t):
        """Advance cumulative graph and T-PPR state to ``t`` exactly once."""
        if not isinstance(t, int):
            raise TypeError("t must be an integer snapshot index")
        if t < 0 or t >= len(self.snapshots) - 1:
            raise IndexError("prediction requires both t and t+1 snapshots")
        if t < self._current_time:
            raise ValueError("predictor time cannot move backwards")
        if self._context is not None and t == self._current_time:
            return self._context

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

        self.t_ppr_index.advance_to(t)
        self._current_time = t
        self._context = HybridTimeContext(
            time=t,
            adjacency=self._adjacency,
            t_ppr_index=self.t_ppr_index,
            structure_cache={},
            tcs_cache={},
            influence_cache={},
            coreness_cache={},
        )
        return self._context

    def predict(self, q, k, t, context=None):
        """Predict one community using the query time's node cache."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if k > self.model.kmax:
            raise ValueError("k exceeds the checkpoint coreness range")
        context = context or self.prepare_time(t)
        if context.time != t:
            raise ValueError("context time does not match the query")
        if context is not self._context:
            raise ValueError("context is stale; prepare the query time again")

        def predict_batch(nodes):
            nodes = sorted(set(nodes))
            missing = [
                node for node in nodes if node not in context.coreness_cache
            ]
            if missing:
                arrays = prepare_inference_feature_arrays(
                    self.snapshots,
                    missing,
                    t=t,
                    kmax=self.model.kmax,
                    hmax=self.hmax,
                    t_ppr_index=context.t_ppr_index,
                    order=self.model.order,
                    structure_cache=context.structure_cache,
                    tcs_cache=context.tcs_cache,
                    influence_cache=context.influence_cache,
                )
                context.coreness_cache.update(predict_coreness_map(
                    self.model,
                    arrays,
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


def _build_predictor(args):
    slices_dir = Path(args.slices_dir)
    snapshots, total_nodes, kmax, hmax = build_snapshots(slices_dir)
    device = _resolve_device(args.device)
    model, checkpoint = load_hybrid_coreness_model(
        args.checkpoint, device=device
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
    )
    return predictor, total_nodes


def _query(args):
    predictor, _ = _build_predictor(args)
    started = time.time()
    result = predictor.predict(args.q, args.k, args.t)
    payload = {
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
        "elapsed_s": time.time() - started,
    }
    print(json.dumps(payload, ensure_ascii=True))


def _aggregate(rows):
    result = {}
    metric_names = (
        "precision",
        "recall",
        "f1",
        "jaccard",
        "size_ratio",
        "pred_ratio",
        "predicted_node_count",
        "bfs_layers",
        "elapsed_s",
    )
    for k, values in sorted(rows.items()):
        result[k] = {
            name: float(np.mean([row[name] for row in values]))
            for name in metric_names
        }
        result[k]["samples"] = len(values)
    return result


def _evaluate(args):
    predictor, total_nodes = _build_predictor(args)
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
    wall_start = time.time()
    total_examined_nodes = 0
    total_unique_predictions = 0
    for t, time_samples in sorted(samples_by_t.items()):
        context_started = time.time()
        context = predictor.prepare_time(t)
        context_elapsed = time.time() - context_started
        predicted_at_time = 0
        new_predictions_at_time = 0
        for sample in time_samples:
            started = time.time()
            prediction = predictor.predict(
                sample["query"], sample["k"], sample["t"], context=context
            )
            elapsed = time.time() - started
            predicted_at_time += prediction.predicted_node_count
            new_predictions_at_time += prediction.newly_predicted_node_count
            metrics = set_metrics(prediction.community, sample["community"])
            rows[sample["k"]].append({
                **metrics,
                "size_ratio": len(prediction.community) / len(sample["community"]),
                "pred_ratio": len(prediction.community) / total_nodes * 100,
                "predicted_node_count": prediction.predicted_node_count,
                "bfs_layers": prediction.bfs_layers,
                "elapsed_s": elapsed,
            })
        print(
            "t={} samples={} examined_nodes={} new_predictions={} "
            "cache_hits={} index_s={:.3f}".format(
                t,
                len(time_samples),
                predicted_at_time,
                new_predictions_at_time,
                predicted_at_time - new_predictions_at_time,
                context_elapsed,
            ),
            flush=True,
        )
        total_examined_nodes += predicted_at_time
        total_unique_predictions += new_predictions_at_time

    per_k = _aggregate(rows)
    macro = {}
    if per_k:
        for name in (
            "precision",
            "recall",
            "f1",
            "jaccard",
            "size_ratio",
            "pred_ratio",
            "predicted_node_count",
            "bfs_layers",
            "elapsed_s",
        ):
            macro[name] = float(np.mean([
                result[name] for result in per_k.values()
            ]))
    payload = {
        "dataset": dataset_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "start_t": start_t,
        "samples": len(samples),
        "examined_node_count": total_examined_nodes,
        "unique_predicted_node_count": total_unique_predictions,
        "reused_prediction_count": (
            total_examined_nodes - total_unique_predictions
        ),
        "per_k": per_k,
        "macro": macro,
        "wall_s": time.time() - wall_start,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def _add_common_arguments(parser):
    parser.add_argument("--slices-dir", default=str(DEFAULT_SLICES))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
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

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
