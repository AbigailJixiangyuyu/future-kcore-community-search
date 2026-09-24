#!/usr/bin/env python3
"""Predict next-snapshot k-core communities from Zebra link scores."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from community.baselines.baseline_graph import (
    PredictedGraph,
    historical_community_union,
)
from community.baselines.zebra_progress_store import (
    COMMUNITY_METRICS,
    EMBEDDING_CACHE_VERSION,
    PROGRESSIVE_KS,
    PROGRESSIVE_RESULT_VERSION,
    _atomic_write_json,
    _build_progress_payload,
    _load_completed_records,
    _prepare_progressive_samples,
    _progressive_run_signature,
)
from community.baselines.zebra_progressive import (
    _progressive_evaluate as _run_progressive_evaluation,
    _progressive_status,
)
from community.baselines.zebra_runtime import (
    CACHE_VERSION,
    DEFAULT_PAIR_BATCH_SIZE,
    DEFAULT_THRESHOLD,
    HistoricalEdgeIndex,
    TimedZebraSession,
    ZebraCommunityPredictor,
    _build_projected_decoder,
)
from datasets.community_eval_builder import set_metrics
from datasets.dataset_builder import build_snapshots, load_time_slice_manifest
from datasets.baseline_eval import (evaluation_start_t, load_test_samples,
                                    non_empty_samples, query_set_sha256)
from methods.zebra_history_cache import ZebraHistoryCache


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZEBRA_ROOT = ROOT / "third_party" / "Zebra"
DEFAULT_SLICES = (
    ROOT / "data/mooc/time_slices/step_43200_window_86400"
)
DEFAULT_ZEBRA_DATASET = "mooc-snapshot-7x3-baseline_7x3_20260922_164330_i13z9hpl"
DEFAULT_CHECKPOINT = (
    DEFAULT_ZEBRA_ROOT
    / "saved_checkpoints"
    / (DEFAULT_ZEBRA_DATASET + "-50-0.0001-streaming-"
       "[0.1, 0.1]-[0.5, 0.95]-20.pth")
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


def _progressive_evaluate(args):
    return _run_progressive_evaluation(args, _build_predictor)


def _evaluate(args):
    wall_start = time.perf_counter()
    predictor, total_nodes = _build_predictor(args)
    session = TimedZebraSession(predictor)
    load_s = session.now() - wall_start
    sample_started = time.perf_counter()
    manifest = load_time_slice_manifest(args.slices_dir)
    dataset_name = manifest["dataset"]
    start_t = evaluation_start_t(len(predictor.snapshots)) if args.start_t is None else args.start_t
    if not 0 <= start_t < len(predictor.snapshots) - 1:
        raise ValueError("start_t must have a next snapshot")
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
