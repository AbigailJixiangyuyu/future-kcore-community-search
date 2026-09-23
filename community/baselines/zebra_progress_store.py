"""Persist progressive samples and summarize resumable Zebra runs."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from community.baselines.baseline_graph import historical_community_union
from community.baselines.zebra_runtime import HistoricalEdgeIndex


EMBEDDING_CACHE_VERSION = 1
PROGRESSIVE_RESULT_VERSION = 2
PROGRESSIVE_KS = (7, 6, 5, 4, 3)
COMMUNITY_METRICS = (
    "precision", "recall", "f1", "jaccard", "size_ratio", "pred_ratio", "elapsed_s",
)


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
