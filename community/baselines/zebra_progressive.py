"""Resumable Zebra evaluation and per-snapshot embedding caches."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from community.baselines.zebra_progress_store import (
    _atomic_save_npy,
    _atomic_write_json,
    _build_progress_payload,
    _load_completed_records,
    _now,
    _prepare_progressive_samples,
    _progressive_run_signature,
    _sample_result_path,
    _write_comparison_markdown,
)
from datasets.community_eval_builder import sample_qk_coreness_weighted, set_metrics
from datasets.dataset_builder import load_time_slice_manifest


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


def _progressive_evaluate(args, build_predictor):
    predictor, total_nodes = build_predictor(args)
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
