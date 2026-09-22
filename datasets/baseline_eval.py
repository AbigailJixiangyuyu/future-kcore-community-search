"""Validate the persisted, shared community test cohort for baselines."""

from pathlib import Path
import hashlib
import json

from datasets.baseline_split import test_start_t
from datasets.community_eval_builder import load_community_eval_dataset
from datasets.dataset_builder import load_time_slice_manifest


def require_fit_boundary(fit_end_t, snapshot_count):
    start = test_start_t(snapshot_count)
    if fit_end_t != start:
        raise ValueError(
            "checkpoint fit_end_t={} does not match 7:3 test_start_t={}; "
            "retrain with the snapshot split".format(fit_end_t, start)
        )
    return start


def query_set_sha256(samples):
    queries = sorted((int(s["query"]), int(s["k"]), int(s["t"])) for s in samples)
    return hashlib.sha256(json.dumps(queries).encode("ascii")).hexdigest()


def load_test_samples(slices_dir, snapshot_count, path=None):
    slices_dir = Path(slices_dir)
    manifest = load_time_slice_manifest(slices_dir)
    if len(manifest["slices"]) != snapshot_count:
        raise ValueError("snapshot count differs from slice manifest")
    expected = slices_dir / "community_eval" / (manifest["dataset"] + ".pkl")
    if path is not None and Path(path).resolve() != expected.resolve():
        raise ValueError("evaluation must use the shared community_eval pickle")
    data = load_community_eval_dataset(expected)
    if data["metadata"]["time_slice_config"] != {
        "step_seconds": manifest["step_seconds"],
        "window_seconds": manifest["window_seconds"],
    } or data["metadata"]["valid_ks"] != [3, 4, 5, 6, 7]:
        raise ValueError("community evaluation metadata does not match slices")
    samples = data["samples"]
    start = test_start_t(snapshot_count)
    seen = set()
    if not samples:
        raise ValueError("community evaluation set is empty")
    for sample in samples:
        q, k, t = (int(sample[field]) for field in ("query", "k", "t"))
        if (not start <= t < snapshot_count - 1 or k not in range(3, 8)
                or (q, k, t) in seen or (sample["community"] and q not in sample["community"])):
            raise ValueError("invalid or duplicate held-out community query")
        seen.add((q, k, t))
    return samples
