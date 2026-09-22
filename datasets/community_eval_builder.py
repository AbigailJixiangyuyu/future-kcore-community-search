#!/usr/bin/env python3
"""
Community Evaluation Dataset Builder
====================================
Builds fixed (q, k, t) samples with ground-truth k-core communities in the
next snapshot. Predictors may use snapshots through t and predict the connected
k-core component containing q in snapshot t + 1.

Sampling strategy: coreness-weighted. Nodes that remain in the k-core are
sampled most heavily, with a smaller number of nodes that leave the k-core.

    weight(q, k, t) = coreness(q, t) - k + 1

Usage (CLI):
    python -m datasets.community_eval_builder <time_slices_dir> [--max-per-kt 20]

Usage (library):
    from datasets.community_eval_builder import build_community_eval_dataset
    data = build_community_eval_dataset(
        "data/email-Eu-core-temporal/time_slices/step_604800_window_604800"
    )
    for s in data["samples"]:
        print(s["query"], s["k"], s["t"], len(s["community"]))
"""

import argparse
import os
import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np

from datasets.dataset_builder import (
    TARGET_KS,
    DATASET_VALID_KS,
    DEFAULT_TEST_RATIO,
    build_snapshots,
    load_time_slice_manifest,
)

DEFAULT_MAX_PER_KT = 20
DEFAULT_SEED = 42
DATASET_SAMPLE_FRACTIONS = {
    "wiki-talk-temporal": 1 / 6,
}


def _get_community(snaps, t, q, k):
    k_info = snaps[t]["k_core_comps"].get(k)
    if k_info is None or q not in k_info["node_set"]:
        return frozenset()
    for component in k_info["components"]:
        if q in component:
            return frozenset(component)
    return frozenset()


def _weighted_sample(rng, candidates, cur_cd, k, n):
    if len(candidates) <= n:
        return list(candidates)
    weights = np.array(
        [cur_cd.get(q, 0) - k + 1 for q in candidates], dtype=np.float32
    )
    weights = np.maximum(weights, 1.0)
    weights /= weights.sum()
    chosen_idx = rng.choice(len(candidates), size=n, replace=False, p=weights)
    return [candidates[i] for i in chosen_idx]


def _resolve_sample_fraction(dataset_name, sample_fraction):
    if sample_fraction is None:
        sample_fraction = DATASET_SAMPLE_FRACTIONS.get(dataset_name, 1.0)
    sample_fraction = float(sample_fraction)
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    return sample_fraction


def _fraction_cache_suffix(sample_fraction):
    if sample_fraction == 1.0:
        return ""
    numerator, denominator = (
        Fraction(sample_fraction).limit_denominator().as_integer_ratio()
    )
    return f"_f{numerator}of{denominator}"


def _sample_cache_key(dataset_name, split_ti, valid_ks, max_per_kt,
                      empty_per_kt, seed, sample_fraction=1.0):
    ks_str = "_".join(str(k) for k in sorted(valid_ks))
    fraction_suffix = _fraction_cache_suffix(sample_fraction)
    return (
        f"community_samples_v1_{dataset_name}_split{split_ti}_k{ks_str}_"
        f"n{max_per_kt}_e{empty_per_kt}_s{seed}{fraction_suffix}.pkl"
    )


def stratified_downsample(samples, sample_fraction, seed=DEFAULT_SEED):
    """Select an exact fraction while preserving (t, k, truth-empty) strata."""
    sample_fraction = _resolve_sample_fraction(None, sample_fraction)
    if sample_fraction == 1.0 or not samples:
        return list(samples)

    target_size = max(1, round(len(samples) * sample_fraction))
    strata = {}
    for index, sample in enumerate(samples):
        key = (sample["t"], sample["k"], not bool(sample["community"]))
        strata.setdefault(key, []).append(index)

    rng = np.random.RandomState(seed)
    allocations = {}
    remainders = []
    allocated = 0
    for key, indices in sorted(strata.items()):
        exact = len(indices) * sample_fraction
        count = int(np.floor(exact))
        allocations[key] = count
        allocated += count
        remainders.append((exact - count, rng.random_sample(), key))

    for _remainder, _tie_breaker, key in sorted(
        remainders, key=lambda item: (-item[0], item[1])
    )[:target_size - allocated]:
        allocations[key] += 1

    selected = []
    for key, indices in sorted(strata.items()):
        count = allocations[key]
        if count == len(indices):
            selected.extend(indices)
        elif count:
            selected.extend(
                rng.choice(indices, size=count, replace=False).tolist()
            )
    return [samples[index] for index in sorted(selected)]


def sample_qk_coreness_weighted(snaps, split_ti, valid_ks,
                                max_per_kt=DEFAULT_MAX_PER_KT,
                                empty_per_kt=5,
                                seed=DEFAULT_SEED,
                                dataset_name=None,
                                cache_dir=None,
                                sample_fraction=None):
    if not 0 <= split_ti < len(snaps) - 1:
        raise ValueError("Community evaluation requires a current and next snapshot")
    sample_fraction = _resolve_sample_fraction(dataset_name, sample_fraction)
    cache_path = None
    if dataset_name is not None and cache_dir is not None:
        cache_key = _sample_cache_key(dataset_name, split_ti, valid_ks,
                                      max_per_kt, empty_per_kt, seed,
                                      sample_fraction)
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            print(f"[community_eval] Loading cached samples from {cache_path.name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        full_cache_path = cache_dir / _sample_cache_key(
            dataset_name, split_ti, valid_ks, max_per_kt, empty_per_kt, seed
        )
        if sample_fraction < 1.0 and full_cache_path.exists():
            print(
                "[community_eval] Downsampling cached samples from "
                f"{full_cache_path.name}"
            )
            with open(full_cache_path, "rb") as f:
                samples = pickle.load(f)
            samples = stratified_downsample(samples, sample_fraction, seed)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[community_eval] Cached samples to {cache_path.name}")
            return samples

    rng = np.random.RandomState(seed)
    samples = []

    for t in range(split_ti, len(snaps) - 1):
        current_cd = snaps[t]["core_dict"]
        next_cd = snaps[t + 1]["core_dict"]

        for k in valid_ks:
            current_k_info = snaps[t]["k_core_comps"].get(k)
            if current_k_info is None:
                continue
            current_k_nodes = list(current_k_info["node_set"])
            if not current_k_nodes:
                continue

            stayers = [q for q in current_k_nodes if next_cd.get(q, 0) >= k]
            leavers = [q for q in current_k_nodes if next_cd.get(q, 0) < k]

            stayers_count = min(max_per_kt, len(stayers))
            for q in _weighted_sample(rng, stayers, current_cd, k, stayers_count):
                samples.append({
                    "query": q,
                    "k": k,
                    "t": t,
                    "community": _get_community(snaps, t + 1, q, k),
                })

            if leavers and stayers_count > 0:
                leavers_count = min(
                    empty_per_kt,
                    max(1, stayers_count // 4),
                    len(leavers),
                )
                for q in _weighted_sample(
                    rng, leavers, current_cd, k, leavers_count
                ):
                    samples.append({
                        "query": q,
                        "k": k,
                        "t": t,
                        "community": frozenset(),
                    })

    samples = stratified_downsample(samples, sample_fraction, seed)

    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[community_eval] Cached samples to {cache_path.name}")

    return samples


def build_community_eval_dataset(slices_dir,
                                 test_ratio=DEFAULT_TEST_RATIO,
                                 max_per_kt=DEFAULT_MAX_PER_KT,
                                 seed=DEFAULT_SEED,
                                 save_dir=None,
                                 sample_fraction=None):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    slices_dir = Path(slices_dir)
    slice_manifest = load_time_slice_manifest(slices_dir)
    dataset_name = slice_manifest["dataset"]
    sample_fraction = _resolve_sample_fraction(dataset_name, sample_fraction)
    valid_ks = DATASET_VALID_KS.get(dataset_name, TARGET_KS)
    save_dir = save_dir or slices_dir / "community_eval"

    print(f"[community_eval] Loading {slices_dir} ...")
    snaps, total_nodes, kmax, _hmax = build_snapshots(slices_dir)
    print(f"  {len(snaps)} active snapshots, max coreness={kmax}")

    split_ti = int(len(snaps) * (1 - test_ratio))
    if not 0 <= split_ti < len(snaps) - 1:
        raise ValueError("Community evaluation requires at least two snapshots")
    print(f"  Eval current snapshots: {split_ti}-{len(snaps) - 2}")

    print(
        "[community_eval] Sampling "
        f"(max_per_kt={max_per_kt}, fraction={sample_fraction:g}, seed={seed}) ..."
    )
    samples = sample_qk_coreness_weighted(
        snaps, split_ti, valid_ks, max_per_kt, seed=seed, dataset_name=dataset_name,
        cache_dir=slices_dir / "sample_cache",
        sample_fraction=sample_fraction,
    )
    print(f"  {len(samples)} samples collected.")

    per_k_counts = {}
    for s in samples:
        per_k_counts.setdefault(s["k"], 0)
        per_k_counts[s["k"]] += 1
    per_k_str = ", ".join(f"k={k}:{n}" for k, n in sorted(per_k_counts.items()))
    print(f"  Distribution: {per_k_str}")

    non_empty = sum(1 for sample in samples if sample["community"])
    print(f"  Non-empty communities: {non_empty}, empty: {len(samples) - non_empty}")

    result = {
        "metadata": {
            "valid_ks": valid_ks,
            "sample_fraction": sample_fraction,
            "total_nodes": total_nodes,
            "kmax": kmax,
            "time_slice_config": {
                "step_seconds": slice_manifest["step_seconds"],
                "window_seconds": slice_manifest["window_seconds"],
            },
        },
        "samples": samples,
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset_name}.pkl")
    temp_path = f"{pkl_path}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, pkl_path)
    print(f"[community_eval] Saved to {pkl_path}")

    return result


def load_community_eval_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_metrics(prediction, truth):
    prediction = set(prediction)
    truth = set(truth)
    if not prediction and not truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    if not prediction or not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}
    intersection_size = len(prediction & truth)
    precision = intersection_size / len(prediction)
    recall = intersection_size / len(truth)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if intersection_size else 0.0,
        "jaccard": intersection_size / len(prediction | truth),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build community evaluation dataset"
    )
    parser.add_argument("input", help="Directory generated by datasets.build_time_slices")
    parser.add_argument("--max-per-kt", type=int, default=DEFAULT_MAX_PER_KT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    build_community_eval_dataset(
        args.input,
        test_ratio=args.test_ratio,
        max_per_kt=args.max_per_kt,
        seed=args.seed,
        save_dir=args.save_dir,
        sample_fraction=args.sample_fraction,
    )


if __name__ == "__main__":
    main()
