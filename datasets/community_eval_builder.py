#!/usr/bin/env python3
"""
Community Evaluation Dataset Builder
=====================================
Builds a fixed evaluation set of (q, k, t) samples with ground-truth
k-core communities for comparing different community prediction methods.

Sampling strategy: coreness-weighted — nodes closer to the dense core
are sampled more often, since predicting their communities is more
practically meaningful.

    weight(q, k, t) = coreness(q, t) - k + 1

Usage (CLI):
    python -m datasets.community_eval_builder <edge_list> [--max-per-kt 20]

Usage (library):
    from datasets.community_eval_builder import build_community_eval_dataset
    data = build_community_eval_dataset("datasets/email-Eu-core-temporal.csv")
    for s in data["samples"]:
        print(s["query"], s["k"], s["t"], len(s["community"]))
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from datasets.dataset_builder import (
    DAY,
    THREE_DAY,
    WEEK,
    MONTH,
    TARGET_KS,
    DATASET_VALID_KS,
    DATASET_WINDOW,
    DEFAULT_TEST_RATIO,
    load_edges,
    build_snapshots,
)

DEFAULT_MAX_PER_KT = 20
DEFAULT_SEED = 42
SAVE_DIR = os.path.join(os.path.dirname(__file__), "community_eval")
SAMPLE_CACHE_DIR = Path(__file__).parent / "sample_cache"


def _get_community(snaps, t, q, k):
    k_info = snaps[t]["k_core_comps"].get(k)
    if k_info is None:
        return frozenset()
    if q not in k_info["node_set"]:
        return frozenset()
    for comp in k_info["components"]:
        if q in comp:
            return frozenset(comp)
    return frozenset()


def _weighted_sample(rng, candidates, cur_cd, k, n):
    if len(candidates) <= n:
        return list(candidates)
    weights = np.array(
        [cur_cd.get(q, 0) - k + 1 for q in candidates], dtype=np.float64
    )
    weights = np.maximum(weights, 1.0)
    weights /= weights.sum()
    chosen_idx = rng.choice(len(candidates), size=n, replace=False, p=weights)
    return [candidates[i] for i in chosen_idx]


def _sample_cache_key(dataset_name, split_ti, valid_ks, max_per_kt, empty_per_kt, seed):
    ks_str = "_".join(str(k) for k in sorted(valid_ks))
    return f"samples_{dataset_name}_split{split_ti}_k{ks_str}_n{max_per_kt}_e{empty_per_kt}_s{seed}.pkl"


def sample_qk_coreness_weighted(snaps, split_ti, valid_ks,
                                max_per_kt=DEFAULT_MAX_PER_KT,
                                empty_per_kt=5,
                                seed=DEFAULT_SEED,
                                dataset_name=None):
    if dataset_name is not None:
        cache_key = _sample_cache_key(dataset_name, split_ti, valid_ks,
                                      max_per_kt, empty_per_kt, seed)
        cache_path = SAMPLE_CACHE_DIR / cache_key
        if cache_path.exists():
            print(f"[community_eval] Loading cached samples from {cache_path.name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    rng = np.random.RandomState(seed)
    samples = []

    for t in range(split_ti, len(snaps) - 1):
        cur_cd = snaps[t]["core_dict"]
        nxt_cd = snaps[t + 1]["core_dict"]

        for k in valid_ks:
            cur_k_info = snaps[t]["k_core_comps"].get(k)
            if cur_k_info is None:
                # print("no k info in ", k)
                continue
            cur_k_nodes = list(cur_k_info["node_set"])
            if not cur_k_nodes:
                continue

            stayers = [q for q in cur_k_nodes if nxt_cd.get(q, 0) >= k]
            leavers = [q for q in cur_k_nodes if nxt_cd.get(q, 0) < k]

            n_stayers_sampled = min(max_per_kt, len(stayers))
            for q in _weighted_sample(rng, stayers, cur_cd, k, n_stayers_sampled):
                community = _get_community(snaps, t + 1, q, k)
                samples.append({
                    "query": q,
                    "k": k,
                    "t": t,
                    "community": community,
                })

            if leavers and n_stayers_sampled > 0:
                n_empty = min(max(1, n_stayers_sampled // 4), len(leavers))
                for q in _weighted_sample(rng, leavers, cur_cd, k, n_empty):
                    samples.append({
                        "query": q,
                        "k": k,
                        "t": t,
                        "community": frozenset(),
                    })

    if dataset_name is not None:
        SAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[community_eval] Cached samples to {cache_path.name}")

    return samples


def build_community_eval_dataset(edge_path,
                                 test_ratio=DEFAULT_TEST_RATIO,
                                 window=None,
                                 max_per_kt=DEFAULT_MAX_PER_KT,
                                 seed=DEFAULT_SEED,
                                 save_dir=None):
    dataset_name = Path(edge_path).stem
    valid_ks = DATASET_VALID_KS.get(dataset_name, TARGET_KS)
    if window is None:
        window = DATASET_WINDOW.get(dataset_name, WEEK)
    save_dir = save_dir or SAVE_DIR

    print(f"[community_eval] Loading {edge_path} ...")
    snaps, total_nodes = build_snapshots(edge_path, window)
    print(f"  {len(snaps)} active snapshots, "
          f"max coreness={max(s['max_core'] for s in snaps)}")

    split_ti = int(len(snaps) * 0.7)
    print(f"  Eval: snaps {split_ti}-{len(snaps) - 1}")

    print(f"[community_eval] Sampling (max_per_kt={max_per_kt}, seed={seed}) ...")
    samples = sample_qk_coreness_weighted(
        snaps, split_ti, valid_ks, max_per_kt, seed=seed, dataset_name=dataset_name
    )
    print(f"  {len(samples)} samples collected.")

    per_k_counts = {}
    for s in samples:
        per_k_counts.setdefault(s["k"], 0)
        per_k_counts[s["k"]] += 1
    per_k_str = ", ".join(f"k={k}:{n}" for k, n in sorted(per_k_counts.items()))
    print(f"  Distribution: {per_k_str}")

    non_empty = sum(1 for s in samples if len(s["community"]) > 0)
    empty = len(samples) - non_empty
    print(f"  Non-empty communities: {non_empty}, empty: {empty}")

    cumulative = {}
    snap_idx = 0
    ug_adj = {}
    for t in range(len(snaps)):
        while snap_idx <= t:
            for u, v, _ in snaps[snap_idx]["edge_list"]:
                if u not in cumulative:
                    cumulative[u] = set()
                if v not in cumulative:
                    cumulative[v] = set()
                cumulative[u].add(v)
                cumulative[v].add(u)
            snap_idx += 1
        ug_adj[t] = {n: list(nb) for n, nb in cumulative.items()}

    result = {
        "metadata": {
            "valid_ks": valid_ks,
            "total_nodes": total_nodes,
        },
        "snapshots": snaps,
        "samples": samples,
        "ug_adj": ug_adj,
    }

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, f"{dataset_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[community_eval] Saved to {pkl_path}")

    return result


def load_community_eval_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def set_metrics(pred, truth):
    pred = set(pred)
    truth = set(truth)
    if not pred and not truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    if not pred or not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}
    inter = len(pred & truth)
    prec = inter / len(pred)
    rec = inter / len(truth)
    f1 = 2 * prec * rec / (prec + rec)
    jacc = inter / len(pred | truth)
    return {"precision": prec, "recall": rec, "f1": f1, "jaccard": jacc}


def main():
    parser = argparse.ArgumentParser(
        description="Build community evaluation dataset"
    )
    parser.add_argument("input", help="Edge list file")
    parser.add_argument("--max-per-kt", type=int, default=DEFAULT_MAX_PER_KT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window", choices=["day", "3day", "week", "month"],
                        default=None)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    window_map = {"day": DAY, "3day": THREE_DAY, "week": WEEK, "month": MONTH}

    build_community_eval_dataset(
        args.input,
        test_ratio=args.test_ratio,
        window=window_map.get(args.window) if args.window else None,
        max_per_kt=args.max_per_kt,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
