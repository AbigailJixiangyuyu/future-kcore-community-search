"""Cached training features and batch assembly for the Ours predictor."""

import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

from datasets.coreness_prediction_builder import (
    add_core_history_tokens,
    prepare_feature_arrays_by_split,
)


def _tensor_dataset(arrays):
    return TensorDataset(
        torch.from_numpy(arrays["temporal"]),
        torch.from_numpy(arrays["core_history"]),
        torch.from_numpy(arrays["structure_indices"]),
        torch.from_numpy(arrays["time_deltas"]),
        torch.from_numpy(arrays["weights"]),
        torch.from_numpy(arrays["mask"]),
        torch.from_numpy(arrays["labels"]),
    )


def _model_batch(batch, structure_table, device):
    """Gather one batch's shared structure rows and move inputs to device."""
    (
        temporal,
        core_history,
        structure_indices,
        time_deltas,
        weights,
        mask,
        target,
    ) = batch
    neighbor_structures = torch.from_numpy(
        np.asarray(structure_table[structure_indices.numpy()])
    )
    features = [
        temporal.to(device),
        core_history.to(device),
        neighbor_structures.to(device),
        time_deltas.to(device),
        weights.to(device),
        mask.to(device),
    ]
    return features, target.to(device)


def _feature_cache_stem(kmax, hmax, config):
    max_nodes = config["max_nodes_per_time"]
    return (
        f"hybrid_features_v9_float32_current_snapshot_tsplit_v2_k{kmax}_h{hmax}_n{max_nodes}_l{config['top_l']}_"
        f"ik{config['t_ppr_internal_top_k']}_"
        f"o{config['order']}_a{config['t_ppr_alpha']}_"
        f"b{config['t_ppr_beta']}_p{config['min_probability']}_"
        f"tr{config['train_ratio']}_vr{config['val_ratio']}_"
        f"s{config['seed']}"
    )


def _load_or_prepare_features(
    cache_dir,
    samples_by_split,
    snapshots,
    kmax,
    hmax,
    config,
    t_ppr,
):
    cache_stem = _feature_cache_stem(kmax, hmax, config)
    cache_paths = {
        split: cache_dir / f"{cache_stem}_{split}.npz"
        for split in samples_by_split
    }
    structure_cache_path = cache_dir / f"{cache_stem}_structures.npy"
    if (
        structure_cache_path.exists()
        and all(path.exists() for path in cache_paths.values())
    ):
        arrays = {}
        for split, cache_path in cache_paths.items():
            print(f"[features] Loading {split}: {cache_path}")
            with np.load(str(cache_path), allow_pickle=False) as cached:
                arrays[split] = {
                    name: cached[name] for name in cached.files
                }
            add_core_history_tokens(
                arrays[split],
                snapshots,
                kmax,
                lookback=config["core_lookback"],
            )
        print(f"[features] Loading structures: {structure_cache_path}")
        structure_table = np.load(
            str(structure_cache_path), mmap_mode="r", allow_pickle=False
        )
        return arrays, structure_table

    print(
        "[features] Building all splits with one incremental T-PPR scan: "
        + " ".join(
            f"{split}={len(samples)}"
            for split, samples in samples_by_split.items()
        )
    )
    last_report = [time.time()]

    def report(done, total):
        now = time.time()
        if done == total or now - last_report[0] >= 10.0:
            print(f"  incremental features: {done}/{total}")
            last_report[0] = now

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays, structure_table = prepare_feature_arrays_by_split(
        snapshots,
        samples_by_split,
        kmax=kmax,
        hmax=hmax,
        top_l=config["top_l"],
        internal_top_k=config["t_ppr_internal_top_k"],
        order=config["order"],
        t_ppr_alpha=config["t_ppr_alpha"],
        t_ppr_beta=config["t_ppr_beta"],
        min_probability=config["min_probability"],
        t_ppr=t_ppr,
        progress=report,
        structure_table_path=structure_cache_path,
    )
    for split_arrays in arrays.values():
        add_core_history_tokens(
            split_arrays,
            snapshots,
            kmax,
            lookback=config["core_lookback"],
        )
    for split, split_arrays in arrays.items():
        cache_path = cache_paths[split]
        np.savez_compressed(str(cache_path), **split_arrays)
        print(f"[features] Cached {split}: {cache_path}")
    print(
        f"[features] Cached {len(structure_table)} shared structures: "
        f"{structure_cache_path}"
    )
    if isinstance(structure_table, np.memmap):
        del structure_table
        structure_table = np.load(
            str(structure_cache_path), mmap_mode="r", allow_pickle=False
        )
    return arrays, structure_table
