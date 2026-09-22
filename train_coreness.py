#!/usr/bin/env python3
"""Train the hybrid TCS + T-PPR next-snapshot coreness predictor."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datasets.coreness_prediction_builder import (
    STRUCTURE_TIME_REFERENCE,
    add_core_history_tokens,
    build_coreness_samples,
    prepare_feature_arrays_by_split,
)
from datasets.dataset_builder import build_snapshots
from methods.hybrid_coreness import (
    HybridCorenessPredictor,
    cumulative_ordinal_targets,
)
from methods.t_ppr import TemporalPPR


EVALUATION_KS = tuple(range(3, 8))


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        f"hybrid_features_v9_float32_current_snapshot_k{kmax}_h{hmax}_n{max_nodes}_l{config['top_l']}_"
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


def _binary_metrics(prediction, truth):
    prediction = prediction.astype(np.bool_)
    truth = truth.astype(np.bool_)
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ordinal_positive_weights(labels, kmax, power=0.5, cap=10.0):
    """Return capped positive weights for cumulative threshold targets."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("ordinal weight power must be in [0, 1]")
    if cap < 1.0:
        raise ValueError("ordinal weight cap must be at least 1")
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.arange(1, kmax + 1, dtype=np.int64)
    positives = np.sum(labels[:, None] >= thresholds[None, :], axis=0)
    negatives = len(labels) - positives
    weights = np.ones(kmax, dtype=np.float32)
    valid = (positives > 0) & (negatives > positives)
    weights[valid] = np.minimum(
        np.power(negatives[valid] / positives[valid], power), cap
    ).astype(np.float32)
    return torch.from_numpy(weights)


def evaluate_model(model, loader, structure_table, criterion, device, kmax):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            features, target = _model_batch(batch, structure_table, device)
            class_logits = model(*features)
            ordinal_logits = model.ordinal_logits(class_logits)
            ordinal_target = cumulative_ordinal_targets(target, kmax)
            total_loss += (
                criterion(ordinal_logits, ordinal_target).item() * target.size(0)
            )
            predictions.append(
                (ordinal_logits > 0.0).sum(dim=1).cpu().numpy()
            )
            labels.append(target.cpu().numpy())

    if not labels:
        raise ValueError("evaluation split has no samples")
    prediction = np.concatenate(predictions)
    truth = np.concatenate(labels)
    result = {
        "loss": total_loss / len(truth),
        "accuracy": float(np.mean(prediction == truth, dtype=np.float32)),
        "mae": float(np.mean(np.abs(prediction - truth), dtype=np.float32)),
        "per_k": {},
    }
    for k in EVALUATION_KS:
        if k <= kmax:
            result["per_k"][k] = _binary_metrics(prediction >= k, truth >= k)
    return result


def _print_metrics(name, metrics):
    print(
        f"[{name}] loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} mae={metrics['mae']:.4f}"
    )
    for k, values in metrics["per_k"].items():
        print(
            f"  k={k}: accuracy={values['accuracy']:.4f} "
            f"precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} f1={values['f1']:.4f}"
        )


def train(args):
    _set_seed(args.seed)
    slices_dir = Path(args.input)
    snapshots, total_nodes, kmax, hmax = build_snapshots(slices_dir)
    print(
        f"[dataset] snapshots={len(snapshots)} nodes={total_nodes} "
        f"kmax={kmax} hmax={hmax} "
        f"structure_buckets=0..{hmax - 1},>={hmax}"
    )

    samples = build_coreness_samples(
        snapshots,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_nodes_per_time=args.max_nodes_per_time,
        seed=args.seed,
    )
    print(
        "[samples] "
        + " ".join(f"{split}={len(items)}" for split, items in samples.items())
    )
    if any(not samples[split] for split in ("train", "val", "test")):
        raise ValueError("train, validation, and test splits must all be non-empty")

    feature_config = {
        "numeric_dtype": "float32",
        "structure_time_reference": STRUCTURE_TIME_REFERENCE,
        "top_l": args.top_l,
        "t_ppr_internal_top_k": args.t_ppr_internal_top_k,
        "order": args.order,
        "hmax": hmax,
        "t_ppr_alpha": args.t_ppr_alpha,
        "t_ppr_beta": args.t_ppr_beta,
        "min_probability": args.min_probability,
        "core_lookback": args.core_lookback,
        "max_nodes_per_time": args.max_nodes_per_time,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    t_ppr = TemporalPPR(
        snapshots,
        alpha=args.t_ppr_alpha,
        beta=args.t_ppr_beta,
    )
    cache_dir = slices_dir / "model_cache" / "features"
    arrays, structure_table = _load_or_prepare_features(
        cache_dir,
        samples,
        snapshots,
        kmax,
        hmax,
        feature_config,
        t_ppr,
    )

    loaders = {
        split: DataLoader(
            _tensor_dataset(arrays[split]),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for split in arrays
    }
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"[train] device={device}")

    model_config = {
        "kmax": kmax,
        "hmax": hmax,
        "order": args.order,
        "time_dim": args.time_dim,
        "structure_hidden": args.structure_hidden,
        "temporal_hidden": args.temporal_hidden,
        "fusion_hidden": args.fusion_hidden,
        "core_dim": args.core_dim,
        "core_lookback": args.core_lookback,
        "bucket_dim": args.bucket_dim,
        "attention_heads": args.attention_heads,
        "persistence_scale": args.persistence_scale,
        "dropout": args.dropout,
    }
    model = HybridCorenessPredictor(**model_config).to(device)
    positive_weights = _ordinal_positive_weights(
        arrays["train"]["labels"],
        kmax,
        power=args.ordinal_weight_power,
        cap=args.ordinal_weight_cap,
    ).to(device)
    print(
        "[train] ordinal_pos_weights="
        + ",".join(f"{value:.4f}" for value in positive_weights.tolist())
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loaders["train"]:
            features, target = _model_batch(batch, structure_table, device)
            optimizer.zero_grad()
            class_logits = model(*features)
            ordinal_logits = model.ordinal_logits(class_logits)
            ordinal_target = cumulative_ordinal_targets(target, kmax)
            loss = criterion(ordinal_logits, ordinal_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            total_count += target.size(0)

        validation = evaluate_model(
            model, loaders["val"], structure_table, criterion, device, kmax
        )
        print(
            f"[epoch {epoch:03d}] train_loss={total_loss / total_count:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_accuracy={validation['accuracy']:.4f}"
        )
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[train] Early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    validation = evaluate_model(
        model, loaders["val"], structure_table, criterion, device, kmax
    )
    test = evaluate_model(
        model, loaders["test"], structure_table, criterion, device, kmax
    )
    _print_metrics("validation", validation)
    _print_metrics("test", test)

    output_path = Path(args.output) if args.output else (
        slices_dir / "model_cache" / "hybrid_coreness.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_config": dict(
            model_config, structure_pooling="b", fusion_type="concat",
            output_head_type="linear",
        ),
        "feature_config": feature_config,
        "objective": {
            "name": "hybrid_cumulative_ordinal_bce",
            "thresholds": list(range(1, kmax + 1)),
            "decision_logit": 0.0,
            "monotone": True,
            "positive_weights": positive_weights.cpu().tolist(),
            "positive_weight_power": args.ordinal_weight_power,
            "positive_weight_cap": args.ordinal_weight_cap,
        },
        "split_config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
        "training_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
        },
        "metrics": {"validation": validation, "test": test},
    }
    torch.save(checkpoint, str(output_path))
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "objective": checkpoint["objective"],
                "model_config": checkpoint["model_config"],
                "training_config": checkpoint["training_config"],
                **checkpoint["metrics"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    print(f"[train] Saved model to {output_path}")
    return model, checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Directory generated by build_time_slices")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-nodes-per-time", type=int, default=2000)
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--t-ppr-internal-top-k", type=int, default=20)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--t-ppr-alpha", type=float, default=0.3)
    parser.add_argument("--t-ppr-beta", type=float, default=0.5)
    parser.add_argument("--min-probability", type=float, default=1e-8)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--structure-hidden", type=int, default=64)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--fusion-hidden", type=int, default=128)
    parser.add_argument("--core-dim", type=int, default=32)
    parser.add_argument("--core-lookback", type=int, default=5)
    parser.add_argument("--bucket-dim", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--persistence-scale", type=float, default=2.0)
    parser.add_argument("--ordinal-weight-power", type=float, default=0.5)
    parser.add_argument("--ordinal-weight-cap", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
