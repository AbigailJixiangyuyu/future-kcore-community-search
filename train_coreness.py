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
    build_coreness_samples,
    prepare_feature_arrays_by_split,
)
from datasets.dataset_builder import build_snapshots
from methods.hybrid_coreness import HybridCorenessPredictor
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
        torch.from_numpy(arrays["neighbor_structures"]),
        torch.from_numpy(arrays["time_deltas"]),
        torch.from_numpy(arrays["weights"]),
        torch.from_numpy(arrays["mask"]),
        torch.from_numpy(arrays["labels"]),
    )


def _feature_cache_name(split, kmax, hmax, config):
    max_nodes = config["max_nodes_per_time"]
    return (
        f"hybrid_features_v5_{split}_k{kmax}_h{hmax}_n{max_nodes}_l{config['top_l']}_"
        f"ik{config['t_ppr_internal_top_k']}_"
        f"o{config['order']}_a{config['t_ppr_alpha']}_"
        f"b{config['t_ppr_beta']}_p{config['min_probability']}_"
        f"tr{config['train_ratio']}_vr{config['val_ratio']}_"
        f"s{config['seed']}.npz"
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
    cache_paths = {
        split: cache_dir / _feature_cache_name(split, kmax, hmax, config)
        for split in samples_by_split
    }
    if all(path.exists() for path in cache_paths.values()):
        arrays = {}
        for split, cache_path in cache_paths.items():
            print(f"[features] Loading {split}: {cache_path}")
            with np.load(str(cache_path), allow_pickle=False) as cached:
                arrays[split] = {
                    name: cached[name] for name in cached.files
                }
        return arrays

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

    arrays = prepare_feature_arrays_by_split(
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
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, split_arrays in arrays.items():
        cache_path = cache_paths[split]
        np.savez_compressed(str(cache_path), **split_arrays)
        print(f"[features] Cached {split}: {cache_path}")
    return arrays


def _class_weights(labels, class_count):
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    weights = np.zeros(class_count, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (
        present.sum() * counts[present]
    )
    return torch.from_numpy(weights)


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


def evaluate_model(model, loader, criterion, device, kmax):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            *features, target = [tensor.to(device) for tensor in batch]
            logits = model(*features)
            total_loss += criterion(logits, target).item() * target.size(0)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(target.cpu().numpy())

    if not labels:
        raise ValueError("evaluation split has no samples")
    prediction = np.concatenate(predictions)
    truth = np.concatenate(labels)
    result = {
        "loss": total_loss / len(truth),
        "accuracy": float(np.mean(prediction == truth)),
        "mae": float(np.mean(np.abs(prediction - truth))),
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
        "top_l": args.top_l,
        "t_ppr_internal_top_k": args.t_ppr_internal_top_k,
        "order": args.order,
        "hmax": hmax,
        "t_ppr_alpha": args.t_ppr_alpha,
        "t_ppr_beta": args.t_ppr_beta,
        "min_probability": args.min_probability,
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
    arrays = _load_or_prepare_features(
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
        "dropout": args.dropout,
    }
    model = HybridCorenessPredictor(**model_config).to(device)
    weights = _class_weights(arrays["train"]["labels"], kmax + 1).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loaders["train"]:
            *features, target = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            logits = model(*features)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            total_count += target.size(0)

        validation = evaluate_model(
            model, loaders["val"], criterion, device, kmax
        )
        print(
            f"[epoch {epoch:03d}] train_loss={total_loss / total_count:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_accuracy={validation['accuracy']:.4f}"
        )
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[train] Early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    validation = evaluate_model(model, loaders["val"], criterion, device, kmax)
    test = evaluate_model(model, loaders["test"], criterion, device, kmax)
    _print_metrics("validation", validation)
    _print_metrics("test", test)

    output_path = Path(args.output) if args.output else (
        slices_dir / "model_cache" / "hybrid_coreness.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "feature_config": feature_config,
        "split_config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
        "metrics": {"validation": validation, "test": test},
    }
    torch.save(checkpoint, str(output_path))
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(checkpoint["metrics"], indent=2, sort_keys=True) + "\n"
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
    parser.add_argument("--t-ppr-internal-top-k", type=int, default=80)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--t-ppr-alpha", type=float, default=0.3)
    parser.add_argument("--t-ppr-beta", type=float, default=0.5)
    parser.add_argument("--min-probability", type=float, default=1e-8)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--structure-hidden", type=int, default=64)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--fusion-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
