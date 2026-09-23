"""Fit the Ours coreness predictor and persist its best checkpoint."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.coreness_prediction_builder import (
    STRUCTURE_TIME_REFERENCE,
    build_coreness_samples,
    prediction_split_config,
)
from datasets.dataset_builder import build_snapshots
from methods.hybrid_coreness import (
    HybridCorenessPredictor,
    cumulative_ordinal_targets,
)
from methods.t_ppr import TemporalPPR
from training.ours_features import (
    _load_or_prepare_features,
    _model_batch,
    _tensor_dataset,
)
from training.ours_metrics import (
    _ordinal_positive_weights,
    _print_metrics,
    evaluate_model,
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    _set_seed(args.seed)
    slices_dir = Path(args.input)
    snapshots, total_nodes, kmax, hmax = build_snapshots(slices_dir)
    print(
        f"[dataset] snapshots={len(snapshots)} nodes={total_nodes} "
        f"kmax={kmax} hmax={hmax} "
        f"structure_buckets=0..{hmax - 1},>={hmax}"
    )

    split_config = prediction_split_config(
        len(snapshots), args.train_ratio, args.val_ratio
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
        "split_rule": split_config["split_rule"],
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
        "split_config": split_config,
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
                "split_config": checkpoint["split_config"],
                "training_config": checkpoint["training_config"],
                **checkpoint["metrics"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    print(f"[train] Saved model to {output_path}")
    return model, checkpoint
