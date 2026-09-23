"""Ordinal training weights and held-out coreness metrics."""

import numpy as np
import torch

from methods.hybrid_coreness import cumulative_ordinal_targets
from training.ours_features import _model_batch


EVALUATION_KS = tuple(range(3, 8))


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
