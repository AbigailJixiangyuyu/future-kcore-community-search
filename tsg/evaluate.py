import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def top_m_infer(scores, num_target_edges):
    M = num_target_edges
    top_indices = np.argsort(scores)[::-1][:M]
    return top_indices


def compute_metrics(scores, labels):
    num_pos = int(labels.sum())
    if num_pos == 0:
        return {"auc": 0.0, "ap": 0.0, "precision_at_m": 0.0, "recall_at_m": 0.0}

    M = num_pos
    top_indices = top_m_infer(scores, M)

    hits = labels[top_indices].sum()
    precision_at_m = hits / M
    recall_at_m = hits / num_pos

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "precision_at_m": float(precision_at_m),
        "recall_at_m": float(recall_at_m),
    }
