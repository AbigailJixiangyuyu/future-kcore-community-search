"""Trainable hybrid TCS + T-PPR structural coreness predictor."""

from collections import deque

import numpy as np
import torch
from torch import nn


class HybridCorenessPredictor(nn.Module):
    """Predict next-snapshot coreness from temporal and structural features."""

    def __init__(
        self,
        kmax,
        hmax=None,
        order=4,
        time_dim=16,
        structure_hidden=64,
        temporal_hidden=64,
        fusion_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        if kmax <= 0:
            raise ValueError("kmax must be positive")
        if hmax is None:
            hmax = kmax
        if hmax < 0:
            raise ValueError("hmax must be non-negative")
        if not 1 <= order <= 4:
            raise ValueError("order must be in [1, 4]")

        self.kmax = int(kmax)
        self.hmax = int(hmax)
        self.order = int(order)
        self.structure_width = order * (hmax + 1)

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
        )
        self.structure_transform = nn.Sequential(
            nn.Linear(self.structure_width + time_dim, structure_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.Sequential(
            nn.LayerNorm(kmax),
            nn.Linear(kmax, temporal_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.predictor = nn.Sequential(
            nn.Linear(temporal_hidden + structure_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, kmax + 1),
        )

    def forward(
        self,
        temporal,
        neighbor_structures,
        time_deltas,
        weights,
        mask,
    ):
        """Return logits for coreness classes ``0..kmax``."""
        if temporal.dim() != 2 or temporal.size(1) != self.kmax:
            raise ValueError("temporal input has an invalid shape")
        if neighbor_structures.dim() != 3:
            raise ValueError("neighbor_structures must be a 3D tensor")
        if neighbor_structures.size(2) != self.structure_width:
            raise ValueError("neighbor structure width does not match the model")
        if time_deltas.shape != neighbor_structures.shape[:2]:
            raise ValueError("time_deltas shape must match neighbor slots")
        if weights.shape != time_deltas.shape or mask.shape != time_deltas.shape:
            raise ValueError("weights and mask must match time_deltas")

        encoded_time = self.time_encoder(
            torch.log1p(time_deltas.clamp_min(0.0)).unsqueeze(-1)
        )
        transformed = self.structure_transform(
            torch.cat((neighbor_structures, encoded_time), dim=-1)
        )

        effective_weights = weights * mask.to(weights.dtype)
        weight_sum = effective_weights.sum(dim=1, keepdim=True)
        normalized_weights = torch.where(
            weight_sum > 0,
            effective_weights / weight_sum.clamp_min(1e-12),
            torch.zeros_like(effective_weights),
        )
        structural = torch.sum(
            normalized_weights.unsqueeze(-1) * transformed,
            dim=1,
        )
        temporal_state = self.temporal_encoder(temporal)
        return self.predictor(torch.cat((temporal_state, structural), dim=1))

    def predict_coreness(self, *inputs):
        """Return the most likely integer coreness for each input sample."""
        return self(*inputs).argmax(dim=1)


def load_hybrid_coreness_model(checkpoint_path, device="cpu"):
    """Load a model checkpoint written by ``train_coreness.py``."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HybridCorenessPredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_coreness_map(model, feature_arrays, batch_size=512, device=None):
    """Predict a ``node -> next coreness`` mapping from prepared arrays."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sample_count = len(feature_arrays["nodes"])
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            inputs = [
                torch.from_numpy(feature_arrays[name][start:end]).to(device)
                for name in (
                    "temporal",
                    "neighbor_structures",
                    "time_deltas",
                    "weights",
                    "mask",
                )
            ]
            predictions.append(model.predict_coreness(*inputs).cpu().numpy())
    if not predictions:
        return {}
    values = np.concatenate(predictions)
    return {
        int(node): int(coreness)
        for node, coreness in zip(feature_arrays["nodes"], values)
    }


def community_from_predicted_coreness(q, k, t, predicted_coreness, snapshots):
    """Return q's predicted connected k-core candidate component.

    ``predicted_coreness`` maps every candidate node to its predicted coreness
    in ``G_(t+1)``. Connectivity is evaluated on the cumulative union graph
    observed through ``G_t``.
    """
    if not isinstance(t, int) or t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    if k <= 0:
        raise ValueError("k must be positive")
    if predicted_coreness.get(q, 0) < k:
        return frozenset()

    eligible = {
        node for node, coreness in predicted_coreness.items() if coreness >= k
    }
    adjacency = {node: set() for node in eligible}
    for snapshot in snapshots[:t + 1]:
        for edge in snapshot["edge_list"]:
            u, v = edge[0], edge[1]
            if u in eligible and v in eligible and u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)

    peel_queue = deque(
        node for node, neighbors in adjacency.items() if len(neighbors) < k
    )
    removed = set(peel_queue)
    while peel_queue:
        node = peel_queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in removed:
                continue
            adjacency[neighbor].discard(node)
            if len(adjacency[neighbor]) < k:
                removed.add(neighbor)
                peel_queue.append(neighbor)
    if q in removed:
        return frozenset()

    visited = {q}
    queue = deque([q])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in removed and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return frozenset(visited)
