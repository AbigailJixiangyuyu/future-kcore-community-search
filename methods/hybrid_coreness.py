"""Trainable hybrid TCS + T-PPR structural coreness predictor."""

from collections import deque
from dataclasses import dataclass, replace

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


def predict_coreness_indexed_map(
    model,
    feature_table,
    structure_table,
    nodes,
    batch_size=512,
    device=None,
):
    """Predict requested nodes by gathering precomputed time-slice rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    requested = np.asarray(sorted(set(nodes)), dtype=np.int64)
    if not len(requested):
        return {}

    node_rows = feature_table.get("node_rows")
    if node_rows is None:
        table_nodes = feature_table["nodes"]
        rows = np.searchsorted(table_nodes, requested)
        if (
            np.any(rows >= len(table_nodes))
            or not np.array_equal(table_nodes[rows], requested)
        ):
            raise KeyError("requested nodes are missing from the feature table")
    else:
        try:
            rows = np.fromiter(
                (node_rows[int(node)] for node in requested),
                dtype=np.int64,
                count=len(requested),
            )
        except KeyError as error:
            raise KeyError(
                "requested nodes are missing from the feature table"
            ) from error
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(requested), batch_size):
            end = min(start + batch_size, len(requested))
            batch_rows = rows[start:end]
            structure_indices = feature_table["structure_indices"][batch_rows]
            structure_values = getattr(
                structure_table, "values", structure_table
            )
            inputs = [
                torch.from_numpy(feature_table["temporal"][batch_rows]).to(device),
                torch.from_numpy(
                    np.asarray(structure_values[structure_indices])
                ).to(device),
                torch.from_numpy(
                    feature_table["time_deltas"][batch_rows]
                ).to(device),
                torch.from_numpy(feature_table["weights"][batch_rows]).to(device),
                torch.from_numpy(feature_table["mask"][batch_rows]).to(device),
            ]
            predictions.append(model.predict_coreness(*inputs).cpu().numpy())

    values = np.concatenate(predictions)
    return {
        int(node): int(coreness)
        for node, coreness in zip(requested, values)
    }


@dataclass(frozen=True)
class LayeredCommunityResult:
    """Result and work counters for one threshold-driven BFS query."""

    community: frozenset
    predicted_coreness: dict
    bfs_layers: int
    newly_predicted_node_count: int

    @property
    def predicted_node_count(self):
        return len(self.predicted_coreness)

    @property
    def accepted_node_count(self):
        return len(self.community)

    @property
    def reused_prediction_count(self):
        return self.predicted_node_count - self.newly_predicted_node_count

    def with_new_prediction_count(self, count):
        if count < 0 or count > self.predicted_node_count:
            raise ValueError("new prediction count is outside the query range")
        return replace(self, newly_predicted_node_count=count)


def layered_threshold_bfs(q, k, adjacency, predict_batch):
    """Expand from q while batched next-coreness predictions stay >= k."""
    if k <= 0:
        raise ValueError("k must be positive")

    predicted = {}

    def predict(nodes):
        nodes = sorted(set(nodes) - set(predicted))
        if not nodes:
            return {}
        values = predict_batch(nodes)
        missing = set(nodes) - set(values)
        extra = set(values) - set(nodes)
        if missing or extra:
            raise ValueError(
                "predict_batch must return exactly the requested nodes"
            )
        normalized = {int(node): int(values[node]) for node in nodes}
        predicted.update(normalized)
        return normalized

    query_prediction = predict([q])
    if query_prediction[q] < k:
        return LayeredCommunityResult(frozenset(), predicted, 1, len(predicted))

    community = {q}
    frontier = {q}
    layers = 1
    while frontier:
        candidates = set()
        for node in frontier:
            candidates.update(adjacency.get(node, ()))
        candidates.difference_update(predicted)
        if not candidates:
            break
        layer_predictions = predict(candidates)
        layers += 1
        frontier = {
            node for node, coreness in layer_predictions.items()
            if coreness >= k
        }
        community.update(frontier)

    return LayeredCommunityResult(
        frozenset(community), predicted, layers, len(predicted)
    )


def community_from_predicted_coreness(q, k, t, predicted_coreness, snapshots):
    """Return q's threshold-connected component in the cumulative graph.

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

    visited = {q}
    queue = deque([q])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return frozenset(visited)
