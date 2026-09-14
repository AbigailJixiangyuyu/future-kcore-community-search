"""Trainable hybrid TCS + T-PPR structural coreness predictor."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cumulative_ordinal_targets(labels, kmax):
    """Encode integer coreness labels as ``coreness >= k`` targets."""
    if labels.dim() != 1:
        raise ValueError("labels must be a 1D tensor")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    if torch.any(labels < 0) or torch.any(labels > kmax):
        raise ValueError("labels must be in [0, kmax]")
    thresholds = torch.arange(1, kmax + 1, device=labels.device)
    return (labels.unsqueeze(1) >= thresholds.unsqueeze(0)).to(torch.float32)


def cumulative_logits_from_class_logits(class_logits):
    """Return stable logits for ``P(coreness >= k)`` from class logits."""
    if class_logits.dim() != 2 or class_logits.size(1) < 2:
        raise ValueError("class_logits must have shape [batch, class_count >= 2]")
    return torch.stack(
        [
            torch.logsumexp(class_logits[:, threshold:], dim=1)
            - torch.logsumexp(class_logits[:, :threshold], dim=1)
            for threshold in range(1, class_logits.size(1))
        ],
        dim=1,
    )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.transform = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.transform(self.norm(values))


class MaskedSetAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention for padded temporal-node sets."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by heads")
        self.heads = int(heads)
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = ResidualMLPBlock(width, dropout)

    def forward(self, values, mask):
        batch, length, width = values.shape
        qkv = self.qkv(values).view(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = [
            part.permute(0, 2, 1, 3) for part in qkv.unbind(dim=2)
        ]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.head_width ** 0.5
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * mask[:, None, None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, width)
        result = self.norm1(values + self.dropout(self.output(attended)))
        result = self.ffn(result)
        return result * mask.unsqueeze(-1).to(result.dtype)


class QuerySetAttention(nn.Module):
    """Pool a temporal-node set using the query and T-PPR weights."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by heads")
        self.heads = int(heads)
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.prior_strength_raw = nn.Parameter(
            torch.tensor(0.541324854612918, dtype=torch.float32)
        )

    def forward(self, query, values, weights, mask):
        batch, length, width = values.shape
        projected_query = self.query(query).view(
            batch, self.heads, self.head_width
        )
        projected_key = self.key(values).view(
            batch, length, self.heads, self.head_width
        ).permute(0, 2, 1, 3)
        projected_value = self.value(values).view(
            batch, length, self.heads, self.head_width
        ).permute(0, 2, 1, 3)
        scores = torch.sum(projected_query.unsqueeze(2) * projected_key, dim=-1)
        scores = scores / self.head_width ** 0.5
        prior_strength = F.softplus(self.prior_strength_raw)
        scores = scores + prior_strength * torch.log(
            weights.clamp_min(1e-12)
        ).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * mask.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(
            self.dropout(attention).unsqueeze(-1) * projected_value, dim=2
        )
        return self.output(pooled.reshape(batch, width))


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
        core_dim=32,
        core_lookback=5,
        bucket_dim=16,
        attention_heads=4,
        persistence_scale=2.0,
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
        if core_dim <= 0 or core_lookback <= 0 or bucket_dim <= 0:
            raise ValueError("embedding dimensions and lookback must be positive")
        if structure_hidden % attention_heads:
            raise ValueError("structure_hidden must be divisible by attention_heads")
        if persistence_scale <= 0:
            raise ValueError("persistence_scale must be positive")

        self.kmax = int(kmax)
        self.hmax = int(hmax)
        self.order = int(order)
        self.core_dim = int(core_dim)
        self.core_lookback = int(core_lookback)
        self.structure_width = order * (hmax + 1)
        self.coreness_table = nn.Parameter(
            torch.empty(kmax + 1, core_dim).normal_(mean=0.0, std=0.02)
        )
        self.h_index_bucket_table = nn.Parameter(
            torch.empty(hmax + 1, bucket_dim).normal_(mean=0.0, std=0.02)
        )
        self.coreness_bucket_table = nn.Parameter(
            torch.empty(hmax + 1, bucket_dim).normal_(mean=0.0, std=0.02)
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
        )
        self.structure_token_encoder = nn.Sequential(
            nn.Linear(order * bucket_dim + time_dim, structure_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.Sequential(
            nn.LayerNorm(kmax),
            nn.Linear(kmax, temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(core_dim + 2 * temporal_hidden, structure_hidden),
            nn.GELU(),
        )
        self.query_blocks = nn.ModuleList(
            [ResidualMLPBlock(structure_hidden, dropout) for _ in range(2)]
        )
        self.set_attention = MaskedSetAttentionBlock(
            structure_hidden, attention_heads, dropout
        )
        self.query_attention = QuerySetAttention(
            structure_hidden, attention_heads, dropout
        )
        self.structure_norm = nn.LayerNorm(structure_hidden)
        self.fusion_projection = nn.Sequential(
            nn.Linear(structure_hidden * 2, fusion_hidden),
            nn.GELU(),
        )
        self.fusion_blocks = nn.ModuleList(
            [ResidualMLPBlock(fusion_hidden, dropout) for _ in range(2)]
        )
        self.core_projection = nn.Linear(fusion_hidden, core_dim)
        nn.init.zeros_(self.core_projection.weight)
        nn.init.zeros_(self.core_projection.bias)

        levels = torch.arange(kmax + 1, dtype=torch.float32)
        persistence_logits = -float(persistence_scale) * torch.abs(
            levels[:, None] - levels[None, :]
        )
        self.register_buffer("persistence_logits", persistence_logits)
        # Initialize history after all shared modules, without
        # changing the RNG state used by shuffling and subsequent training.
        with torch.random.fork_rng(devices=[]):
            self.history_encoder = nn.GRU(
                core_dim, temporal_hidden, batch_first=True
            )
        # Preserve established initialization and RNG state; copy values only,
        # never share Parameter/storage with the input embedding table.
        with torch.random.fork_rng(devices=[]):
            self.output_head = nn.Linear(core_dim, kmax + 1)
        with torch.no_grad():
            self.output_head.weight.copy_(self.coreness_table)
            self.output_head.bias.zero_()

    def pool_structure(self, query, tokens, weights, mask):
        """Pool with set attention and query-conditioned T-PPR attention."""
        set_tokens = self.set_attention(tokens, mask)
        learned_pool = self.query_attention(query, set_tokens, weights, mask)
        return self.structure_norm(learned_pool)

    def decode_core_state(self, core_state):
        """Produce learned class logits; the persistence prior is added later."""
        return self.output_head(core_state)

    def forward(
        self,
        temporal,
        core_history,
        neighbor_structures,
        time_deltas,
        weights,
        mask,
    ):
        """Return logits for coreness classes ``0..kmax``."""
        if temporal.dim() != 2 or temporal.size(1) != self.kmax:
            raise ValueError("temporal input has an invalid shape")
        if core_history.dim() != 2 or core_history.size(1) != self.core_lookback:
            raise ValueError("core_history has an invalid shape")
        if torch.any(core_history < 0) or torch.any(core_history > self.kmax):
            raise ValueError("core_history contains an invalid token")
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
        histograms = neighbor_structures.view(
            *neighbor_structures.shape[:2], self.order, self.hmax + 1
        )
        structure_blocks = []
        for level in range(self.order):
            table = (
                self.coreness_bucket_table
                if level == self.order - 1
                else self.h_index_bucket_table
            )
            embedded = torch.matmul(histograms[:, :, level], table)
            structure_blocks.append(embedded)
        tokens = self.structure_token_encoder(
            torch.cat((*structure_blocks, encoded_time), dim=-1)
        )
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)

        history_tokens = self.coreness_table[core_history]
        history_tokens = torch.flip(history_tokens, dims=(1,))
        _history_output, history_state = self.history_encoder(history_tokens)
        history_state = history_state[-1]
        temporal_state = self.temporal_encoder(temporal)
        current_token = core_history[:, 0]
        current_embedding = self.coreness_table[current_token]
        query = self.query_projection(
            torch.cat((current_embedding, history_state, temporal_state), dim=1)
        )
        for block in self.query_blocks:
            query = block(query)

        structural = self.pool_structure(query, tokens, weights, mask)
        fused = self.fusion_projection(
            torch.cat((query, structural), dim=1)
        )
        for block in self.fusion_blocks:
            fused = block(fused)

        core_state = self.core_projection(fused)
        learned_logits = self.decode_core_state(core_state)
        return self.persistence_logits[current_token] + learned_logits

    @staticmethod
    def ordinal_logits(class_logits):
        return cumulative_logits_from_class_logits(class_logits)

    def predict_coreness(self, *inputs):
        """Decode coreness as the number of passed ordinal thresholds."""
        return (self.ordinal_logits(self(*inputs)) > 0.0).sum(dim=1)


def load_hybrid_coreness_model(checkpoint_path, device="cpu"):
    """Load a model checkpoint written by ``train_coreness.py``."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    objective = checkpoint.get("objective", {})
    if objective.get("name") not in (
        "token_tied_cumulative_ordinal_bce", "hybrid_cumulative_ordinal_bce",
    ):
        raise ValueError(
            "checkpoint does not use the current hybrid ordinal model; retrain it"
        )
    model_config = dict(checkpoint["model_config"])
    checkpoint_fusion = model_config.pop("fusion_type", None)
    if model_config.pop("fusion_ffn_type", "mlp") != "mlp" or any(
        name.startswith("fusion_blocks.") and ".ffn." in name
        for name in checkpoint["state_dict"]
    ):
        raise ValueError(
            "SwiGLU fusion blocks have been retired; use an MLP checkpoint or retrain."
        )
    checkpoint_pooling = model_config.pop("structure_pooling", "both")
    if model_config.pop("history_encoder_type", "gru") != "gru" or any(
        name.startswith(("history_encoder.blocks.", "history_encoder.input_projection.",
                         "history_encoder.norm."))
        for name in checkpoint["state_dict"]
    ):
        raise ValueError(
            "Transformer history encoders have been retired; use a GRU checkpoint "
            "or retrain."
        )
    model_config.pop("history_attention_heads", None)
    model_config.pop("history_transformer_layers", None)
    if model_config.pop("history_structure", False) or any(
        name.startswith("history_input_projection.")
        for name in checkpoint["state_dict"]
    ):
        raise ValueError(
            "Structure-history inputs have been retired; use a coreness-only "
            "checkpoint or retrain."
        )
    # Checkpoints predating independent output heads always used tied weights.
    checkpoint_head = model_config.pop("output_head_type", "tied")
    legacy_lag = model_config.pop("use_lag", False)
    if legacy_lag or "lag_table" in checkpoint["state_dict"]:
        raise ValueError(
            "Lag embeddings have been retired; select a no-Lag checkpoint "
            "or retrain. Lag weights cannot be silently discarded."
        )
    if checkpoint["state_dict"]["coreness_table"].shape[0] == model_config["kmax"] + 2:
        raise ValueError(
            "This checkpoint uses a separate ABSENT embedding; missing history "
            "now uses coreness zero. Retrain the model with the current code."
        )
    if checkpoint_head != "linear" or "output_bias" in checkpoint["state_dict"]:
        raise ValueError(
            "Tied output heads have been retired; use an explicitly marked "
            "linear checkpoint or retrain. Missing output-head metadata denotes "
            "a legacy tied checkpoint."
        )
    model = HybridCorenessPredictor(**model_config)
    state_dict = dict(checkpoint["state_dict"])
    if checkpoint_fusion not in (None, "concat") or (
        state_dict["fusion_projection.0.weight"].shape[1]
        != model.fusion_projection[0].in_features
    ):
        raise ValueError(
            "Four-term interaction fusion has been retired; use a concat "
            "[Q,S] checkpoint or retrain."
        )
    if state_dict["structure_token_encoder.0.weight"].shape[1] == (
        model.structure_token_encoder[0].in_features + 1
    ):
        raise ValueError(
            "This checkpoint uses T-PPR weight as a structure token input; "
            "retrain with the current aggregation-only weight design."
        )
    if checkpoint_pooling != "b":
        raise ValueError(
            "Path A and dual-path pooling have been retired; use a path B "
            "checkpoint or retrain. Missing pooling metadata denotes a legacy "
            "dual-path checkpoint."
        )
    legacy_order = state_dict.pop("structure_order_table", None)
    if legacy_order is not None:
        # Fixed group offsets precede a biased linear layer, so fold them
        # into its bias without changing the checkpoint's inference function.
        expected_shape = (model.order, model.h_index_bucket_table.shape[1])
        if tuple(legacy_order.shape) != expected_shape:
            raise ValueError("checkpoint structure_order_table has an invalid shape")
        weight = state_dict["structure_token_encoder.0.weight"]
        state_dict["structure_token_encoder.0.bias"] = (
            state_dict["structure_token_encoder.0.bias"]
            + weight[:, :legacy_order.numel()] @ legacy_order.reshape(-1)
        )
    model.load_state_dict(state_dict)
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
                    "core_history",
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
                    feature_table["core_history"][batch_rows]
                ).to(device),
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
        nodes = set(nodes)
        nodes.difference_update(predicted)
        nodes = sorted(nodes)
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
