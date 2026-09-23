"""TFWaveFormer on frozen, undirected snapshots without raw attributes."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral

import numpy as np
import torch
from torch import nn

if __package__:
    from .models.TFWaveFormer import TFWaveFormer
    from .models.modules import MergeLayer
else:
    from models.TFWaveFormer import TFWaveFormer
    from models.modules import MergeLayer


PROTOCOL_VERSION = "tfwaveformer_snapshot_v1"


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def edge_array(edges, canonical=False):
    array = np.asarray(edges)
    if array.shape == (0,):
        array = np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in "iu":
        raise ValueError("edges must be an integer (N, 2) array")
    if array.size and array.dtype.kind == "u" and int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError("node IDs must fit int64")
    array = array.astype(np.int64, copy=True)
    if canonical:
        array.sort(axis=1)
        array = np.unique(array[array[:, 0] != array[:, 1]], axis=0)
    return array


@dataclass(frozen=True)
class SnapshotConfig:
    feature_dim: int = 172
    time_feat_dim: int = 100
    channel_embedding_dim: int = 50
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    max_input_sequence_length: int = 32

    def __post_init__(self):
        for name in ("feature_dim", "time_feat_dim", "channel_embedding_dim",
                     "num_layers", "num_heads", "max_input_sequence_length"):
            integer(getattr(self, name), name, 1)
        if self.feature_dim < 4 or self.feature_dim % 2 or self.feature_dim % self.num_heads:
            raise ValueError("feature_dim must be even, >= 4 and divisible by num_heads")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class SnapshotHistory:
    """Keep only the last L incident edges of each observed node."""

    sample_neighbor_strategy = "recent"

    def __init__(self, length):
        self.length = integer(length, "length", 1)
        self.mapping = {}
        self.rows = [deque(maxlen=length)]
        self.prepared_t = -1
        self.digest = hashlib.sha256()

    def observe(self, edges, t):
        if integer(t, "t") != self.prepared_t + 1:
            raise ValueError("history must advance by one snapshot")
        edges = edge_array(edges, canonical=True)
        for node in sorted(set(edges.ravel().tolist()) - self.mapping.keys()):
            self.mapping[node] = len(self.rows)
            self.rows.append(deque(maxlen=self.length))
        for u, v in edges:
            a, b = self.mapping[int(u)], self.mapping[int(v)]
            self.rows[a].append((b, t))
            self.rows[b].append((a, t))
        self.digest.update(np.asarray([t, len(edges)], dtype="<i8").tobytes())
        self.digest.update(edges.astype("<i8", copy=False).tobytes())
        self.prepared_t = t

    def map_queries(self, edges):
        # Unknown query IDs remain distinct and never enter observed history.
        unknown = {node: -i - 1 for i, node in enumerate(
            sorted(set(edges.ravel().tolist()) - self.mapping.keys())
        )}
        return np.asarray([
            (self.mapping[u] if u in self.mapping else unknown[u],
             self.mapping[v] if v in self.mapping else unknown[v])
            for u, v in edges
        ], dtype=np.int64).reshape(-1, 2)

    def get_historical_neighbors(self, node_ids, node_interact_times, num_neighbors):
        if num_neighbors != self.length:
            raise ValueError("history length does not match model")
        if not np.all(node_interact_times == self.prepared_t + 1):
            raise ValueError("only the immediate next snapshot can be scored")
        shape = (len(node_ids), self.length)
        neighbors = np.zeros(shape, dtype=np.int64)
        times = np.zeros(shape, dtype=np.float32)
        for index, node in enumerate(node_ids):
            row = self.rows[int(node)] if node > 0 else ()
            if row:
                values = np.asarray(row, dtype=np.int64)
                neighbors[index, -len(row):] = values[:, 0]
                times[index, -len(row):] = values[:, 1]
        return neighbors, np.zeros(shape, dtype=np.int64), times


class _ZeroAttributeTFWaveFormer(TFWaveFormer):
    def get_features(self, node_interact_times, nodes_neighbor_ids, nodes_edge_ids,
                     nodes_neighbor_times, time_encoder):
        # All real and padding attributes are zero; do not allocate O(events * D).
        shape = (*nodes_neighbor_ids.shape, self.edge_feat_dim)
        zeros = torch.zeros(shape, dtype=torch.float32, device=self.device)
        deltas = torch.from_numpy(node_interact_times[:, None] - nodes_neighbor_times).to(self.device)
        temporal = time_encoder(deltas)
        mask = torch.from_numpy(nodes_neighbor_ids == 0).to(self.device)
        temporal = temporal.masked_fill(mask.unsqueeze(-1), 0)
        return zeros, zeros, temporal


def build_model(config, history, device):
    features = np.zeros((1, config.feature_dim), dtype=np.float32)
    backbone = _ZeroAttributeTFWaveFormer(
        features, features, history, time_feat_dim=config.time_feat_dim,
        channel_embedding_dim=config.channel_embedding_dim, num_layers=config.num_layers,
        num_heads=config.num_heads, dropout=config.dropout,
        max_input_sequence_length=config.max_input_sequence_length, device=str(device),
    )
    decoder = MergeLayer(config.feature_dim, config.feature_dim, config.feature_dim, 1)
    return nn.Sequential(backbone, decoder).to(device).float()


def edge_logits(model, history, edges):
    mapped = history.map_queries(edge_array(edges))
    times = np.full(len(mapped), history.prepared_t + 1, dtype=np.float32)
    left, right = model[0].compute_src_dst_node_temporal_embeddings(
        mapped[:, 0], mapped[:, 1], times,
    )
    return (model[1](left, right) + model[1](right, left)).squeeze(-1) * 0.5


class TFWaveFormerSnapshotPredictor:
    """Load a trusted snapshot checkpoint and lazily replay only observed history."""

    def __init__(self, snapshots, *, checkpoint, device="cpu"):
        if not len(snapshots):
            raise ValueError("snapshots cannot be empty")
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("not a TFWaveFormer snapshot checkpoint")
        self.config = SnapshotConfig(**payload["config"])
        self.fit_end_t = integer(payload["fit_end_t"], "fit_end_t")
        self.fit_history_sha256 = payload["fit_history_sha256"]
        self.training_metadata = payload.get("training")
        self.snapshots = snapshots
        self.history = SnapshotHistory(self.config.max_input_sequence_length)
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            self.model = build_model(self.config, self.history, "cpu")
        self.model.load_state_dict(payload["state_dict"], strict=True)
        if not all(torch.isfinite(value).all().item() for value in self.model.state_dict().values()):
            raise ValueError("checkpoint contains non-finite weights")
        self.model.to(self.device)
        self.model[0].device = str(self.device)
        self.model[0].nif_encoder.device = str(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._fit_verified = False

    @property
    def prepared_t(self):
        return self.history.prepared_t

    @property
    def metadata(self):
        return dict(protocol=PROTOCOL_VERSION, config=asdict(self.config),
                    prepared_t=self.prepared_t, fit_end_t=self.fit_end_t,
                    history_sha256=self.history.digest.hexdigest(),
                    observed_nodes=len(self.history.mapping), attributes="zero_float32",
                    time_unit="snapshot_index", undirected="mean_logits")

    def prepare_time(self, t):
        t = integer(t, "t")
        if t >= len(self.snapshots):
            raise IndexError("t outside snapshot sequence")
        if t < self.fit_end_t:
            raise ValueError("t precedes fit_end_t; fitted weights would leak future data")
        if t < self.prepared_t:
            raise ValueError("cannot move history backwards")
        if self.prepared_t >= self.fit_end_t and not self._fit_verified:
            raise ValueError("fitting history did not match checkpoint")
        for step in range(self.prepared_t + 1, t + 1):
            self.history.observe(self.snapshots[step], step)
            if step == self.fit_end_t:
                if self.history.digest.hexdigest() != self.fit_history_sha256:
                    raise ValueError("fitting history does not match checkpoint")
                self._fit_verified = True
        return self.metadata

    @torch.no_grad()
    def score_edges(self, edges, target_t=None, *, batch_size=128):
        if not self._fit_verified:
            raise RuntimeError("call prepare_time(t) before scoring")
        target = self.prepared_t + 1 if target_t is None else integer(target_t, "target_t")
        if target != self.prepared_t + 1:
            raise ValueError("target_t must equal prepared_t + 1")
        batch_size = integer(batch_size, "batch_size", 1)
        edges = edge_array(edges)
        result = np.empty(len(edges), dtype=np.float32)
        for start in range(0, len(edges), batch_size):
            scores = edge_logits(self.model, self.history, edges[start:start + batch_size]).sigmoid()
            result[start:start + len(scores)] = scores.cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite edge scores")
        return result
