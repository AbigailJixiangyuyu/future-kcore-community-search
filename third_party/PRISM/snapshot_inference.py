"""Synchronous next-snapshot adaptation of PRISM's memory, GAT and decoder."""

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import torch

from .modules.decoder import LinkPredictor
from .modules.emb_module import GraphAttentionEmbedding
from .modules.memory_module import DAATGNMemory
from .modules.msg_agg import MeanAggregator
from .modules.msg_func import IdentityMessage


PROTOCOL = "prism_snapshot_v1"


@dataclass(frozen=True)
class Config:
    mem_dim: int = 100
    time_dim: int = 100
    emb_dim: int = 100
    m_pass: int = 3
    neighbors: int = 10

    def validate(self):
        if min(asdict(self).values()) < 1 or self.emb_dim % 2:
            raise ValueError("positive dimensions and an even emb_dim are required")


def edge_array(rows):
    result = np.asarray(rows, dtype=np.int64)
    if not result.size:
        return np.empty((0, 2), dtype=np.int64)
    if result.ndim != 2 or result.shape[1] < 2:
        raise ValueError("expected rows of node ID pairs")
    result = np.sort(result[:, :2], axis=1)
    return np.unique(result[result[:, 0] != result[:, 1]], axis=0)


class SnapshotHistory:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.mapping = {}
        self.adjacency = {}
        self.edges = set()
        self.prepared_t = -1
        self.digest = hashlib.sha256()

    def observe(self, rows, t):
        if t != self.prepared_t + 1:
            raise ValueError("snapshots must be observed chronologically")
        pairs = edge_array(rows)
        self.digest.update(np.asarray((t, len(pairs)), dtype=np.int64).tobytes())
        self.digest.update(pairs.tobytes())
        for u, v in pairs:
            u, v = int(u), int(v)
            for node in (u, v):
                if node not in self.mapping:
                    if len(self.mapping) >= self.capacity:
                        raise ValueError("node capacity exceeded")
                    self.mapping[node] = len(self.mapping) + 1
            a, b = self.mapping[u], self.mapping[v]
            self.adjacency.setdefault(a, []).append((b, t))
            self.adjacency.setdefault(b, []).append((a, t))
            self.edges.add((u, v))
        self.prepared_t = t
        return pairs

    def edge_pairs(self, candidate):
        candidate = set(candidate)
        return np.asarray(sorted(pair for pair in self.edges
                                 if pair[0] in candidate and pair[1] in candidate),
                          dtype=np.int64).reshape(-1, 2)


class PrismSnapshotModel(torch.nn.Module):
    def __init__(self, capacity, config):
        super().__init__()
        config.validate()
        self.config = config
        self.memory = DAATGNMemory(
            capacity + 1, 1, config.mem_dim, config.time_dim,
            message_module=IdentityMessage(1, config.mem_dim, config.time_dim),
            aggregator_module=MeanAggregator(1 + 2 * config.mem_dim + config.time_dim),
            layer=config.m_pass,
        )
        self.gnn = GraphAttentionEmbedding(config.mem_dim, config.emb_dim, 1,
                                           self.memory.time_enc)
        self.link_pred = LinkPredictor(config.emb_dim)
        self.register_buffer("state", torch.zeros(capacity + 1, config.mem_dim))
        self.register_buffer("last_seen", torch.zeros(capacity + 1, dtype=torch.long))

    def reset_history(self):
        self.state = torch.zeros_like(self.state)
        self.last_seen.zero_()
        self.memory.reset_state()

    def observe(self, pairs, history, t):
        """Apply all simultaneous interactions in one synchronous PRISM update."""
        if not len(pairs):
            return
        device = self.state.device
        a = torch.as_tensor([history.mapping[int(u)] for u in pairs[:, 0]], device=device)
        b = torch.as_tensor([history.mapping[int(v)] for v in pairs[:, 1]], device=device)
        src = torch.cat((a, b))
        dst = torch.cat((b, a))
        when = torch.full_like(src, t + 1)
        msg = self.state.new_zeros((len(src), 1))
        enc = self.memory.time_enc((when - self.last_seen[src]).to(self.state.dtype))
        messages = torch.cat((self.memory.msg_s_module(self.state[a], self.state[b],
                          msg[:len(a)], enc[:len(a)]),
                          self.memory.msg_d_module(self.state[b], self.state[a],
                          msg[len(a):], enc[len(a):])), dim=0)
        nodes, inverse = src.unique(sorted=True, return_inverse=True)
        aggregate = self.memory.aggr_module(messages, inverse, when, len(nodes))
        state = self.state[nodes]
        for _ in range(self.config.m_pass):
            state = self.memory.memory_updater(aggregate, state)
        self.state = self.state.index_copy(0, nodes, state)
        self.last_seen[nodes] = t + 1

    def probabilities(self, pairs, history):
        pairs = edge_array_for_scoring(pairs)
        if not len(pairs):
            return self.state.new_empty(0)
        device = self.state.device
        ids = np.asarray([(history.mapping.get(int(u), 0), history.mapping.get(int(v), 0))
                          for u, v in pairs], dtype=np.int64)
        # Duplicate IDs only in query positions; unique nodes get one embedding.
        nodes = sorted(set(ids.flat))
        for node in list(nodes):
            for other, _ in history.adjacency.get(node, [])[-self.config.neighbors:]:
                if other not in nodes:
                    nodes.append(other)
        position = {node: i for i, node in enumerate(nodes)}
        idx, ts = [], []
        for node in nodes:
            for other, stamp in history.adjacency.get(node, [])[-self.config.neighbors:]:
                if other in position:
                    idx.append((position[other], position[node]))
                    ts.append(stamp + 1)
        n = torch.as_tensor(nodes, dtype=torch.long, device=device)
        if idx:
            edge_index = torch.as_tensor(idx, dtype=torch.long, device=device).T.contiguous()
            stamps = torch.as_tensor(ts, dtype=torch.long, device=device)
            raw = self.state.new_zeros((len(ts), 1))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            stamps = torch.empty(0, dtype=torch.long, device=device)
            raw = self.state.new_zeros((0, 1))
        z = self.gnn(self.state[n], self.last_seen[n], edge_index, stamps, raw)
        left = torch.as_tensor([position[u] for u in ids[:, 0]], device=device)
        right = torch.as_tensor([position[v] for v in ids[:, 1]], device=device)
        return (self.link_pred(z[left], z[right]).flatten() +
                self.link_pred(z[right], z[left]).flatten()) / 2


def edge_array_for_scoring(rows):
    result = np.asarray(rows, dtype=np.int64)
    if result.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("candidate edges must have shape (N,2)")
    return result


class PrismSnapshotPredictor:
    def __init__(self, snapshots, checkpoint, device="cpu"):
        payload = torch.load(str(checkpoint), map_location="cpu")
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("unsupported PRISM checkpoint protocol")
        self.snapshots = snapshots
        self.fit_end_t = int(payload["fit_end_t"])
        self.config = Config(**payload["config"])
        self.history = SnapshotHistory(payload["capacity"])
        self.model = PrismSnapshotModel(payload["capacity"], self.config).to(device)
        self.model.load_state_dict(payload["model"])
        # The checkpoint was chosen after validation; its runtime buffers are
        # never a starting point for replaying the snapshot prefix.
        self.model.reset_history()
        self.model.eval()
        self.prefix_digest = payload["prefix_digest"]

    def prepare_time(self, t):
        if not self.fit_end_t <= t < len(self.snapshots) - 1:
            raise ValueError("t must be at least fitted time and before final snapshot")
        if t < self.history.prepared_t:
            raise ValueError("cannot rewind prepared history")
        with torch.no_grad():
            for i in range(self.history.prepared_t + 1, t + 1):
                pairs = self.history.observe(self.snapshots[i], i)
                if i == self.fit_end_t and self.history.digest.hexdigest() != self.prefix_digest:
                    raise ValueError("checkpoint training prefix does not match snapshots")
                self.model.observe(pairs, self.history, i)

    def score_edges(self, edges, target_t=None, batch_size=128):
        if target_t is not None and target_t != self.history.prepared_t + 1:
            raise ValueError("target must be the next snapshot")
        if self.history.prepared_t < self.fit_end_t:
            raise ValueError("call prepare_time first")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        edges = edge_array_for_scoring(edges)
        with torch.no_grad():
            return np.concatenate([
                self.model.probabilities(edges[i:i + batch_size], self.history)
                .cpu().numpy().astype(np.float32)
                for i in range(0, len(edges), batch_size)
            ]) if len(edges) else np.empty(0, dtype=np.float32)
