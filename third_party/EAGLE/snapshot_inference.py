"""Frozen-history, next-snapshot inference for EAGLE link prediction."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Optional

import numba as nb
from numba import types
import numpy as np
import torch

from link_prediction.utils.model import Mixer_per_node
from link_prediction.utils.util import tppr_node_finder


PROTOCOL_VERSION = "eagle_snapshot_v1"


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return int(value)


def _edge_array(edges):
    array = np.asarray(edges)
    if array.shape == (0,):
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edges must have shape (N, 2)")
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError("node IDs must be integers")
    if array.dtype.kind == "u" and int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError("node IDs must fit int64")
    return array.astype(np.int64, copy=True)


@dataclass(frozen=True)
class SnapshotInferenceConfig:
    branch: str = "structure"
    structure_topk: int = 100
    alpha: float = 0.9
    beta: float = 0.8
    time_topk: int = 15
    time_channels: int = 100
    num_layers: int = 1
    yita: Optional[float] = None
    # Latest snapshot used for ANY fitting, including checkpoint selection.
    fit_end_t: Optional[int] = None

    def __post_init__(self):
        if self.branch not in ("structure", "time", "hybrid"):
            raise ValueError("branch must be structure, time, or hybrid")
        for name in ("structure_topk", "time_channels", "num_layers"):
            _integer(getattr(self, name), name, 1)
        _integer(self.time_topk, "time_topk", 2)
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not np.isfinite(self.beta) or not 0 < self.beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        if self.yita is not None:
            if not np.isfinite(self.yita) or not 0 <= self.yita <= np.finfo(np.float32).max:
                raise ValueError("yita must be finite, nonnegative and fit float32")
        if self.branch == "hybrid" and self.yita is None:
            raise ValueError("hybrid requires an explicitly calibrated yita")
        if self.fit_end_t is not None:
            _integer(self.fit_end_t, "fit_end_t")
        if self.branch != "structure" and self.fit_end_t is None:
            raise ValueError("time/hybrid require an explicit fit_end_t")


@nb.njit
def _grow_structure(finder, count):
    old_count = finder.num_nodes
    if count <= old_count:
        return
    norms = np.zeros((finder.n_tppr, count), dtype=np.float64)
    norms[:, :old_count] = finder.norm_list
    for index in range(finder.n_tppr):
        for _ in range(count - old_count):
            finder.PPR_list[index].append(
                nb.typed.Dict.empty(key_type=types.int64, value_type=types.float64)
            )
    finder.norm_list = norms
    finder.num_nodes = count


@nb.njit
def _structure_scores(finder, edges):
    scores = np.zeros(len(edges), dtype=np.float32)
    for index in range(len(edges)):
        source, target = edges[index]
        if source >= 0 and target >= 0:
            scores[index] = finder.get_similarity(0, source, target)
    return scores


class EagleSnapshotPredictor:
    """Consume a lazy sequence of undirected edge arrays, using zero-based t.

    Only prepare_time reads snapshots. score_edges reads frozen history and
    accepts arbitrary integer node IDs, including IDs not yet observed.
    Model weights must be trained with the same snapshot time units and
    latest-interactions ("last") sampling. This class does not train or select
    a threshold. Structure/Hybrid scores are not probabilities.
    """

    def __init__(self, snapshots, *, config=None, checkpoint=None, device="cpu"):
        if len(snapshots) == 0:
            raise ValueError("snapshots must contain at least one snapshot")
        self.snapshots = snapshots
        self.config = config or SnapshotInferenceConfig()
        if not isinstance(self.config, SnapshotInferenceConfig):
            raise TypeError("config must be SnapshotInferenceConfig")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use device='cpu'")

        self._model = None
        self.checkpoint_path = None
        self.checkpoint_hash = None
        if self.config.branch == "structure":
            if checkpoint is not None:
                raise ValueError("structure does not use a checkpoint")
        else:
            if checkpoint is None:
                raise ValueError("time/hybrid require a Time model checkpoint")
            self.checkpoint_path = str(Path(checkpoint).resolve())
            digest = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.checkpoint_hash = digest.hexdigest()
            mixer_config = dict(
                per_graph_size=self.config.time_topk,
                time_channels=self.config.time_channels,
                num_layers=self.config.num_layers,
                use_single_layer=False,
                device="cpu",
            )
            # Construction should not change the caller's random stream.
            with torch.random.fork_rng(devices=[]):
                self._model = Mixer_per_node(
                    mixer_config, {"dim": self.config.time_channels}
                ).to(self.device).float()
            self._model.base_model.device = self.device
            state = torch.load(self.checkpoint_path, map_location="cpu")
            self._model.load_state_dict(state, strict=True)
            if not all(torch.isfinite(value).all().item()
                       for value in self._model.state_dict().values()):
                raise ValueError("checkpoint contains non-finite parameters")
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)

        self._structure = None
        if self.config.branch != "time":
            # Preserve the original external EAGLE TPPR numeric types.
            self._structure = tppr_node_finder(
                0, self.config.structure_topk, float(self.config.alpha),
                float(self.config.beta), "mul_wo_norm",
            )
        self._node_mapping = {}
        self._histories = []
        self._prepared_t = None
        self._observed_edges = 0
        self._prefix_hash = hashlib.sha256()
        self._recency = np.empty(0, dtype=np.float32)
        self._cold_recency = np.float32(1)
        self._padding_delta = np.float32(1)
        self._mean_delta = np.float32(1)

    @property
    def prepared_t(self):
        return self._prepared_t

    @property
    def node_mapping(self):
        """A copy of the observed original-ID -> dense-ID mapping."""
        return dict(self._node_mapping)

    @property
    def metadata(self):
        return {
            "protocol": PROTOCOL_VERSION,
            "config": asdict(self.config),
            "prepared_t": self.prepared_t,
            "observed_nodes": len(self._node_mapping),
            "observed_snapshot_edges": self._observed_edges,
            "history_sha256": self._prefix_hash.hexdigest(),
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_hash,
            "history_policy": "deduplicated_undirected_edges_per_snapshot",
            "time_unit": "snapshot_index",
            "time_sampling": "last",
            "hybrid_normalization": "observed_nodes_at_frozen_boundary",
            "structure_runtime_dtype": "float64",
            "neural_and_score_dtype": "float32",
            "padding_delta": float(self._padding_delta),
            "mean_delta": float(self._mean_delta),
        }

    def prepare_time(self, t):
        """Advance through t once, never reading any snapshot beyond t."""
        t = _integer(t, "t")
        if t >= len(self.snapshots):
            raise IndexError("t is outside the snapshot sequence")
        if self.config.fit_end_t is not None and t < self.config.fit_end_t:
            raise ValueError("t precedes fit_end_t; fitted parameters would leak future data")
        if self.prepared_t is not None:
            if t < self.prepared_t:
                raise ValueError("cannot move history backwards; construct a new predictor")
            if t == self.prepared_t:
                return self.metadata
        start = 0 if self.prepared_t is None else self.prepared_t + 1
        try:
            for snapshot_t in range(start, t + 1):
                edges = _edge_array(self.snapshots[snapshot_t])
                edges.sort(axis=1)
                edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
                self._observe(edges, snapshot_t)
                self._prefix_hash.update(np.asarray([snapshot_t, len(edges)], dtype="<i8").tobytes())
                self._prefix_hash.update(edges.astype("<i8", copy=False).tobytes())
                self._observed_edges += len(edges)
                self._prepared_t = snapshot_t
        finally:
            # A later lazy read may fail after earlier snapshots were committed.
            if self._model is not None and self.prepared_t is not None:
                self._prepare_recency()
        return self.metadata

    def _observe(self, edges, t):
        new_nodes = sorted(set(edges.ravel().tolist()) - self._node_mapping.keys())
        for node in new_nodes:
            self._node_mapping[node] = len(self._node_mapping)
            if self._model is not None:
                self._histories.append(deque(maxlen=self.config.time_topk))
        if self._structure is not None:
            _grow_structure(self._structure, len(self._node_mapping))
        if not len(edges):
            return
        mapped = np.asarray(
            [(self._node_mapping[int(u)], self._node_mapping[int(v)]) for u, v in edges],
            dtype=np.int64,
        )
        if self._structure is not None:
            # This mutating original API is used ONLY for observed history.
            # With zero negatives its incidental pre-update scores are discarded.
            self._structure.precompute_link_prediction(
                np.concatenate((mapped[:, 0], mapped[:, 1])), 0
            )
        if self._model is not None:
            for source, target in mapped:
                self._histories[source].append(t)
                self._histories[target].append(t)

    def _prepare_recency(self):
        target = self.prepared_t + 1
        topk = self.config.time_topk
        padding = max((target - history[0] for history in self._histories), default=1)
        self._padding_delta = np.float32(padding)
        averages = np.asarray([
            (sum(target - time for time in history) + (topk - len(history)) * padding) / topk
            for history in self._histories
        ], dtype=np.float32)
        self._mean_delta = np.float32(averages.mean()) if len(averages) else np.float32(1)
        self._recency = np.exp(
            np.float32(1) - averages / self._mean_delta
        ).astype(np.float32)
        self._cold_recency = np.float32(
            np.exp(np.float32(1) - self._padding_delta / self._mean_delta)
        )

    def _encode_nodes(self, node_ids, target_t):
        deltas, indices = [], []
        for row, node in enumerate(node_ids):
            history = self._histories[node] if node >= 0 else ()
            values = [target_t - time for time in reversed(history)]
            deltas.extend(values)
            indices.extend(range(row * self.config.time_topk,
                                 row * self.config.time_topk + len(values)))
        times = torch.tensor(deltas, dtype=torch.float32, device=self.device).reshape(-1, 1)
        positions = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self._model.base_model(times, positions, len(node_ids))

    def _time_scores(self, mapped, target_t, undirected):
        nodes, inverse = np.unique(mapped, return_inverse=True)
        embeddings = self._encode_nodes(nodes, target_t)
        indices = torch.as_tensor(inverse.reshape(-1, 2), dtype=torch.long,
                                  device=self.device)
        source, target = embeddings[indices[:, 0]], embeddings[indices[:, 1]]
        decoder = self._model.edge_predictor

        def score(left, right):
            return decoder.out_fc(
                torch.relu(decoder.src_fc(left) + decoder.dst_fc(right))
            ).squeeze(1).sigmoid()

        probabilities = score(source, target)
        if undirected:
            probabilities = (probabilities + score(target, source)) * 0.5
        # The event CLI uses zero logits for a wholly empty batch. Make the
        # analogous cold/cold fallback per edge, independent of its batch.
        cold = torch.as_tensor((mapped < 0).all(axis=1), device=self.device)
        probabilities = probabilities.masked_fill(cold, 0.5)
        return probabilities.cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def score_edges(self, edges, target_t=None, *, batch_size=65536, undirected=True):
        """Return float32 scores in input order without updating history.

        undirected=True averages the two Time probabilities. Structure is
        symmetric. Hybrid uses boundary-wide recency weights, not batch stats.
        Only the immediate next snapshot is supported; target_t defaults to it.
        """
        if self.prepared_t is None:
            raise RuntimeError("call prepare_time(t) before scoring")
        target_t = self.prepared_t + 1 if target_t is None else _integer(target_t, "target_t")
        if target_t != self.prepared_t + 1:
            raise ValueError("target_t must equal prepared_t + 1")
        batch_size = _integer(batch_size, "batch_size", 1)
        if not isinstance(undirected, (bool, np.bool_)):
            raise ValueError("undirected must be boolean")
        edges = _edge_array(edges)
        result = np.empty(len(edges), dtype=np.float32)
        for start in range(0, len(edges), batch_size):
            batch = edges[start:start + batch_size]
            mapped = np.asarray([
                (self._node_mapping.get(int(u), -1), self._node_mapping.get(int(v), -1))
                for u, v in batch
            ], dtype=np.int64)
            structure = (_structure_scores(self._structure, mapped)
                         if self._structure is not None else None)
            if self.config.branch == "structure":
                values = structure
            else:
                time = self._time_scores(mapped, target_t, undirected)
                if self.config.branch == "time":
                    values = time
                else:
                    weights = np.full(mapped.shape, self._cold_recency, dtype=np.float32)
                    known = mapped >= 0
                    weights[known] = self._recency[mapped[known]]
                    factor = np.float32(self.config.yita) * weights.mean(axis=1)
                    values = structure + factor * time
            if not np.isfinite(values).all():
                raise FloatingPointError("non-finite edge scores")
            result[start:start + len(batch)] = values
        return result
