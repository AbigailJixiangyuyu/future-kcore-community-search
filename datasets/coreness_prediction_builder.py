"""Build causal node-level samples and tensors for coreness prediction."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.baseline_eval import evaluation_start_t
from datasets.baseline_split import training_boundaries
from methods.t_ppr import TemporalPPR
from methods.h_index_representation import (
    validated_structure_distributions,
)
from methods.tcs_representation import TCSStreamingIndex, tcs_representation


DEFAULT_CORE_HISTORY = 5
STRUCTURE_TIME_REFERENCE = "current_observed_snapshot"


@dataclass(frozen=True)
class CorenessSample:
    """Predict one node's coreness in the snapshot after ``time``."""

    node: int
    time: int
    label: int


class TimeSliceStructureFeatureTable:
    """Deduplicate temporal-node structures within one query time slice."""

    def __init__(self, temporal_ppr, order, hmax):
        self.temporal_ppr = temporal_ppr
        self.order = int(order)
        self.hmax = int(hmax)
        self.width = self.order * (self.hmax + 1)
        self._indices = {}
        self._keys = [None]
        self._values = None

    def index(self, node, time):
        if self._values is not None:
            raise RuntimeError("a materialized structure table is immutable")
        key = (int(node), int(time))
        structure_index = self._indices.get(key)
        if structure_index is not None:
            return structure_index
        structure_index = len(self._keys)
        if structure_index > np.iinfo(np.int32).max:
            raise OverflowError(
                "the inference structure table exceeds int32 indexing"
            )
        self._indices[key] = structure_index
        self._keys.append(key)
        return structure_index

    def index_many(self, nodes, times, mask):
        """Map temporal-node slots in bulk, preserving first-appearance order."""
        if self._values is not None:
            raise RuntimeError("a materialized structure table is immutable")
        nodes = np.asarray(nodes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        mask = np.asarray(mask, dtype=np.bool_)
        if nodes.shape != times.shape or nodes.shape != mask.shape:
            raise ValueError("nodes, times and mask must have identical shapes")
        result = np.zeros(nodes.shape, dtype=np.int32)
        if not np.any(mask):
            return result
        # Separate int64 fields avoid overflow/collisions from packed node IDs.
        keys = np.empty(int(mask.sum()), dtype=[("node", np.int64), ("time", np.int64)])
        keys["node"] = nodes[mask]
        keys["time"] = times[mask]
        unique, first, inverse = np.unique(
            keys, return_index=True, return_inverse=True
        )
        indices = np.empty(len(unique), dtype=np.int32)
        for position in np.argsort(first):
            key = unique[position]
            indices[position] = self.index(key["node"], key["time"])
        result[mask] = indices[inverse]
        return result

    def materialize(self):
        if self._values is not None:
            return self
        self._values = np.empty(
            (len(self._keys), self.width), dtype=np.float32
        )
        self._values[0] = 0.0
        for structure_index, key in enumerate(self._keys[1:], start=1):
            self._values[structure_index] = (
                self.temporal_ppr.structure_feature(
                    key[0],
                    key[1],
                    order=self.order,
                    cmax=self.hmax,
                ).astype(np.float32, copy=False)
            )
        return self

    @property
    def values(self):
        self.materialize()
        return self._values

    def clear(self):
        self._indices.clear()
        self._keys.clear()
        self._values = np.empty((0, self.width), dtype=np.float32)

    def __len__(self):
        return len(self._keys)


class CurrentSnapshotStructureFeatureTable:
    """Read-only model structures from one current observed snapshot only."""

    def __init__(self, values):
        self.values = values.view()
        self.values.flags.writeable = False

    def clear(self):
        self.values = np.empty((0, self.values.shape[1]), dtype=np.float32)

    def __len__(self):
        return len(self.values)


def _current_structure_table(temporal_ppr, order, hmax, t, influences):
    """Map selected neighbor IDs to G_t, never to their T-PPR event times."""
    snapshot = temporal_ppr.snapshots[t]
    if order != 4:
        return None
    validated = validated_structure_distributions(snapshot, hmax)
    if validated is None:
        return None
    cached, node_ids = validated
    known = np.asarray(node_ids, dtype=np.int64)
    if len(known) + 1 > np.iinfo(np.int32).max:
        raise OverflowError("current structure table exceeds int32 indexing")
    values = np.zeros((len(known) + 2, order * (hmax + 1)), dtype=np.float32)
    # Row 0: invalid slot. Row 1: a selected node absent from G_t; its
    # closed neighborhood is itself with zero h-index/core in all four groups.
    values[1, ::hmax + 1] = 1.0
    values[2:] = cached["values"]
    mask = influences["mask"]
    requested = influences["nodes"][mask]
    positions = np.searchsorted(known, requested)
    found = positions < len(known)
    found[found] &= known[positions[found]] == requested[found]
    selected = np.ones(len(requested), dtype=np.int32)
    selected[found] = positions[found] + 2
    indices = np.zeros(mask.shape, dtype=np.int32)
    indices[mask] = selected
    return indices, CurrentSnapshotStructureFeatureTable(values)


def _sample_by_current_coreness(nodes, current_core, limit, seed):
    """Causally stratify nodes using only their coreness in G_t."""
    groups = {}
    for node in nodes:
        groups.setdefault(int(current_core.get(node, 0)), []).append(node)
    rng = np.random.RandomState(seed)
    shuffled_groups = {}
    for coreness, values in groups.items():
        group = np.asarray(values, dtype=np.int64)
        rng.shuffle(group)
        shuffled_groups[coreness] = group

    selected = []
    positions = {coreness: 0 for coreness in shuffled_groups}
    group_order = sorted(shuffled_groups, reverse=True)
    while len(selected) < limit:
        added = False
        for coreness in group_order:
            position = positions[coreness]
            group = shuffled_groups[coreness]
            if position < len(group):
                selected.append(int(group[position]))
                positions[coreness] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return np.array(sorted(selected), dtype=np.int64)


def prediction_time_splits(snapshot_count, train_ratio=0.7, val_ratio=0.15):
    """Split by current time t, with each label taken from snapshot t+1."""
    if snapshot_count < 4:
        raise ValueError("coreness prediction requires at least four snapshots")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    if train_ratio == 0.55 and val_ratio == 0.15:
        train_end, fit_end = training_boundaries(snapshot_count)
    elif train_ratio == 0.7 and val_ratio == 0.15:
        train_end = int(snapshot_count * train_ratio) - 1
        fit_end = evaluation_start_t(snapshot_count)
        if not 1 <= train_end < fit_end:
            raise ValueError("not enough snapshots for nonempty train/val/test splits")
    else:
        train_end = int(snapshot_count * train_ratio) - 1
        fit_end = int(snapshot_count * (train_ratio + val_ratio))
        if not 1 <= train_end < fit_end < snapshot_count - 1:
            raise ValueError("not enough snapshots for nonempty train/val/test splits")

    return {
        "train": list(range(train_end)),
        "val": list(range(train_end, fit_end)),
        "test": list(range(fit_end, snapshot_count - 1)),
    }


def prediction_split_config(snapshot_count, train_ratio=0.7, val_ratio=0.15):
    """Record the actual query-time boundaries used to prepare Ours samples."""
    times = prediction_time_splits(snapshot_count, train_ratio, val_ratio)
    train_end_t = times["val"][0]
    fit_end_t = times["test"][0]
    if (train_ratio, val_ratio) == (0.7, 0.15):
        split_rule = "snapshot_70_15_15_current_query_v1"
    elif (train_ratio, val_ratio) == (0.55, 0.15):
        split_rule = "snapshot_55_15_30_v1"
    else:
        split_rule = "custom_snapshot_split_v1"
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "split_rule": split_rule,
        "snapshot_count": snapshot_count,
        "train_end_t": train_end_t,
        "fit_end_t": fit_end_t,
        "first_test_target_t": fit_end_t + 1,
    }


def build_coreness_samples(
    snapshots,
    train_ratio=0.7,
    val_ratio=0.15,
    max_nodes_per_time=None,
    seed=42,
):
    """Build ``(u, t) -> c_(t+1)(u)`` samples without future-node leakage.

    Candidate nodes are those observed at least once through ``G_t``. Nodes
    appearing for the first time in ``G_(t+1)`` are deliberately excluded
    because they are not available to a causal predictor at time ``t``.
    """
    if max_nodes_per_time is not None:
        if not isinstance(max_nodes_per_time, int):
            raise TypeError("max_nodes_per_time must be an integer or None")
        if max_nodes_per_time <= 0:
            raise ValueError("max_nodes_per_time must be positive")

    times_by_split = prediction_time_splits(
        len(snapshots), train_ratio=train_ratio, val_ratio=val_ratio
    )
    split_by_time = {
        time: split
        for split, times in times_by_split.items()
        for time in times
    }
    samples = {"train": [], "val": [], "test": []}
    observed_nodes = set()

    for time in range(len(snapshots) - 1):
        observed_nodes.update(snapshots[time]["core_dict"])
        candidates = np.array(sorted(observed_nodes), dtype=np.int64)
        if max_nodes_per_time is not None and len(candidates) > max_nodes_per_time:
            candidates = _sample_by_current_coreness(
                candidates,
                snapshots[time]["core_dict"],
                max_nodes_per_time,
                seed + time,
            )

        next_core = snapshots[time + 1]["core_dict"]
        split = split_by_time[time]
        samples[split].extend(
            CorenessSample(
                node=int(node),
                time=time,
                label=int(next_core.get(int(node), 0)),
            )
            for node in candidates
        )
    return samples


def build_prediction_samples(snapshots, t):
    """Return every causally observable node at time ``t`` for inference."""
    if not isinstance(t, int):
        raise TypeError("t must be an integer snapshot index")
    if t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    observed_nodes = set()
    for snapshot in snapshots[:t + 1]:
        observed_nodes.update(snapshot["core_dict"])
    return [
        CorenessSample(node=int(node), time=t, label=0)
        for node in sorted(observed_nodes)
    ]


def add_core_history_tokens(
    arrays, snapshots, kmax, lookback=DEFAULT_CORE_HISTORY
):
    """Attach recent coreness token IDs to sample feature arrays in place."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    nodes = arrays.get("nodes")
    times = arrays.get("times")
    if nodes is None or times is None or len(nodes) != len(times):
        raise ValueError("arrays must contain aligned nodes and times")
    # Missing nodes and positions before the first snapshot have coreness zero.
    history = np.zeros((len(nodes), lookback), dtype=np.int64)
    history_time = None
    history_cores = ()
    for row, (node_value, time_value) in enumerate(zip(nodes, times)):
        node = int(node_value)
        time = int(time_value)
        if time != history_time:
            # Borrow at most one history window. Drop the previous references
            # before loading another; never pin all training-time snapshots.
            history_cores = ()
            core_dict = None
            history_cores = tuple(
                snapshots[snapshot_time]["core_dict"]
                for snapshot_time in range(time, max(-1, time - lookback), -1)
            )
            history_time = time
        for lag, core_dict in enumerate(history_cores):
            if node not in core_dict:
                continue
            coreness = int(core_dict[node])
            if coreness < 0 or coreness > kmax:
                raise ValueError("snapshot coreness is outside [0, kmax]")
            history[row, lag] = coreness
    arrays["core_history"] = history
    return arrays


def _empty_feature_arrays(sample_count, kmax, top_l, structure_width,
                          lookback=DEFAULT_CORE_HISTORY):
    return {
        "temporal": np.zeros((sample_count, kmax), dtype=np.float32),
        "core_history": np.zeros((sample_count, lookback), dtype=np.int64),
        "neighbor_structures": np.zeros(
            (sample_count, top_l, structure_width), dtype=np.float32
        ),
        "time_deltas": np.zeros((sample_count, top_l), dtype=np.float32),
        "weights": np.zeros((sample_count, top_l), dtype=np.float32),
        "mask": np.zeros((sample_count, top_l), dtype=np.bool_),
        "labels": np.zeros(sample_count, dtype=np.int64),
        "nodes": np.zeros(sample_count, dtype=np.int64),
        "times": np.zeros(sample_count, dtype=np.int64),
    }


def _empty_indexed_feature_arrays(sample_count, kmax, top_l,
                                  lookback=DEFAULT_CORE_HISTORY):
    return {
        "temporal": np.zeros((sample_count, kmax), dtype=np.float32),
        "core_history": np.zeros((sample_count, lookback), dtype=np.int64),
        "structure_indices": np.zeros(
            (sample_count, top_l), dtype=np.int32
        ),
        "time_deltas": np.zeros((sample_count, top_l), dtype=np.float32),
        "weights": np.zeros((sample_count, top_l), dtype=np.float32),
        "mask": np.zeros((sample_count, top_l), dtype=np.bool_),
        "labels": np.zeros(sample_count, dtype=np.int64),
        "nodes": np.zeros(sample_count, dtype=np.int64),
        "times": np.zeros(sample_count, dtype=np.int64),
    }


def prepare_feature_arrays_by_split(
    snapshots,
    samples_by_split,
    kmax,
    hmax,
    top_l=20,
    internal_top_k=80,
    order=4,
    t_ppr_alpha=0.3,
    t_ppr_beta=0.5,
    min_probability=1e-8,
    t_ppr=None,
    progress=None,
    structure_table_path=None,
):
    """Build indexed split features with one incremental T-PPR scan.

    The returned structure table is shared by every split. Each sample stores
    only Top-L row indices into that table, rather than its own copies of the
    corresponding structure vectors. Row zero is reserved for padded slots.
    """
    if not isinstance(top_l, int) or top_l <= 0:
        raise ValueError("top_l must be a positive integer")
    if not isinstance(internal_top_k, int) or internal_top_k < top_l:
        raise ValueError("internal_top_k must be an integer at least top_l")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    if hmax < 0:
        raise ValueError("hmax must be non-negative")
    structure_width = order * (hmax + 1)
    arrays_by_split = {}
    locations = defaultdict(list)
    queries_by_time = defaultdict(set)

    for split, samples in samples_by_split.items():
        arrays = _empty_indexed_feature_arrays(len(samples), kmax, top_l)
        arrays_by_split[split] = arrays
        for index, sample in enumerate(samples):
            arrays["labels"][index] = sample.label
            arrays["nodes"][index] = sample.node
            arrays["times"][index] = sample.time
            locations[(sample.node, sample.time)].append((split, index))
            queries_by_time[sample.time].add(sample.node)
        add_core_history_tokens(arrays, snapshots, kmax)

    tcs_index = TCSStreamingIndex(snapshots, kmax=kmax)
    for time in sorted(queries_by_time):
        tcs_index.advance_to(time)
        nodes = sorted(queries_by_time[time])
        representations = tcs_index.representations(nodes).astype(
            np.float32, copy=False
        )
        for node, representation in zip(nodes, representations):
            for split, index in locations[(node, time)]:
                arrays_by_split[split]["temporal"][index] = representation

    total = sum(len(samples) for samples in samples_by_split.values())
    done = 0
    structure_indices = {}
    structure_keys = [None]
    t_ppr = t_ppr or TemporalPPR(
        snapshots, alpha=t_ppr_alpha, beta=t_ppr_beta
    )
    for node, time, influences in t_ppr.incremental_top_neighbors(
        queries_by_time,
        top_l=top_l,
        internal_top_k=internal_top_k,
        min_score=min_probability,
    ):
        for split, index in locations[(node, time)]:
            arrays = arrays_by_split[split]
            for position, influence in enumerate(influences):
                structure_key = (influence.node, time)
                structure_index = structure_indices.get(structure_key)
                if structure_index is None:
                    structure_index = len(structure_keys)
                    if structure_index > np.iinfo(np.int32).max:
                        raise OverflowError(
                            "the global structure table exceeds int32 indexing"
                        )
                    structure_indices[structure_key] = structure_index
                    structure_keys.append(structure_key)
                arrays["structure_indices"][index, position] = structure_index
                arrays["time_deltas"][index, position] = (
                    time - influence.time
                )
                arrays["weights"][index, position] = influence.weight
                arrays["mask"][index, position] = True

            done += 1
            if progress is not None:
                progress(done, total)

    del locations, queries_by_time, structure_indices
    table_shape = (len(structure_keys), structure_width)
    if structure_table_path is None:
        structure_table = np.empty(table_shape, dtype=np.float32)
    else:
        structure_table_path = Path(structure_table_path)
        structure_table_path.parent.mkdir(parents=True, exist_ok=True)
        structure_table = np.lib.format.open_memmap(
            structure_table_path,
            mode="w+",
            dtype=np.float32,
            shape=table_shape,
        )
    structure_table[0] = 0.0
    for structure_index, structure_key in enumerate(
        structure_keys[1:], start=1
    ):
        node, time = structure_key
        structure_table[structure_index] = t_ppr.structure_feature(
            node,
            time,
            order=order,
            cmax=hmax,
        ).astype(np.float32, copy=False)
    if isinstance(structure_table, np.memmap):
        structure_table.flush()
    return arrays_by_split, structure_table


def prepare_feature_arrays(
    snapshots,
    samples,
    kmax,
    hmax,
    top_l=20,
    internal_top_k=80,
    order=4,
    t_ppr_alpha=0.3,
    t_ppr_beta=0.5,
    min_probability=1e-8,
    t_ppr=None,
    progress=None,
):
    """Materialize one sample collection using incremental T-PPR."""
    arrays_by_split, structure_table = prepare_feature_arrays_by_split(
        snapshots,
        {"samples": samples},
        kmax=kmax,
        hmax=hmax,
        top_l=top_l,
        internal_top_k=internal_top_k,
        order=order,
        t_ppr_alpha=t_ppr_alpha,
        t_ppr_beta=t_ppr_beta,
        min_probability=min_probability,
        t_ppr=t_ppr,
        progress=progress,
    )
    arrays = arrays_by_split["samples"]
    structure_indices = arrays.pop("structure_indices")
    arrays["neighbor_structures"] = structure_table[structure_indices]
    return arrays


def prepare_inference_feature_table(
    snapshots,
    nodes,
    t,
    kmax,
    hmax,
    t_ppr_index,
    tcs_index,
    order=4,
    core_lookback=DEFAULT_CORE_HISTORY,
):
    """Materialize every requested node's indexed model input at one time.

    The returned node rows are sorted and contain only compact indices into one
    shared structure table. Community queries can therefore gather rows without
    recomputing TCS, T-PPR Top-L, or structural representations.
    """
    if not isinstance(t, int) or t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    if t_ppr_index.current_time != t:
        raise ValueError("t_ppr_index must be advanced exactly to t")
    if tcs_index.current_time != t:
        raise ValueError("tcs_index must be advanced exactly to t")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    if hmax < 0:
        raise ValueError("hmax must be non-negative")

    nodes = np.asarray(sorted(set(nodes)), dtype=np.int64)
    top_l = t_ppr_index.top_l
    arrays = _empty_indexed_feature_arrays(
        len(nodes), kmax, top_l, lookback=core_lookback
    )
    arrays["nodes"] = nodes
    arrays["times"].fill(t)
    add_core_history_tokens(
        arrays, snapshots, kmax, lookback=core_lookback
    )
    if len(nodes):
        arrays["temporal"][:] = tcs_index.representations(nodes).astype(
            np.float32, copy=False
        )

    influences = t_ppr_index.top_neighbor_arrays(nodes)
    mask = influences["mask"]
    fixed = _current_structure_table(
        t_ppr_index.temporal_ppr, order, hmax, t, influences
    )
    if fixed is None:
        # Legacy/manual snapshots or unsupported cache widths retain old behavior.
        structure_table = TimeSliceStructureFeatureTable(
            t_ppr_index.temporal_ppr, order=order, hmax=hmax
        )
        arrays["structure_indices"] = structure_table.index_many(
            influences["nodes"], np.full(mask.shape, t, dtype=np.int64), mask
        )
        structure_table.materialize()
    else:
        arrays["structure_indices"], structure_table = fixed
    np.subtract(
        t, influences["times"].astype(np.int64),
        out=arrays["time_deltas"], where=mask, casting="unsafe",
    )
    arrays["weights"][:] = influences["weights"]
    arrays["mask"][:] = mask

    return arrays, structure_table


def prepare_inference_feature_arrays(
    snapshots,
    nodes,
    t,
    kmax,
    hmax,
    t_ppr_index,
    tcs_index=None,
    order=4,
    structure_cache=None,
    tcs_cache=None,
    influence_cache=None,
    core_lookback=DEFAULT_CORE_HISTORY,
):
    """Build model inputs only for the requested nodes at one query time.

    ``t_ppr_index`` must already be advanced to ``t``. The three optional caches
    are scoped to one query time. Reusing them across queries ensures each
    node's TCS and Top-L influences, and each temporal node's structure
    representation, are computed at most once at that time.
    """
    if not isinstance(t, int) or t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    if t_ppr_index.current_time != t:
        raise ValueError("t_ppr_index must be advanced exactly to t")
    if tcs_index is not None and tcs_index.current_time != t:
        raise ValueError("tcs_index must be advanced exactly to t")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    if hmax < 0:
        raise ValueError("hmax must be non-negative")

    nodes = np.asarray(sorted(set(nodes)), dtype=np.int64)
    top_l = t_ppr_index.top_l
    structure_width = order * (hmax + 1)
    arrays = _empty_feature_arrays(
        len(nodes), kmax, top_l, structure_width, lookback=core_lookback
    )
    arrays["nodes"] = nodes
    arrays["times"].fill(t)
    add_core_history_tokens(
        arrays, snapshots, kmax, lookback=core_lookback
    )
    structure_cache = structure_cache if structure_cache is not None else {}
    tcs_cache = tcs_cache if tcs_cache is not None else {}
    influence_cache = influence_cache if influence_cache is not None else {}

    for index, node_value in enumerate(nodes):
        node = int(node_value)
        temporal = tcs_cache.get(node)
        if temporal is None:
            if tcs_index is None:
                temporal = tcs_representation(
                    snapshots, node, t, kmax=kmax
                )
            else:
                temporal = tcs_index.representation(node)
            temporal = temporal.astype(np.float32, copy=False)
            tcs_cache[node] = temporal
        arrays["temporal"][index] = temporal
        influences = influence_cache.get(node)
        if influences is None:
            influences = tuple(t_ppr_index.top_neighbors(node))
            influence_cache[node] = influences
        for position, influence in enumerate(influences):
            structure_key = (influence.node, t)
            structure = structure_cache.get(structure_key)
            if structure is None:
                structure = t_ppr_index.temporal_ppr.structure_feature(
                    influence.node,
                    t,
                    order=order,
                    cmax=hmax,
                ).astype(np.float32, copy=False)
                structure_cache[structure_key] = structure
            arrays["neighbor_structures"][index, position] = structure
            arrays["time_deltas"][index, position] = t - influence.time
            arrays["weights"][index, position] = influence.weight
            arrays["mask"][index, position] = True
    return arrays
