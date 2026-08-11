"""Build causal node-level samples and tensors for coreness prediction."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from methods.t_ppr import TemporalPPR
from methods.tcs_representation import tcs_representation


@dataclass(frozen=True)
class CorenessSample:
    """Predict one node's coreness in the snapshot after ``time``."""

    node: int
    time: int
    label: int


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
    """Split target snapshots chronologically into train, validation, and test."""
    if snapshot_count < 4:
        raise ValueError("coreness prediction requires at least four snapshots")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    # The split is based on target snapshot t+1, so no target snapshot appears
    # in more than one split. Snapshot zero is history only.
    train_end = max(2, int(snapshot_count * train_ratio))
    val_end = max(train_end + 1, int(snapshot_count * (train_ratio + val_ratio)))
    val_end = min(val_end, snapshot_count - 1)
    if val_end <= train_end:
        raise ValueError("not enough snapshots for a validation interval")

    current_times = range(snapshot_count - 1)
    return {
        "train": [time for time in current_times if time + 1 < train_end],
        "val": [
            time
            for time in current_times
            if train_end <= time + 1 < val_end
        ],
        "test": [time for time in current_times if time + 1 >= val_end],
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


def _empty_feature_arrays(sample_count, kmax, top_l, structure_width):
    return {
        "temporal": np.zeros((sample_count, kmax), dtype=np.float32),
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
):
    """Materialize multiple sample splits with one incremental T-PPR scan."""
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
        arrays = _empty_feature_arrays(
            len(samples), kmax, top_l, structure_width
        )
        arrays_by_split[split] = arrays
        for index, sample in enumerate(samples):
            arrays["temporal"][index] = tcs_representation(
                snapshots,
                sample.node,
                sample.time,
                kmax=kmax,
            )
            arrays["labels"][index] = sample.label
            arrays["nodes"][index] = sample.node
            arrays["times"][index] = sample.time
            locations[(sample.node, sample.time)].append((split, index))
            queries_by_time[sample.time].add(sample.node)

    total = sum(len(samples) for samples in samples_by_split.values())
    done = 0
    structure_cache = {}
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
                structure_key = (influence.node, influence.time)
                structure = structure_cache.get(structure_key)
                if structure is None:
                    structure = t_ppr.structure_feature(
                        influence.node,
                        influence.time,
                        order=order,
                        cmax=hmax,
                    ).astype(np.float32, copy=False)
                    structure_cache[structure_key] = structure
                arrays["neighbor_structures"][index, position] = structure
                arrays["time_deltas"][index, position] = (
                    time - influence.time
                )
                arrays["weights"][index, position] = influence.weight
                arrays["mask"][index, position] = True

            done += 1
            if progress is not None:
                progress(done, total)

    return arrays_by_split


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
    return prepare_feature_arrays_by_split(
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
    )["samples"]
