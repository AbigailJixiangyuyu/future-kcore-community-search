"""Temporal coreness stability representation for a single node."""

from collections.abc import Mapping, Sequence

import numpy as np


class TCSStreamingIndex:
    """Maintain every observed node's TCS state as snapshots arrive.

    A snapshot update touches only nodes present in that snapshot. Missing
    snapshots contribute the same unit penalty for every k, so gaps can be
    applied lazily with the closed form of the exponential recurrence.
    """

    _INITIAL_CAPACITY = 1024
    _MAX_DENSE_NODE_ID = 10_000_000

    def __init__(self, snapshots, kmax, alpha=0.7):
        if not isinstance(snapshots, Sequence):
            raise TypeError("snapshots must be an ordered sequence")
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        if not isinstance(kmax, int):
            raise TypeError("kmax must be an integer")
        if kmax < 0:
            raise ValueError("kmax must be non-negative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.snapshots = snapshots
        self.kmax = kmax
        self.alpha = np.float32(alpha)
        self.current_time = -1
        self.weight_sum = np.float32(0.0)
        self._ks = np.arange(1, kmax + 1, dtype=np.float32)
        self._penalties = np.zeros(
            (self._INITIAL_CAPACITY, kmax), dtype=np.float32
        )
        self._last_times = np.full(
            self._INITIAL_CAPACITY, -1, dtype=np.int32
        )
        self._sparse_states = {}

    def _ensure_capacity(self, node):
        if node < len(self._last_times):
            return
        capacity = len(self._last_times)
        while capacity <= node:
            capacity *= 2
        capacity = min(capacity, self._MAX_DENSE_NODE_ID + 1)
        if capacity <= node:
            return

        penalties = np.zeros((capacity, self.kmax), dtype=np.float32)
        penalties[:len(self._penalties)] = self._penalties
        last_times = np.full(capacity, -1, dtype=np.int32)
        last_times[:len(self._last_times)] = self._last_times
        self._penalties = penalties
        self._last_times = last_times

    def _is_dense_node(self, node):
        return 0 <= node <= self._MAX_DENSE_NODE_ID

    def _apply_absent_steps(self, penalties, steps):
        if steps <= 0 or self.kmax == 0:
            return penalties
        decay = np.power(self.alpha, np.float32(steps))
        return (
            decay * penalties
            + (1.0 - decay) / (1.0 - self.alpha)
        )

    def _advance_node(self, node, coreness, time, previous_weight_sum):
        state = self._sparse_states.get(node)
        if state is None:
            penalties = np.full(
                self.kmax, previous_weight_sum, dtype=np.float32
            )
        else:
            penalties = self._apply_absent_steps(
                state[0], time - state[1] - 1
            )

        if self.kmax:
            current_penalty = (
                (self._ks - float(coreness))
                / np.maximum(self._ks, float(coreness))
            )
            penalties = self.alpha * penalties + current_penalty

        self._sparse_states[node] = (penalties, time)

    def _advance_dense_nodes(
        self, nodes, coreness, time, previous_weight_sum
    ):
        """Update one snapshot's dense-ID nodes with array operations."""
        if not len(nodes):
            return

        self._ensure_capacity(int(nodes.max()))
        last_times = self._last_times[nodes]
        observed = last_times >= 0
        penalties = np.full(
            (len(nodes), self.kmax),
            previous_weight_sum,
            dtype=np.float32,
        )

        if self.kmax and np.any(observed):
            observed_nodes = nodes[observed]
            observed_penalties = self._penalties[observed_nodes]
            absent_steps = time - last_times[observed] - 1
            decay = np.power(
                self.alpha, absent_steps.astype(np.float32)
            )[:, None]
            penalties[observed] = (
                decay * observed_penalties
                + (1.0 - decay) / (1.0 - self.alpha)
            )

        if self.kmax:
            coreness = coreness.astype(np.float32, copy=False)[:, None]
            current_penalties = (
                (self._ks[None, :] - coreness)
                / np.maximum(self._ks[None, :], coreness)
            )
            penalties = self.alpha * penalties + current_penalties

        self._penalties[nodes] = penalties
        self._last_times[nodes] = time

    def advance_to(self, time):
        """Consume every snapshot through ``time`` exactly once."""
        if not isinstance(time, int):
            raise TypeError("time must be an integer snapshot index")
        if time < 0 or time >= len(self.snapshots):
            raise IndexError(
                f"time must be in [0, {len(self.snapshots) - 1}]"
            )
        if time < self.current_time:
            raise ValueError("a streaming TCS index cannot move backwards")

        for snapshot_time in range(self.current_time + 1, time + 1):
            snapshot = self.snapshots[snapshot_time]
            if not isinstance(snapshot, Mapping):
                raise TypeError(f"snapshot {snapshot_time} must be a mapping")
            core_dict = snapshot.get("core_dict")
            if not isinstance(core_dict, Mapping):
                raise ValueError(
                    f"snapshot {snapshot_time} must contain a core_dict mapping"
                )

            previous_weight_sum = self.weight_sum
            self.weight_sum = self.alpha * self.weight_sum + np.float32(1.0)

            nodes = np.fromiter(core_dict.keys(), dtype=np.int64)
            coreness = np.fromiter(core_dict.values(), dtype=np.int64)
            dense = (
                (nodes >= 0) & (nodes <= self._MAX_DENSE_NODE_ID)
            )
            self._advance_dense_nodes(
                nodes[dense], coreness[dense], snapshot_time,
                previous_weight_sum,
            )
            for node, node_coreness in zip(nodes[~dense], coreness[~dense]):
                self._advance_node(
                    int(node), int(node_coreness), snapshot_time,
                    previous_weight_sum,
                )
            self.current_time = snapshot_time
        return self

    def representation(self, node):
        """Return one node's TCS vector at the current snapshot."""
        if self.current_time < 0:
            raise RuntimeError("advance_to must be called before querying")
        node = int(node)
        if self.kmax == 0:
            return np.empty(0, dtype=np.float32)

        if self._is_dense_node(node):
            if node >= len(self._last_times) or self._last_times[node] < 0:
                return np.zeros(self.kmax, dtype=np.float32)
            last_time = int(self._last_times[node])
            if last_time < self.current_time:
                self._penalties[node] = self._apply_absent_steps(
                    self._penalties[node], self.current_time - last_time
                )
                self._last_times[node] = self.current_time
            penalties = self._penalties[node]
        else:
            state = self._sparse_states.get(node)
            if state is None:
                return np.zeros(self.kmax, dtype=np.float32)
            penalties, last_time = state
            if last_time < self.current_time:
                penalties = self._apply_absent_steps(
                    penalties, self.current_time - last_time
                )
                self._sparse_states[node] = (penalties, self.current_time)

        return 1.0 - penalties / self.weight_sum

    def representations(self, nodes):
        """Return TCS rows for ``nodes`` in the supplied order."""
        nodes = np.asarray(list(nodes), dtype=np.int64)
        if not len(nodes):
            return np.empty((0, self.kmax), dtype=np.float32)
        if self.current_time < 0:
            raise RuntimeError("advance_to must be called before querying")
        if self.kmax == 0:
            return np.empty((len(nodes), 0), dtype=np.float32)

        representations = np.zeros(
            (len(nodes), self.kmax), dtype=np.float32
        )
        dense_positions = np.flatnonzero(
            (nodes >= 0)
            & (nodes <= self._MAX_DENSE_NODE_ID)
            & (nodes < len(self._last_times))
        )
        if len(dense_positions):
            dense_nodes = nodes[dense_positions]
            last_times = self._last_times[dense_nodes]
            observed = last_times >= 0
            if np.any(observed):
                observed_positions = dense_positions[observed]
                observed_nodes = dense_nodes[observed]
                observed_last_times = last_times[observed]
                penalties = self._penalties[observed_nodes]
                steps = self.current_time - observed_last_times
                stale = steps > 0
                if np.any(stale):
                    decay = np.power(
                        self.alpha, steps[stale].astype(np.float32)
                    )[:, None]
                    penalties[stale] = (
                        decay * penalties[stale]
                        + (1.0 - decay) / (1.0 - self.alpha)
                    )
                    self._penalties[observed_nodes[stale]] = penalties[stale]
                    self._last_times[observed_nodes[stale]] = self.current_time
                representations[observed_positions] = (
                    1.0 - penalties / self.weight_sum
                )

        sparse_positions = np.flatnonzero(
            (nodes < 0) | (nodes > self._MAX_DENSE_NODE_ID)
        )
        for position in sparse_positions:
            representations[position] = self.representation(nodes[position])
        return representations


def tcs_representation(snapshots, u, t, kmax, alpha=0.7):
    """Return ``T_t(u) = [TCS_t(1, u), ..., TCS_t(kmax, u)]``.

    Only snapshots ``G_0`` through ``G_t`` are used. ``kmax`` must be the fixed
    dataset-level value produced during snapshot preprocessing. A node absent
    from a snapshot has coreness zero for that update.

    Args:
        snapshots: Ordered snapshots containing a ``core_dict`` mapping.
        u: Node identifier.
        t: Index of the latest observable snapshot.
        kmax: Fixed maximum k determined during dataset preprocessing.
        alpha: Exponential decay applied to older snapshots.

    Returns:
        A float32 NumPy vector whose element at index ``k - 1`` is
        ``TCS_t(k, u)``. The vector is empty when the observed ``kmax`` is zero.
    """
    if not isinstance(snapshots, Sequence):
        raise TypeError("snapshots must be an ordered sequence")
    if not isinstance(t, int):
        raise TypeError("t must be an integer snapshot index")
    if t < 0 or t >= len(snapshots):
        raise IndexError(f"t must be in [0, {len(snapshots) - 1}]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    history = snapshots[:t + 1]
    for index, snapshot in enumerate(history):
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"snapshot {index} must be a mapping")
        core_dict = snapshot.get("core_dict")
        if not isinstance(core_dict, Mapping):
            raise ValueError(f"snapshot {index} must contain a core_dict mapping")

    if not isinstance(kmax, int):
        raise TypeError("kmax must be an integer")
    if kmax < 0:
        raise ValueError("kmax must be non-negative")
    if kmax == 0:
        return np.empty(0, dtype=np.float32)

    ks = np.arange(1, kmax + 1, dtype=np.float32)
    cumulative_penalty = np.zeros(kmax, dtype=np.float32)
    weight_sum = np.float32(0.0)
    alpha = np.float32(alpha)

    for snapshot in history:
        coreness = float(snapshot["core_dict"].get(u, 0))
        denominators = np.maximum(ks, coreness)
        penalty = (ks - coreness) / denominators
        cumulative_penalty *= alpha
        cumulative_penalty += penalty
        weight_sum = alpha * weight_sum + np.float32(1.0)

    return 1.0 - cumulative_penalty / weight_sum
