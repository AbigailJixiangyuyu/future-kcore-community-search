"""Temporal coreness stability representation for a single node."""

from collections.abc import Mapping, Sequence

import numpy as np


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
        A float64 NumPy vector whose element at index ``k - 1`` is
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
        return np.empty(0, dtype=np.float64)

    ks = np.arange(1, kmax + 1, dtype=np.float64)
    cumulative_penalty = np.zeros(kmax, dtype=np.float64)
    weight_sum = 0.0

    for snapshot in history:
        coreness = float(snapshot["core_dict"].get(u, 0))
        denominators = np.maximum(ks, coreness)
        penalty = (ks - coreness) / denominators
        cumulative_penalty *= alpha
        cumulative_penalty += penalty
        weight_sum = alpha * weight_sum + 1.0

    return 1.0 - cumulative_penalty / weight_sum
