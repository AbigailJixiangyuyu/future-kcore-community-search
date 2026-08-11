"""Multi-order h-index features for the structural awareness module."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


MAX_H_INDEX_ORDER = 3


def _h_index(values):
    """Return the h-index of a finite iterable of non-negative integers."""
    values = list(values)
    limit = len(values)
    counts = [0] * (limit + 1)
    for value in values:
        counts[min(int(value), limit)] += 1

    at_least = 0
    for candidate in range(limit, -1, -1):
        at_least += counts[candidate]
        if at_least >= candidate:
            return candidate
    return 0


def compute_h_index_levels(edge_list, nodes, max_order=MAX_H_INDEX_ORDER):
    """Compute snapshot-wide h-index values from order 1 through max_order.

    Order zero is the node degree. Every higher order applies the h-index to
    the previous-order values of the node's current neighbors.
    """
    if not isinstance(max_order, int):
        raise TypeError("max_order must be an integer")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    if not isinstance(nodes, Iterable):
        raise TypeError("nodes must be iterable")

    node_set = set(nodes)
    adjacency = {node: set() for node in node_set}
    for edge in edge_list:
        if len(edge) < 2:
            raise ValueError("each edge must contain at least two endpoints")
        u, v = edge[0], edge[1]
        node_set.update((u, v))
        adjacency.setdefault(u, set())
        adjacency.setdefault(v, set())
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)

    previous = {node: len(adjacency[node]) for node in node_set}
    levels = {}
    for order in range(1, max_order + 1):
        current = {
            node: _h_index(previous[neighbor] for neighbor in adjacency[node])
            for node in node_set
        }
        levels[order] = current
        previous = current
    return levels


def structure_representation(snapshots, u, t, order, cmax, adjacency=None):
    """Return the fixed-width structural representation S_t(u).

    The first ``order - 1`` blocks are closed-neighborhood distributions for
    h-index orders 1 through ``order - 1``. The last block is the corresponding
    distribution of current coreness. Every block uses exact buckets 0 through
    ``cmax - 1`` plus one overflow bucket for values greater than or equal to
    ``cmax``. Its width is therefore ``cmax + 1``.
    """
    if not isinstance(snapshots, Sequence):
        raise TypeError("snapshots must be an ordered sequence")
    if not isinstance(t, int):
        raise TypeError("t must be an integer snapshot index")
    if t < 0 or t >= len(snapshots):
        raise IndexError(f"t must be in [0, {len(snapshots) - 1}]")
    if not isinstance(order, int):
        raise TypeError("order must be an integer")
    if not 1 <= order <= MAX_H_INDEX_ORDER + 1:
        raise ValueError(f"order must be in [1, {MAX_H_INDEX_ORDER + 1}]")
    if not isinstance(cmax, int):
        raise TypeError("cmax must be an integer")
    if cmax < 0:
        raise ValueError("cmax must be non-negative")

    snapshot = snapshots[t]
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"snapshot {t} must be a mapping")
    core_dict = snapshot.get("core_dict")
    edge_list = snapshot.get("edge_list")
    if not isinstance(core_dict, Mapping):
        raise ValueError(f"snapshot {t} must contain a core_dict mapping")
    if not isinstance(edge_list, Sequence):
        raise ValueError(f"snapshot {t} must contain an edge_list sequence")

    value_maps = []
    if order > 1:
        h_index_dicts = snapshot.get("h_index_dicts")
        if not isinstance(h_index_dicts, Mapping):
            raise ValueError(
                f"snapshot {t} has no preprocessed h_index_dicts; "
                "rebuild its snapshot cache"
            )
        for h_order in range(1, order):
            values = h_index_dicts.get(h_order)
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"snapshot {t} has no preprocessed order-{h_order} h-index"
                )
            value_maps.append(values)
    value_maps.append(core_dict)

    closed_neighborhood = {u}
    if adjacency is not None:
        if not isinstance(adjacency, Mapping):
            raise TypeError("adjacency must be a mapping")
        closed_neighborhood.update(adjacency.get(u, ()))
    else:
        for edge in edge_list:
            if len(edge) < 2:
                raise ValueError("each edge must contain at least two endpoints")
            left, right = edge[0], edge[1]
            if left == u:
                closed_neighborhood.add(right)
            elif right == u:
                closed_neighborhood.add(left)

    blocks = []
    for values in value_maps:
        block = np.zeros(cmax + 1, dtype=np.float64)
        for node in closed_neighborhood:
            value = int(values.get(node, 0))
            if value < 0:
                raise ValueError("structural feature values must be non-negative")
            block[min(value, cmax)] += 1.0
        block /= len(closed_neighborhood)
        blocks.append(block)

    return np.concatenate(blocks)
