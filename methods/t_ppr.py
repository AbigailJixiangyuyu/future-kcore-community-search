"""Snapshot-level Temporal Personalized PageRank (T-PPR).

The original T-PPR process is defined on continuous interaction events. This
project works with graph snapshots, so a temporal node is represented as
``(node, snapshot_index)`` and every edge in ``G_t`` is treated as an
interaction at index ``t``. Walks always move to a strictly older snapshot,
which keeps the temporal transition graph acyclic and prevents future leakage.
"""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange

from methods.h_index_representation import structure_representation


@dataclass(frozen=True)
class TemporalInfluence:
    """One selected temporal node and its T-PPR attention weight."""

    node: int
    time: int
    score: float
    weight: float


@njit(inline="always")
def _insert_temporal_candidate(
    hash_nodes, hash_times, hash_scores, node, time, score
):
    """Insert or aggregate one ``(node, time)`` candidate."""
    mask = len(hash_times) - 1
    slot = (node * 1000003 + time) & mask
    while True:
        if hash_times[slot] == -1:
            hash_nodes[slot] = node
            hash_times[slot] = time
            hash_scores[slot] = score
            return
        if hash_nodes[slot] == node and hash_times[slot] == time:
            hash_scores[slot] += score
            return
        slot = (slot + 1) & mask


@njit(inline="always")
def _is_better_candidate(
    score, time, node, ranked_score, ranked_time, ranked_node
):
    """Apply the deterministic T-PPR ranking order."""
    if score != ranked_score:
        return score > ranked_score
    if time != ranked_time:
        return time > ranked_time
    return node < ranked_node


@njit(cache=True, nogil=True, parallel=True)
def _update_source_rows(
    active_rows,
    adjacency_offsets,
    neighbor_rows,
    all_nodes,
    state_nodes,
    state_times,
    state_scores,
    state_lengths,
    norms,
    snapshot_time,
    alpha,
    beta,
    min_score,
    output_nodes,
    output_times,
    output_scores,
    output_lengths,
    output_norms,
):
    """Update active sources independently from the preceding state."""
    width = state_nodes.shape[1]
    for source_position in prange(len(active_rows)):
        source_row = active_rows[source_position]
        edge_start = adjacency_offsets[source_position]
        edge_end = adjacency_offsets[source_position + 1]
        degree = edge_end - edge_start
        old_norm = norms[source_row]
        new_norm = beta * old_norm + degree
        output_norms[source_position] = new_norm

        candidate_count = state_lengths[source_row]
        candidate_count += degree
        for edge_position in range(edge_start, edge_end):
            candidate_count += state_lengths[neighbor_rows[edge_position]]

        hash_capacity = 1
        while hash_capacity < 2 * candidate_count:
            hash_capacity *= 2
        hash_nodes = np.zeros(hash_capacity, dtype=np.int64)
        hash_times = np.full(hash_capacity, -1, dtype=np.int32)
        hash_scores = np.zeros(hash_capacity, dtype=np.float64)

        if old_norm > 0.0:
            old_scale = beta * old_norm / new_norm
            for position in range(state_lengths[source_row]):
                _insert_temporal_candidate(
                    hash_nodes,
                    hash_times,
                    hash_scores,
                    state_nodes[source_row, position],
                    state_times[source_row, position],
                    state_scores[source_row, position] * old_scale,
                )

        neighbor_scale = (1.0 - alpha) / new_norm
        terminal_score = neighbor_scale * alpha
        for edge_position in range(edge_start, edge_end):
            neighbor_row = neighbor_rows[edge_position]
            _insert_temporal_candidate(
                hash_nodes,
                hash_times,
                hash_scores,
                all_nodes[neighbor_row],
                snapshot_time,
                terminal_score,
            )

        for edge_position in range(edge_start, edge_end):
            neighbor_row = neighbor_rows[edge_position]
            for position in range(state_lengths[neighbor_row]):
                _insert_temporal_candidate(
                    hash_nodes,
                    hash_times,
                    hash_scores,
                    state_nodes[neighbor_row, position],
                    state_times[neighbor_row, position],
                    state_scores[neighbor_row, position] * neighbor_scale,
                )

        ranked_length = 0
        for slot in range(hash_capacity):
            candidate_time = hash_times[slot]
            candidate_score = hash_scores[slot]
            if candidate_time < 0 or candidate_score <= min_score:
                continue
            candidate_node = hash_nodes[slot]
            insert_at = ranked_length
            if insert_at > width:
                insert_at = width
            while insert_at > 0 and _is_better_candidate(
                candidate_score,
                candidate_time,
                candidate_node,
                output_scores[source_position, insert_at - 1],
                output_times[source_position, insert_at - 1],
                output_nodes[source_position, insert_at - 1],
            ):
                insert_at -= 1
            if insert_at >= width:
                continue

            move_from = ranked_length
            if move_from >= width:
                move_from = width - 1
            while move_from > insert_at:
                output_nodes[source_position, move_from] = output_nodes[
                    source_position, move_from - 1
                ]
                output_times[source_position, move_from] = output_times[
                    source_position, move_from - 1
                ]
                output_scores[source_position, move_from] = output_scores[
                    source_position, move_from - 1
                ]
                move_from -= 1
            output_nodes[source_position, insert_at] = candidate_node
            output_times[source_position, insert_at] = candidate_time
            output_scores[source_position, insert_at] = candidate_score
            if ranked_length < width:
                ranked_length += 1
        output_lengths[source_position] = ranked_length


class TemporalPPRStreamingIndex:
    """Maintain approximate T-PPR state while advancing through snapshots."""

    def __init__(self, temporal_ppr, top_l=20, internal_top_k=20,
                 min_score=0.0):
        if not isinstance(top_l, int) or top_l <= 0:
            raise ValueError("top_l must be a positive integer")
        if not isinstance(internal_top_k, int) or internal_top_k <= 0:
            raise ValueError("internal_top_k must be a positive integer")
        if internal_top_k < top_l:
            raise ValueError("internal_top_k must be at least top_l")
        if min_score < 0.0:
            raise ValueError("min_score must be non-negative")
        self.temporal_ppr = temporal_ppr
        self.top_l = top_l
        self.internal_top_k = internal_top_k
        self.min_score = float(min_score)
        self.current_time = -1
        self._nodes = np.asarray(
            sorted(int(node) for node in temporal_ppr._history),
            dtype=np.int64,
        )
        shape = (len(self._nodes), self.internal_top_k)
        self._state_nodes = np.zeros(shape, dtype=np.int64)
        self._state_times = np.full(shape, -1, dtype=np.int32)
        self._state_scores = np.zeros(shape, dtype=np.float64)
        self._state_lengths = np.zeros(len(self._nodes), dtype=np.int32)
        self._norms = np.zeros(len(self._nodes), dtype=np.float64)

    def _rows_for_nodes(self, nodes):
        """Map graph node IDs to rows in the fixed-width state arrays."""
        nodes = np.asarray(nodes, dtype=np.int64)
        if not len(nodes):
            return np.empty(0, dtype=np.int64)
        rows = np.searchsorted(self._nodes, nodes)
        if np.any(rows >= len(self._nodes)):
            raise KeyError("T-PPR node is missing from the static node index")
        if not np.array_equal(self._nodes[rows], nodes):
            raise KeyError("T-PPR node is missing from the static node index")
        return rows

    def _advance_snapshot(self, snapshot_time):
        adjacency = self.temporal_ppr._snapshot_adjacency[snapshot_time]
        if not adjacency:
            return

        active_nodes = np.asarray(sorted(adjacency), dtype=np.int64)
        active_rows = self._rows_for_nodes(active_nodes)
        degrees = np.fromiter(
            (len(adjacency[int(node)]) for node in active_nodes),
            dtype=np.int64,
            count=len(active_nodes),
        )
        adjacency_offsets = np.empty(len(active_nodes) + 1, dtype=np.int64)
        adjacency_offsets[0] = 0
        np.cumsum(degrees, out=adjacency_offsets[1:])
        neighbor_nodes = np.fromiter(
            (
                neighbor
                for node in active_nodes
                for neighbor in adjacency[int(node)]
            ),
            dtype=np.int64,
            count=int(degrees.sum()),
        )
        neighbor_rows = self._rows_for_nodes(neighbor_nodes)
        output_shape = (len(active_rows), self.internal_top_k)
        output_nodes = np.zeros(output_shape, dtype=np.int64)
        output_times = np.full(output_shape, -1, dtype=np.int32)
        output_scores = np.zeros(output_shape, dtype=np.float64)
        output_lengths = np.zeros(len(active_rows), dtype=np.int32)
        output_norms = np.zeros(len(active_rows), dtype=np.float64)

        # The kernel reads only the preceding global state. Each worker writes
        # one private output row, so same-snapshot updates cannot leak between
        # neighboring sources.
        _update_source_rows(
            active_rows,
            adjacency_offsets,
            neighbor_rows,
            self._nodes,
            self._state_nodes,
            self._state_times,
            self._state_scores,
            self._state_lengths,
            self._norms,
            snapshot_time,
            self.temporal_ppr.alpha,
            self.temporal_ppr.beta,
            self.min_score,
            output_nodes,
            output_times,
            output_scores,
            output_lengths,
            output_norms,
        )
        self._state_nodes[active_rows] = output_nodes
        self._state_times[active_rows] = output_times
        self._state_scores[active_rows] = output_scores
        self._state_lengths[active_rows] = output_lengths
        self._norms[active_rows] = output_norms

    def advance_to(self, time):
        """Advance the index to ``time`` without reading future snapshots."""
        if not isinstance(time, int):
            raise TypeError("time must be an integer snapshot index")
        if time < 0:
            raise IndexError("time must be non-negative")
        if time < self.current_time:
            raise ValueError("a streaming T-PPR index cannot move backwards")
        if time >= len(self.temporal_ppr.snapshots):
            raise IndexError(
                "time must be in [0, {}]".format(
                    len(self.temporal_ppr.snapshots) - 1
                )
            )

        for snapshot_time in range(self.current_time + 1, time + 1):
            self._advance_snapshot(snapshot_time)
            self.current_time = snapshot_time
        return self

    def top_neighbors(self, node):
        """Return the configured Top-L influences at the current time."""
        if self.current_time < 0:
            raise RuntimeError("advance_to must be called before querying")
        node = int(node)
        row = int(np.searchsorted(self._nodes, node))
        if row >= len(self._nodes) or self._nodes[row] != node:
            return ()
        length = min(int(self._state_lengths[row]), self.top_l)
        if not length:
            return ()
        scores = self._state_scores[row, :length]
        selected_total = float(scores.sum())
        if selected_total == 0.0:
            return ()
        return tuple(
            TemporalInfluence(
                node=int(self._state_nodes[row, position]),
                time=int(self._state_times[row, position]),
                score=float(scores[position]),
                weight=float(scores[position] / selected_total),
            )
            for position in range(length)
        )


class TemporalPPR:
    """Answer causal top-L T-PPR queries over an ordered snapshot sequence."""

    def __init__(self, snapshots, alpha=0.3, beta=0.5):
        if not isinstance(snapshots, Sequence):
            raise TypeError("snapshots must be an ordered sequence")
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be between 0 and 1")

        self.snapshots = snapshots
        self.alpha = float(alpha)
        self.beta = float(beta)
        history = defaultdict(list)
        snapshot_adjacency = []

        for time, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, Mapping):
                raise TypeError(f"snapshot {time} must be a mapping")
            edge_list = snapshot.get("edge_list")
            if not isinstance(edge_list, Sequence):
                raise ValueError(
                    f"snapshot {time} must contain an edge_list sequence"
                )

            seen_edges = set()
            adjacency = defaultdict(set)
            for edge in edge_list:
                if len(edge) < 2:
                    raise ValueError(
                        "each edge must contain at least two endpoints"
                    )
                u, v = edge[0], edge[1]
                if u == v:
                    continue
                edge_key = (min(u, v), max(u, v))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                history[u].append((time, v))
                history[v].append((time, u))
                adjacency[u].add(v)
                adjacency[v].add(u)
            snapshot_adjacency.append(
                {
                    node: tuple(sorted(neighbors))
                    for node, neighbors in adjacency.items()
                }
            )

        self._history = dict(history)
        self._snapshot_adjacency = snapshot_adjacency
        self._history_times = {
            node: [time for time, _ in interactions]
            for node, interactions in self._history.items()
        }

    def _transitions(self, node, cutoff):
        """Return older temporal neighbors and beta-decayed probabilities."""
        interactions = self._history.get(node, ())
        if not interactions:
            return ()

        end = bisect_left(self._history_times[node], cutoff)
        if end == 0:
            return ()

        # Equal-time interactions receive equal probability. Moving to the
        # next older time group advances the recency rank by the group size.
        weighted = []
        group_weight = 1.0
        position = end
        while position > 0:
            group_end = position
            group_time = interactions[position - 1][0]
            while position > 0 and interactions[position - 1][0] == group_time:
                position -= 1
            for time, neighbor in interactions[position:group_end]:
                weighted.append((neighbor, time, group_weight))
            # Edges in one snapshot form one recency group. Advancing to the
            # next older group applies beta once, regardless of group size.
            group_weight *= self.beta

        denominator = sum(weight for _, _, weight in weighted)
        return tuple(
            (neighbor, time, weight / denominator)
            for neighbor, time, weight in weighted
        )

    def top_neighbors(self, u, t, top_l=20, min_probability=0.0):
        """Return temporal nodes with the largest T-PPR termination scores.

        ``G_0`` through ``G_t`` are observable. Internally, the source is
        treated as ``(u, t+)`` so its first transition may use interactions in
        ``G_t``; every subsequent transition moves to a strictly smaller
        snapshot index. Returned weights normalize only the selected Top-L
        scores, as required by the attention aggregation step.
        """
        if not isinstance(t, int):
            raise TypeError("t must be an integer snapshot index")
        if t < 0 or t >= len(self.snapshots):
            raise IndexError(f"t must be in [0, {len(self.snapshots) - 1}]")
        if not isinstance(top_l, int):
            raise TypeError("top_l must be an integer")
        if top_l <= 0:
            raise ValueError("top_l must be positive")
        if min_probability < 0.0:
            raise ValueError("min_probability must be non-negative")

        # Index t + 1 is a source-only cutoff, not a returned temporal node.
        pending = [defaultdict(float) for _ in range(t + 2)]
        pending[t + 1][u] = 1.0
        scores = defaultdict(float)

        for cutoff in range(t + 1, -1, -1):
            for node, reaching_probability in pending[cutoff].items():
                is_source = cutoff == t + 1 and node == u
                if not is_source:
                    scores[(node, cutoff)] += self.alpha * reaching_probability

                continuation = (1.0 - self.alpha) * reaching_probability
                if continuation <= min_probability:
                    continue
                for neighbor, time, transition in self._transitions(node, cutoff):
                    propagated = continuation * transition
                    if propagated > min_probability:
                        pending[time][neighbor] += propagated

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                -item[0][1],
                item[0][0],
            ),
        )[:top_l]
        selected_total = sum(score for _, score in ranked)
        if selected_total == 0.0:
            return []

        return [
            TemporalInfluence(
                node=node,
                time=time,
                score=score,
                weight=score / selected_total,
            )
            for (node, time), score in ranked
        ]

    @staticmethod
    def _rank_scores(scores, limit, min_score=0.0):
        """Return deterministic highest-scoring temporal nodes."""
        return sorted(
            (
                (key, score)
                for key, score in scores.items()
                if score > min_score
            ),
            key=lambda item: (-item[1], -item[0][1], item[0][0]),
        )[:limit]

    def incremental_top_neighbors(
        self,
        queries_by_time,
        top_l=20,
        internal_top_k=20,
        min_score=0.0,
    ):
        """Yield top-L answers while scanning snapshots exactly once.

        The maintained state adapts Zebra's streaming T-PPR recurrence to
        snapshots: all edges in the same snapshot form one recency group,
        apply beta once to older groups, and read the state from before that
        snapshot. Each node retains at most ``internal_top_k`` temporal states;
        returned answers are normalized over their first ``top_l`` entries.

        Args:
            queries_by_time: Mapping from snapshot index to source node IDs.
            top_l: Number of temporal nodes returned for each query.
            internal_top_k: Width of the persistent approximate T-PPR state.
            min_score: Discard maintained aggregate scores at or below this
                value after each snapshot update.

        Yields:
            ``(node, time, influences)`` in chronological query order.
        """
        if not isinstance(queries_by_time, Mapping):
            raise TypeError("queries_by_time must be a mapping")
        if not isinstance(top_l, int) or top_l <= 0:
            raise ValueError("top_l must be a positive integer")
        if not isinstance(internal_top_k, int) or internal_top_k <= 0:
            raise ValueError("internal_top_k must be a positive integer")
        if internal_top_k < top_l:
            raise ValueError("internal_top_k must be at least top_l")
        if min_score < 0.0:
            raise ValueError("min_score must be non-negative")

        normalized_queries = {}
        for time, nodes in queries_by_time.items():
            if not isinstance(time, int):
                raise TypeError("query times must be integers")
            if time < 0 or time >= len(self.snapshots):
                raise IndexError(
                    f"query time must be in [0, {len(self.snapshots) - 1}]"
                )
            normalized_queries[time] = tuple(sorted(set(nodes)))
        if not normalized_queries:
            return

        index = self.streaming_index(
            top_l=top_l,
            internal_top_k=internal_top_k,
            min_score=min_score,
        )
        last_query_time = max(normalized_queries)

        for time in range(last_query_time + 1):
            index.advance_to(time)
            for node in normalized_queries.get(time, ()):
                yield node, time, index.top_neighbors(node)

    def streaming_index(self, top_l=20, internal_top_k=20, min_score=0.0):
        """Return a reusable causal index that can advance through time."""
        return TemporalPPRStreamingIndex(
            self,
            top_l=top_l,
            internal_top_k=internal_top_k,
            min_score=min_score,
        )

    def structure_feature(self, node, time, order, cmax):
        """Return S_time(node) using the prebuilt snapshot adjacency index."""
        return structure_representation(
            self.snapshots,
            node,
            time,
            order=order,
            cmax=cmax,
            adjacency=self._snapshot_adjacency[time],
        )

    def structural_embedding(
        self,
        u,
        t,
        top_l,
        order,
        cmax,
        transform=None,
        min_probability=0.0,
    ):
        """Aggregate selected nodes' S_tau(v) into h_(u,t).

        ``transform`` receives ``(structure_vector, time_delta)``. Omitting it
        uses the identity transform; a later trainable model can provide a
        fully connected transformation with time encoding.

        Returns:
            ``(embedding, influences)``. The embedding is an empty float64
            vector when the source has no historical interactions.
        """
        influences = self.top_neighbors(
            u,
            t,
            top_l=top_l,
            min_probability=min_probability,
        )

        def feature(node, time):
            return self.structure_feature(node, time, order, cmax)

        embedding = aggregate_temporal_features(
            influences,
            query_time=t,
            feature=feature,
            transform=transform,
        )
        return embedding, influences


def aggregate_temporal_features(
    influences,
    query_time,
    feature: Callable,
    transform: Optional[Callable] = None,
):
    """Apply normalized T-PPR weights to transformed temporal features."""
    if not influences:
        return np.empty(0, dtype=np.float64)

    result = None
    expected_shape = None
    for influence in influences:
        vector = np.asarray(
            feature(influence.node, influence.time),
            dtype=np.float64,
        )
        if transform is not None:
            vector = np.asarray(
                transform(vector, query_time - influence.time),
                dtype=np.float64,
            )
        if vector.ndim != 1:
            raise ValueError("transformed temporal features must be vectors")
        if expected_shape is None:
            expected_shape = vector.shape
            result = np.zeros(expected_shape, dtype=np.float64)
        elif vector.shape != expected_shape:
            raise ValueError("transformed temporal features must have equal widths")
        result += influence.weight * vector
    return result
