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

from methods.h_index_representation import structure_representation


@dataclass(frozen=True)
class TemporalInfluence:
    """One selected temporal node and its T-PPR attention weight."""

    node: int
    time: int
    score: float
    weight: float


class TemporalPPRStreamingIndex:
    """Maintain approximate T-PPR state while advancing through snapshots."""

    def __init__(self, temporal_ppr, top_l=20, internal_top_k=80,
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
        self._norms = {}
        self._states = {}

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
            adjacency = self.temporal_ppr._snapshot_adjacency[snapshot_time]
            updates = {}
            norm_updates = {}

            # All right-hand sides read the state from the preceding snapshot.
            for node, neighbors in adjacency.items():
                degree = len(neighbors)
                decay = self.temporal_ppr.beta
                old_norm = self._norms.get(node, 0.0)
                new_norm = decay * old_norm + degree
                candidates = defaultdict(float)

                if old_norm > 0.0:
                    old_scale = decay * old_norm / new_norm
                    for key, score in self._states.get(node, {}).items():
                        candidates[key] += old_scale * score

                neighbor_scale = (1.0 - self.temporal_ppr.alpha) / new_norm
                terminal_score = neighbor_scale * self.temporal_ppr.alpha
                for neighbor in neighbors:
                    candidates[(neighbor, snapshot_time)] += terminal_score
                    for key, score in self._states.get(neighbor, {}).items():
                        candidates[key] += neighbor_scale * score

                updates[node] = dict(self.temporal_ppr._rank_scores(
                    candidates,
                    self.internal_top_k,
                    min_score=self.min_score,
                ))
                norm_updates[node] = new_norm

            self._states.update(updates)
            self._norms.update(norm_updates)
            self.current_time = snapshot_time
        return self

    def top_neighbors(self, node):
        """Return the configured Top-L influences at the current time."""
        if self.current_time < 0:
            raise RuntimeError("advance_to must be called before querying")
        ranked = self.temporal_ppr._rank_scores(
            self._states.get(node, {}),
            self.top_l,
            min_score=self.min_score,
        )
        selected_total = sum(score for _, score in ranked)
        if selected_total == 0.0:
            return []
        return [
            TemporalInfluence(
                node=target_node,
                time=target_time,
                score=score,
                weight=score / selected_total,
            )
            for (target_node, target_time), score in ranked
        ]


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
        internal_top_k=80,
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

    def streaming_index(self, top_l=20, internal_top_k=80, min_score=0.0):
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
