"""Generate edges with immediate effective-core dependency propagation."""

import heapq
from dataclasses import dataclass
from numbers import Integral


@dataclass
class PredictedEdgeResult:
    """BFS nodes, stable generated edges, and event-based propagation metrics."""

    nodes: tuple
    edges: tuple
    h_max: dict
    effective_coreness: dict
    node_update_count: int
    reprocessed_node_count: int
    core_updates: tuple
    edge_case_counts: tuple

    @property
    def edge_case_ratios(self):
        return tuple(count / len(self.edges) if self.edges else None
                     for count in self.edge_case_counts)

    def edge_metrics(self):
        """JSON-ready final-edge counts and fractions (undefined for no edges)."""
        return {
            **{f"edge_case{i}_count": count
               for i, count in enumerate(self.edge_case_counts, 1)},
            **{f"edge_case{i}_ratio": ratio
               for i, ratio in enumerate(self.edge_case_ratios, 1)},
        }

    def core_reduction_metrics(self):
        """Case-3 unique lowered nodes divided by all selected BFS nodes."""
        count = len({node for node, _, _ in self.core_updates})
        return {
            "case3_lowered_node_count": count,
            "case3_lowered_node_ratio": count / len(self.nodes) if self.nodes else None,
        }


class _LazySecondScores:
    """Query-local two-hop scores, materialized only for requested sources."""

    def __init__(self, direct):
        self.direct = direct
        self.ordered = {}
        self.cache = {}

    def _ordered_row(self, node):
        if node not in self.ordered:
            # Preserve the eager implementation's floating-point sum order.
            self.ordered[node] = tuple(sorted(self.direct[node].items()))
        return self.ordered[node]

    def __getitem__(self, node):
        if node not in self.cache:
            paths = {}
            for middle, first_score in self._ordered_row(node):
                for candidate, score in self._ordered_row(middle):
                    if candidate != node:
                        paths[candidate] = paths.get(candidate, 0.0) + first_score * score
            self.cache[node] = paths
        return self.cache[node]


class _ReverseCandidateDependencies:
    """Expand one/two-hop reverse dependencies on demand, without score rows."""

    def __init__(self, direct):
        self.reverse = {v: set() for v in direct}
        for source, candidates in direct.items():
            for candidate in candidates:
                self.reverse[candidate].add(source)

    def __getitem__(self, node):
        affected = set(self.reverse[node])
        for middle in self.reverse[node]:
            affected.update(self.reverse[middle])
        affected.discard(node)
        return affected


def generate_predicted_edges(nodes, predicted_coreness, influences_for_node, k,
                             *, scores_for_node=None):
    """Use cached raw T-PPR candidates inside the fixed BFS node set.

    Temporal scores for a vertex are summed; two-hop scores sum path products.
    Original predictions are copied, never modified. When a core decreases,
    already-visited affected nodes are revalidated before first-time visits
    continue. Unvisited nodes simply see the latest state when first visited.
    Optional scores_for_node supplies already-merged raw score dictionaries;
    these are read-only and still filtered by this query's node set.
    """
    if not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    nodes = set(nodes)
    for node in nodes:
        value = predicted_coreness[node]
        if not isinstance(value, Integral) or value < 0:
            raise ValueError("predicted coreness must be a nonnegative integer")

    direct = {}
    for node in sorted(nodes):
        if scores_for_node is not None:
            direct[node] = {
                candidate: score for candidate, score in scores_for_node(node).items()
                if candidate in nodes and candidate != node
            }
            continue
        scores = {}
        for influence in influences_for_node(node):
            candidate = influence.node
            if candidate in nodes and candidate != node:
                scores[candidate] = scores.get(candidate, 0.0) + influence.score
        direct[node] = scores
    second = _LazySecondScores(direct)
    engine = _ImmediateEdgePropagation(
        {node: int(predicted_coreness[node]) for node in nodes}, direct, second, k
    )
    return engine.run()


class _ImmediateEdgePropagation:
    """Priority worklist with directed selection ownership for an undirected graph.

    Effective cores only decrease. With cores fixed, normal-branch selections
    are fixed and deficient branches retain valid selections and only add edges.
    This avoids mutual withdrawal/readdition of valid edges and terminates.
    """

    def __init__(self, effective, direct, second, k):
        self.effective = dict(effective)
        self.direct = direct
        self.second = second
        self.k = k
        self.selected = {v: set() for v in effective}
        self.owners = {v: set() for v in effective}
        self.adjacency = {v: set() for v in effective}
        self.edge_cases = {}
        if isinstance(second, _LazySecondScores):
            self.dependents = _ReverseCandidateDependencies(direct)
        else:
            # Explicit candidate tables used by low-level callers may differ
            # from the two-hop closure; preserve their exact dependencies.
            self.dependents = {v: set() for v in effective}
            for source in effective:
                for candidate in set(direct[source]) | set(second[source]):
                    self.dependents[candidate].add(source)
        self.rankings = {}
        for source in effective:
            self.rankings[source] = (
                sorted(direct[source], key=lambda v: (-direct[source][v], v)),
                None,
            )
        self.visited = set()
        self.process_counts = {v: 0 for v in effective}
        self.core_updates = []
        self.heap = []
        self.queued = {}
        self.sequence = 0
        for node in sorted(effective):
            self._enqueue(node, urgent=False)

    def _second_ranking(self, node):
        """Sort only when case 3 reaches two-hop candidates; reuse on revisits."""
        direct, second = self.rankings[node]
        if second is None:
            scores = self.second[node]
            second = sorted(scores, key=lambda v: (-scores[v], v))
            self.rankings[node] = (direct, second)
        return second

    def _enqueue(self, node, urgent):
        priority = 0 if urgent else 1
        key = (priority, -self.effective[node], node)
        previous = self.queued.get(node)
        if previous is not None and previous[:3] == key:
            return
        if previous is not None and previous[0] < priority:
            return
        self.sequence += 1
        entry = key + (self.sequence,)
        self.queued[node] = entry
        heapq.heappush(self.heap, entry)

    def _revalidate(self, nodes):
        for node in sorted(set(nodes) & self.visited):
            self._enqueue(node, urgent=True)

    def _add_selection(self, source, target, case):
        if target not in self.adjacency[source]:
            self.edge_cases[tuple(sorted((source, target)))] = case
        self.selected[source].add(target)
        self.owners[target].add(source)
        self.adjacency[source].add(target)
        self.adjacency[target].add(source)

    def _remove_selection(self, source, target):
        self.selected[source].remove(target)
        self.owners[target].remove(source)
        # Keep the undirected edge if the opposite endpoint still selected it.
        if source not in self.selected[target]:
            self.adjacency[source].remove(target)
            self.adjacency[target].remove(source)
            del self.edge_cases[tuple(sorted((source, target)))]
            self._revalidate((source, target))

    def _capacity(self, node):
        capacity = 0
        for rank, value in enumerate(sorted(
            (self.effective[v] for v in self.direct[node]), reverse=True
        ), 1):
            if value < rank:
                break
            capacity = rank
        return capacity

    def _support_count(self, node):
        target = self.effective[node]
        return sum(self.effective[v] >= target for v in self.adjacency[node])

    def _lower(self, node):
        previous = self.effective[node]
        self.effective[node] = previous - 1  # Visible immediately, not at round end.
        self.core_updates.append((node, previous, previous - 1))
        affected = self.dependents[node] | self.adjacency[node] | {node}
        # Withdraw obsolete selections before any other vertex is processed.
        for owner in sorted(self.owners[node]):
            if self.effective[owner] > self.effective[node]:
                self._remove_selection(owner, node)
        self._revalidate(affected)

    def _process_node(self, node):
        self.visited.add(node)
        self.process_counts[node] += 1
        core = self.effective[node]
        if core == 0:
            return
        capacity = self._capacity(node)
        if capacity >= core:
            eligible = [v for v in self.rankings[node][0]
                        if self.effective[v] >= core]
            desired = set(eligible[:core] if capacity > core else eligible)
            for target in sorted(self.selected[node] - desired):
                self._remove_selection(node, target)
            for target in sorted(desired - self.selected[node]):
                self._add_selection(node, target, 2 if capacity > core else 1)
            return

        # Case 3: preserve valid existing choices; do not withdraw merely because
        # another endpoint now provides sufficient support.
        # No other node is processed during this call. Each added edge below is
        # new and eligible, so its contribution is exactly one until we yield.
        support = self._support_count(node)
        for order in range(2):
            if support >= core:
                return
            ranked = (self.rankings[node][0] if order == 0
                      else self._second_ranking(node))
            for candidate in ranked:
                if support >= core:
                    return
                if (candidate not in self.adjacency[node]
                        and self.effective[candidate] >= core):
                    self._add_selection(node, candidate, 3)
                    support += 1
        if support < core and core > self.k:
            # Yield after each decrease so previously processed affected nodes
            # can react before unrelated, never-visited nodes are processed.
            self._lower(node)

    def run(self):
        while self.heap:
            entry = heapq.heappop(self.heap)
            node = entry[2]
            if self.queued.get(node) != entry:
                continue
            del self.queued[node]
            self._process_node(node)
        return PredictedEdgeResult(
            nodes=tuple(sorted(self.effective)),
            edges=tuple(sorted(
                (u, v) for u, neighbors in self.adjacency.items()
                for v in neighbors if u < v
            )),
            h_max={v: self._capacity(v) for v in sorted(self.effective)},
            effective_coreness=dict(self.effective),
            node_update_count=sum(self.process_counts.values()),
            reprocessed_node_count=sum(max(0, count - 1)
                                       for count in self.process_counts.values()),
            core_updates=tuple(self.core_updates),
            edge_case_counts=tuple(
                sum(case == i for case in self.edge_cases.values())
                for i in (1, 2, 3)
            ),
        )
