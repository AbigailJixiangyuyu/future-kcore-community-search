"""Historical candidates and predicted k-core community recovery."""

from collections import deque
from dataclasses import dataclass

import networkit as nk
import numpy as np
from scipy import sparse


# Networkit 11.0.1 still looks up this NumPy alias when bulk-loading COO edges.
if "ulong" not in np.__dict__:
    np.ulong = np.uint64


def historical_community_union(snapshots, q, k, t):
    """Union q's connected k-core communities in snapshots 0..t."""
    if not isinstance(t, int) or t < 0 or t >= len(snapshots):
        raise IndexError("t is outside the snapshot range")
    if k <= 0:
        raise ValueError("k must be positive")
    candidate = set()
    for snapshot in snapshots[:t + 1]:
        k_info = snapshot.get("k_core_comps", {}).get(k)
        if k_info is None or q not in k_info["node_set"]:
            continue
        for component in k_info["components"]:
            if q in component:
                candidate.update(component)
                break
    return frozenset(candidate)


def component_after_peeling(nodes, edges, q, k):
    """Return q's connected component of the induced k-core graph."""
    adjacency = {node: set() for node in nodes}
    for u, v in edges:
        if u != v and u in adjacency and v in adjacency:
            adjacency[u].add(v)
            adjacency[v].add(u)
    return component_from_adjacency(adjacency, q, k)


def component_from_adjacency(adjacency, q, k):
    """Peel an undirected adjacency mapping and return q's k-core component."""
    if q not in adjacency:
        return frozenset()
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    removed = set()
    queue = deque(node for node, count in degree.items() if count < k)
    while queue:
        node = queue.popleft()
        if node in removed:
            continue
        removed.add(node)
        for neighbor in adjacency[node] - removed:
            degree[neighbor] -= 1
            if degree[neighbor] < k:
                queue.append(neighbor)
    if q in removed:
        return frozenset()
    visited = {q}
    queue = deque([q])
    while queue:
        for neighbor in adjacency[queue.popleft()] - removed - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return frozenset(visited)


@dataclass
class PredictedGraph:
    nodes: np.ndarray
    adjacency: sparse.csr_matrix

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        if self.adjacency.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("predicted adjacency shape does not match nodes")
        self.node_positions = {
            int(node): position for position, node in enumerate(self.nodes)
        }
        self._community_cache = {}

    @property
    def edge_count(self):
        return int(self.adjacency.nnz // 2)

    def community(self, candidate_nodes, q, k):
        """Return q's connected component in the induced predicted k-core."""
        candidate_nodes = sorted(set(candidate_nodes))
        if q not in candidate_nodes or k <= 0:
            return frozenset()
        cache_key = (tuple(candidate_nodes), k)
        cached = self._community_cache.get(cache_key)
        if cached is not None:
            return cached.get(q, frozenset())
        try:
            positions = np.fromiter(
                (self.node_positions[node] for node in candidate_nodes),
                dtype=np.int64,
                count=len(candidate_nodes),
            )
        except KeyError as error:
            raise ValueError(
                "candidate node is absent from the predicted graph"
            ) from error

        induced = sparse.triu(
            self.adjacency[positions][:, positions], k=1, format="coo"
        )
        graph = nk.GraphFromCoo(
            (
                induced.row.astype(np.int64, copy=False),
                induced.col.astype(np.int64, copy=False),
            ),
            n=len(candidate_nodes),
        )
        core_scores = nk.centrality.CoreDecomposition(graph).run().scores()

        communities = {}
        unseen = {
            position for position, score in enumerate(core_scores) if score >= k
        }
        while unseen:
            start = next(iter(unseen))
            visited = {start}
            unseen.remove(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in graph.iterNeighbors(node):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
            component = frozenset(
                candidate_nodes[position] for position in visited
            )
            for node in component:
                communities[node] = component
        self._community_cache[cache_key] = communities
        return communities.get(q, frozenset())
