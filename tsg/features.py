import pickle
from pathlib import Path

import networkit as nk
import numpy as np
from scipy.sparse import csr_matrix


def _compute_h_index(values):
    if len(values) == 0:
        return 0
    arr = np.sort(np.array(values))[::-1]
    h = 0
    for i, v in enumerate(arr):
        if v >= i + 1:
            h = i + 1
        else:
            break
    return int(h)


def load_snapshots(config):
    with open(config.snapshot_path, "rb") as f:
        cached = pickle.load(f)
    snapshots = cached["snapshots"]

    all_nodes = set()
    for snap in snapshots:
        for u, v, _ in snap["edge_list"]:
            all_nodes.add(u)
            all_nodes.add(v)
        for n in snap["core_dict"]:
            all_nodes.add(n)

    sorted_nodes = sorted(all_nodes)
    node_map = {orig: idx for idx, orig in enumerate(sorted_nodes)}
    num_nodes = len(node_map)

    print(f"[features] Loaded {len(snapshots)} snapshots, {num_nodes} unique nodes")
    adj_matrices = build_all_adjacency_matrices(snapshots, node_map, num_nodes)
    return snapshots, node_map, num_nodes, adj_matrices


def build_all_adjacency_matrices(snapshots, node_map, num_nodes):
    adj_matrices = []
    for snap in snapshots:
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v, _ in snap["edge_list"]:
            ui, vi = node_map[u], node_map[v]
            adj[ui, vi] = 1.0
            adj[vi, ui] = 1.0
        adj_matrices.append(adj)
    print(f"[features] Built {len(adj_matrices)} adjacency matrices ({num_nodes}x{num_nodes})")
    return adj_matrices


def compute_node_features(snapshots, adj_matrices, node_map, num_nodes, config):
    cache_dir = Path(config.feature_cache_dir)
    cache_path = cache_dir / "node_features.pkl"
    if cache_path.exists():
        print(f"[features] Loading cached node features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)

    T = len(snapshots)
    feat = np.zeros((T, num_nodes, 10), dtype=np.float32)

    prev_degree = np.zeros(num_nodes, dtype=np.float32)
    prev_core = np.zeros(num_nodes, dtype=np.float32)

    for t, snap in enumerate(snapshots):
        print(f"[features] Computing node features for snapshot {t+1}/{T}")
        edge_list = snap["edge_list"]
        core_dict = snap["core_dict"]

        present_nodes = set()
        for u, v, _ in edge_list:
            present_nodes.add(u)
            present_nodes.add(v)
        for n in core_dict:
            present_nodes.add(n)

        if not edge_list:
            prev_degree = np.zeros(num_nodes, dtype=np.float32)
            prev_core = np.zeros(num_nodes, dtype=np.float32)
            continue

        local_nodes = sorted(present_nodes)
        local_map = {orig: i for i, orig in enumerate(local_nodes)}
        n_local = len(local_nodes)

        g = nk.Graph(n_local)
        seen_edges = set()
        for u, v, _ in edge_list:
            ui, vi = local_map[u], local_map[v]
            key = (min(ui, vi), max(ui, vi))
            if key not in seen_edges:
                seen_edges.add(key)
                g.addEdge(ui, vi)

        degree_arr = np.zeros(n_local, dtype=np.int32)
        for i in range(n_local):
            degree_arr[i] = g.degree(i)

        core_arr = np.zeros(n_local, dtype=np.float32)
        for orig, c in core_dict.items():
            if orig in local_map:
                core_arr[local_map[orig]] = c

        h_values = np.zeros((config.h_index_orders + 1, n_local), dtype=np.int32)
        h_values[0] = degree_arr.copy()

        for order in range(1, config.h_index_orders + 1):
            for i in range(n_local):
                nbrs = list(g.iterNeighbors(i))
                if len(nbrs) == 0:
                    h_values[order][i] = 0
                else:
                    nbr_h = [int(h_values[order - 1][n]) for n in nbrs]
                    h_values[order][i] = _compute_h_index(nbr_h)

        lcc = nk.centrality.LocalClusteringCoefficient(g).run()
        clust_arr = np.array(lcc.scores(), dtype=np.float32)

        triangle_arr = np.zeros(n_local, dtype=np.float32)
        for i in range(n_local):
            d = degree_arr[i]
            if d >= 2:
                triangle_arr[i] = clust_arr[i] * d * (d - 1) / 2.0

        for orig in present_nodes:
            gi = local_map[orig]
            mi = node_map[orig]
            feat[t, mi, 0] = np.log1p(degree_arr[gi]).astype(np.float32)
            feat[t, mi, 1] = core_arr[gi]
            feat[t, mi, 2] = float(h_values[0][gi])
            feat[t, mi, 3] = float(h_values[1][gi])
            feat[t, mi, 4] = float(h_values[2][gi])
            feat[t, mi, 5] = float(h_values[3][gi])
            feat[t, mi, 6] = clust_arr[gi]
            feat[t, mi, 7] = np.log1p(triangle_arr[gi]).astype(np.float32)
            feat[t, mi, 8] = float(degree_arr[gi]) - prev_degree[mi]
            feat[t, mi, 9] = core_arr[gi] - prev_core[mi]

        new_prev_degree = np.zeros(num_nodes, dtype=np.float32)
        new_prev_core = np.zeros(num_nodes, dtype=np.float32)
        for orig in present_nodes:
            mi = node_map[orig]
            gi = local_map[orig]
            new_prev_degree[mi] = float(degree_arr[gi])
            new_prev_core[mi] = core_arr[gi]
        prev_degree = new_prev_degree
        prev_core = new_prev_core

    with open(cache_path, "wb") as f:
        pickle.dump(feat, f)
    print(f"[features] Cached node features to {cache_path} shape={feat.shape}")
    return feat


def build_hop_tokens(adj_matrices, node_feat, config):
    cache_dir = Path(config.feature_cache_dir)
    cache_path = cache_dir / "hop_tokens.pkl"
    if cache_path.exists():
        print(f"[features] Loading cached hop tokens from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)

    T = len(adj_matrices)
    N = adj_matrices[0].shape[0]
    K = config.K
    feat_dim = node_feat.shape[2]

    tokens = np.zeros((T, N, K + 1, feat_dim), dtype=np.float32)

    for t in range(T):
        print(f"[features] Building hop tokens for snapshot {t+1}/{T}")
        A = csr_matrix(adj_matrices[t])
        eye = csr_matrix(np.eye(N, dtype=np.float32))
        reached = eye.copy()
        A_power = eye.copy()

        for k in range(K + 1):
            if k == 0:
                exact = eye
            else:
                A_power = A_power @ A
                new_reached = reached + A_power
                exact = A_power.multiply((new_reached > 0).astype(np.float32) - (reached > 0).astype(np.float32))
                exact.eliminate_zeros()
                reached = new_reached

            for v in range(N):
                col = exact.getcol(v)
                nbrs = col.indices
                if k == 0:
                    tokens[t, v, k] = node_feat[t, v]
                elif len(nbrs) > 0:
                    tokens[t, v, k] = node_feat[t, nbrs].mean(axis=0)

    with open(cache_path, "wb") as f:
        pickle.dump(tokens, f)
    print(f"[features] Cached hop tokens to {cache_path} shape={tokens.shape}")
    return tokens


def compute_all_features(config):
    snapshots, node_map, num_nodes, adj_matrices = load_snapshots(config)
    node_feat = compute_node_features(snapshots, adj_matrices, node_map, num_nodes, config)
    hop_tokens = build_hop_tokens(adj_matrices, node_feat, config)
    return node_feat, hop_tokens, adj_matrices, node_map, num_nodes
