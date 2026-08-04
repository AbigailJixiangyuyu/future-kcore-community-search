import numpy as np
from numba import njit


@njit
def _bfs_csr_numba(adj_ptr, adj_nbr, adj_edge_k, k, S_k, threshold, start_node, n):
    visited = np.zeros(n, dtype=np.bool_)
    visited[start_node] = True
    Q = np.empty(n, dtype=np.int32)
    Q[0] = start_node
    head = 0
    tail = 1
    count = 0
    while head < tail:
        v = Q[head]
        head += 1
        count += 1
        s = adj_ptr[v]
        e = adj_ptr[v + 1]
        for i in range(s, e):
            if adj_edge_k[i] < k-1:
                continue
            u = adj_nbr[i]
            if not visited[u] and S_k[u] <= threshold:
                visited[u] = True
                Q[tail] = u
                tail += 1
    return Q[:count].copy()


@njit
def _merge_sorted_dedup(old_pairs, new_pairs, out):
    n_old = old_pairs.shape[0]
    n_new = new_pairs.shape[0]
    i = 0
    j = 0
    k = 0
    while i < n_old and j < n_new:
        a0 = old_pairs[i, 0]
        a1 = old_pairs[i, 1]
        b0 = new_pairs[j, 0]
        b1 = new_pairs[j, 1]
        if a0 < b0 or (a0 == b0 and a1 < b1):
            out[k, 0] = a0
            out[k, 1] = a1
            out[k, 2] = old_pairs[i, 2]
            k += 1
            i += 1
        elif a0 > b0 or (a0 == b0 and a1 > b1):
            out[k, 0] = b0
            out[k, 1] = b1
            out[k, 2] = new_pairs[j, 2]
            k += 1
            j += 1
        else:
            out[k, 0] = a0
            out[k, 1] = a1
            out[k, 2] = max(old_pairs[i, 2], new_pairs[j, 2])
            k += 1
            i += 1
            j += 1
    while i < n_old:
        out[k, 0] = old_pairs[i, 0]
        out[k, 1] = old_pairs[i, 1]
        out[k, 2] = old_pairs[i, 2]
        k += 1
        i += 1
    while j < n_new:
        out[k, 0] = new_pairs[j, 0]
        out[k, 1] = new_pairs[j, 1]
        out[k, 2] = new_pairs[j, 2]
        k += 1
        j += 1
    return k
