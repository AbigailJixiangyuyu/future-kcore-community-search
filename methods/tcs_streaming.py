import time
import numpy as np
from collections import deque
from methods._bfs_numba import _bfs_csr_numba, _merge_sorted_dedup


class StreamingTCS:
    def __init__(self, snaps, valid_ks, alpha=0.7, tau=0.15, total_nodes=0, relax=None):
        self.valid_ks = valid_ks
        self.alpha = alpha
        self.tau = tau
        self._relax = relax
        self.current_t = len(snaps) - 1

        self.time_ingest_total = 0.0
        self.time_ingest_score = 0.0
        self.time_ingest_adj = 0.0
        self.time_ingest_ug_copy = 0.0
        self.time_ingest_threshold = 0.0
        self.time_predict_total = 0.0
        self.time_predict_bfs = 0.0
        self._predict_count = 0
        self._ingest_count = 0

        self.node_idx = {}
        self.node_list = np.zeros(total_nodes, dtype=np.int64)
        idx = 0
        for snap in snaps:
            for v in snap["core_dict"]:
                if v not in self.node_idx:
                    self.node_idx[v] = idx
                    self.node_list[idx] = v
                    idx += 1
        self._n = total_nodes
        self._used = idx

        self._S = {}
        self._W = 0.0

        for k in valid_ks:
            self._S[k] = np.zeros(total_nodes, dtype=np.float64)

        for snap in snaps:
            cd = snap["core_dict"]
            for k in valid_ks:
                k_f = float(k)
                S_k = self._S[k]
                S_k *= alpha
                S_k += 1.0
                for v, c in cd.items():
                    idx_v = self.node_idx[v]
                    c_f = float(c)
                    denom = max(k_f, c_f)
                    actual = (k_f - c_f) / denom if denom > 0 else 1.0
                    S_k[idx_v] += actual - 1.0
            self._W = self._W * alpha + 1.0

        self._edge_pairs = self._collect_edge_pairs(snaps)
        self._build_csr()

        _d1 = np.array([[0, 1, 3]], dtype=np.int32)
        _d2 = np.array([[0, 2, 3]], dtype=np.int32)
        _d3 = np.empty((2, 3), dtype=np.int32)
        _merge_sorted_dedup(_d1, _d2, _d3)

        _ap = np.array([0, 1, 2], dtype=np.int64)
        _an = np.array([1, 0], dtype=np.int32)
        _aek = np.array([3, 3], dtype=np.int32)
        _sk = np.array([0.0, 0.0], dtype=np.float64)
        _bfs_csr_numba(_ap, _an, _aek, np.int32(3), _sk, np.float64(0.5), np.int32(0), np.int32(2))

        if self._relax is not None:
            self._threshold = self._compute_threshold(snaps[-1])
        else:
            self._threshold = self._W * (1.0 - tau)

    def _collect_edge_pairs(self, snaps):
        pairs = []
        node_idx = self.node_idx
        for snap in snaps:
            for u, v, ek in snap["edge_list"]:
                ui, vi = node_idx[u], node_idx[v]
                pairs.append((ui, vi, ek))
                pairs.append((vi, ui, ek))
        if not pairs:
            return np.empty((0, 3), dtype=np.int32)
        raw = np.array(pairs, dtype=np.int32)
        order = np.lexsort((-raw[:, 2], raw[:, 1], raw[:, 0]))
        sorted_raw = raw[order]
        mask = np.ones(len(sorted_raw), dtype=np.bool_)
        mask[1:] = (sorted_raw[1:, :2] != sorted_raw[:-1, :2]).any(axis=1)
        return sorted_raw[mask]

    def _build_csr(self):
        pairs = self._edge_pairs
        n = self._n
        if len(pairs) == 0:
            self.adj_ptr = np.zeros(n + 1, dtype=np.int64)
            self.adj_nbr = np.empty(0, dtype=np.int32)
            self.adj_edge_k = np.empty(0, dtype=np.int32)
            return
        degrees = np.bincount(pairs[:, 0], minlength=n)
        self.adj_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(degrees, out=self.adj_ptr[1:])
        self.adj_nbr = pairs[:, 1].copy()
        self.adj_edge_k = pairs[:, 2].copy()

    def _compute_threshold(self, snap):
        thresholds = []
        cd = snap["core_dict"]
        for k in self.valid_ks:
            k_core_size = sum(1 for c in cd.values() if c >= k)
            if k_core_size == 0:
                continue
            ratio = min(1.0, self._relax * k_core_size / self._n)
            sorted_S = np.sort(self._S[k][:self._used])
            n_pad = self._n - len(sorted_S)
            if n_pad > 0:
                sorted_S = np.concatenate([sorted_S, np.full(n_pad, self._W)])
            idx = min(int(ratio * len(sorted_S)), len(sorted_S) - 1)
            thresholds.append(sorted_S[idx])
        return sum(thresholds) / len(thresholds) if thresholds else self._W * 0.99

    def predict(self, t, q, k):
        t0 = time.time()
        qi = self.node_idx.get(q)
        if qi is None or self.adj_ptr[qi] == self.adj_ptr[qi + 1]:
            self.time_predict_total += time.time() - t0
            self._predict_count += 1
            return frozenset(), True
        S_k = self._S.get(k)
        if S_k is None:
            self.time_predict_total += time.time() - t0
            self._predict_count += 1
            return frozenset(), True
        if S_k[qi] > self._threshold:
            self.time_predict_total += time.time() - t0
            self._predict_count += 1
            return frozenset(), True
        t1 = time.time()
        indices = _bfs_csr_numba(self.adj_ptr, self.adj_nbr, self.adj_edge_k,
                                  np.int32(k), S_k,
                                  self._threshold, qi, self._n)
        result_indices = self.node_list[indices]
        self.time_predict_bfs += time.time() - t1
        self.time_predict_total += time.time() - t0
        self._predict_count += 1
        return frozenset(result_indices), False

    def ingest(self, snap):
        t0 = time.time()
        self._ingest_count += 1
        self.current_t += 1

        new_nodes = [v for v in snap["core_dict"] if v not in self.node_idx]
        if new_nodes:
            start = self._used
            for v in new_nodes:
                self.node_idx[v] = self._used
                self.node_list[self._used] = v
                self._used += 1
            for k in self.valid_ks:
                self._S[k][start:self._used] = self._W

        t_score = time.time()
        cd = snap["core_dict"]
        for k in self.valid_ks:
            k_f = float(k)
            S_k = self._S[k]
            S_k *= self.alpha
            S_k += 1.0
            for v, c in cd.items():
                idx_v = self.node_idx[v]
                c_f = float(c)
                denom = max(k_f, c_f)
                actual = (k_f - c_f) / denom if denom > 0 else 1.0
                S_k[idx_v] += actual - 1.0

        self._W = self._W * self.alpha + 1.0
        self.time_ingest_score += time.time() - t_score

        t_adj = time.time()
        node_idx = self.node_idx
        new_pairs = []
        for u, v, ek in snap["edge_list"]:
            ui, vi = node_idx[u], node_idx[v]
            new_pairs.append((ui, vi, ek))
            new_pairs.append((vi, ui, ek))
        if new_pairs:
            new_arr = np.array(new_pairs, dtype=np.int32)
            order = np.lexsort((-new_arr[:, 2], new_arr[:, 1], new_arr[:, 0]))
            new_sorted = new_arr[order]
            mask = np.ones(len(new_sorted), dtype=np.bool_)
            mask[1:] = (new_sorted[1:, :2] != new_sorted[:-1, :2]).any(axis=1)
            new_sorted = new_sorted[mask]
            if len(self._edge_pairs) == 0:
                self._edge_pairs = new_sorted
            else:
                n_old = len(self._edge_pairs)
                n_new = len(new_sorted)
                buf = np.empty((n_old + n_new, 3), dtype=np.int32)
                count = _merge_sorted_dedup(self._edge_pairs, new_sorted, buf)
                self._edge_pairs = buf[:count].copy()
            self._build_csr()
        self.time_ingest_adj += time.time() - t_adj

        t_thr = time.time()
        if self._relax is not None:
            self._threshold = self._compute_threshold(snap)
        else:
            self._threshold = self._W * (1.0 - self.tau)
        self.time_ingest_threshold += time.time() - t_thr

        self.time_ingest_total += time.time() - t0

    def profile_summary(self):
        return {
            "ingest_calls": self._ingest_count,
            "ingest_total_s": self.time_ingest_total,
            "ingest_score_s": self.time_ingest_score,
            "ingest_adj_s": self.time_ingest_adj,
            "ingest_ug_copy_s": self.time_ingest_ug_copy,
            "ingest_threshold_s": self.time_ingest_threshold,
            "predict_calls": self._predict_count,
            "predict_total_s": self.time_predict_total,
            "predict_bfs_s": self.time_predict_bfs,
            "predict_overhead_s": self.time_predict_total - self.time_predict_bfs,
        }
