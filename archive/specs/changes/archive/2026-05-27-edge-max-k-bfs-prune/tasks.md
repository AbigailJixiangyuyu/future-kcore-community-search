# Tasks: Edge Max-K 属性与 BFS 剪枝

## 1. 快照层：edge_list 添加 edge_k 属性

- [x] 1.1 修改 `datasets/dataset_builder.py` 第 228 行：在 `filtered_edges` 和 `core_dict` 都就绪后，计算 `edge_k = min(core_dict[u], core_dict[v])`，将 `edge_list` 从 `[(u, v)]` 改为 `[(u, v, edge_k)]`
- [x] 1.2 修改 `datasets/community_eval_builder.py` 第 183 行：`for u, v in ...` 改为 `for u, v, _ in ...`

## 2. Numba 内核：3 列合并 + BFS 边剪枝

- [x] 2.1 重写 `_merge_sorted_dedup`（`methods/_bfs_numba.py`）：3 列合并，按 (col0, col1) 比较，重复时取 max(col2)
- [x] 2.2 修改 `_bfs_csr_numba`（`methods/_bfs_numba.py`）：新增 `adj_edge_k`（int32 数组）和 `k`（int32）参数，遍历边时增加 `adj_edge_k[i] >= k` 条件

## 3. StreamingTCS 适配 3 列数据流

- [x] 3.1 修改 `_collect_edge_pairs`（`methods/tcs_streaming.py`）：解包 `(u, v, edge_k)`，生成 `(E, 3)` 数组；lexsort 改为 `(-col2, col1, col0)` 使 edge_k 大者优先
- [x] 3.2 修改 `_build_csr`（`methods/tcs_streaming.py`）：新增 `self.adj_edge_k = pairs[:, 2].copy()`
- [x] 3.3 修改 `ingest`（`methods/tcs_streaming.py`）：解包 `(u, v, edge_k)`，`new_arr` 改 3 列，merge buffer 改 `(n_old+n_new, 3)`，lexsort/dedup 同步适配
- [x] 3.4 修改 `predict`（`methods/tcs_streaming.py`）：调用 `_bfs_csr_numba` 时传入 `self.adj_edge_k` 和 `np.int32(k)`
- [x] 3.5 更新 `__init__` 中 numba warmup 调用（`methods/tcs_streaming.py`）：dummy 数组改为 3 列

## 4. 缓存清理与验证

- [x] 4.1 清除 `datasets/snapshot_cache/` 目录下所有缓存文件
- [x] 4.2 运行 `python streaming_eval.py`（单数据集如 mooc）确认程序不报错、F1 值合理
- [x] 4.3 对比 edge-k 剪枝前后的 F1/Precision/Recall 变化，确认符合预期
