# Tasks: CSR + Numba BFS 加速

## 1. CSR 数据结构构建

- [x] 1.1 在 `StreamingTCS.__init__` 中，将 `_cumulative_adj` 和 `ug_adj` 的构建替换为 CSR 数组（`adj_ptr` + `adj_nbr`），初始化时从所有 init snapshots 的边构建
- [x] 1.2 移除 `ug_adj` 按时间戳深拷贝逻辑，改为单一 CSR 结构；清理 `_cumulative_adj` dict

## 2. CSR 增量更新

- [x] 2.1 重写 `ingest` 方法：新边追加到 `adj_nbr`，更新 `adj_ptr` 对应位置；实现预分配 + 必要时 1.5x 扩容
- [x] 2.2 移除 `ingest` 中的 `ug_adj[t] = {n_: list(nb) ...}` 深拷贝，确认 `ug_adj` 字段不再使用

## 3. CSR-based BFS（纯 Python）

- [x] 3.1 重写 `predict` 方法中的 BFS：用 `adj_ptr[v]:adj_ptr[v+1]` 切片获取邻居，邻居已是整数索引无需 dict lookup
- [x] 3.2 移除 BFS 内循环中的 `node_idx[u]` 调用；保留 `visited` 和 `S_k` 的 numpy 数组访问

## 4. Numba JIT BFS

- [x] 4.1 编写 `_bfs_csr_numba` 函数（`@njit(cache=True)`），输入 `adj_ptr, adj_nbr, S_k, threshold, start_node`，使用 numba typed list 做 BFS，返回访问节点数组
- [x] 4.2 在 `predict` 中调用 `_bfs_csr_numba`，将返回数组转为 frozenset（需将整数索引映射回原始节点 ID）

## 5. 验证

- [x] 5.1 运行 `python streaming_eval.py` 对比 CSR-only 和 CSR+Numba 两个版本的 PROFILING REPORT，确认加速效果
- [x] 5.2 确认 F1/Precision/Recall 指标与优化前完全一致（数值不变）
