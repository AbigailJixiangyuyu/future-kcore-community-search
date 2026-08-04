# Proposal: Edge Max-K 属性与 BFS 剪枝

## 为什么要做这个变更

当前 BFS 遍历累积并图时仅按节点 TCS 分数剪枝（`S_k[u] <= threshold`），不考虑边本身是否属于 k-core 子图。大量 k-core 以外的边被无谓遍历，浪费 BFS 搜索空间。

为每条边标注其最大所属 k-core（`edge_k = min(coreness(u), coreness(v))`），可在 BFS 中额外剪枝：跳过 `edge_k < k` 的边，显著减少搜索范围。

## 变更内容

1. **快照层**：`snapshot.edge_list` 从 `[(u, v)]` 扩展为 `[(u, v, edge_k)]`，其中 `edge_k = min(core_dict[u], core_dict[v])`
2. **CSR 层**：`_edge_pairs` 从 `(E, 2)` 扩展为 `(E, 3)`，新增 `adj_edge_k` 数组与 `adj_nbr` 平行
3. **合并策略**：重复边取 `max(edge_k)`（乐观策略，保留边的历史最强 k-core 归属）
4. **BFS 剪枝**：`_bfs_csr_numba` 新增 `adj_edge_k` 和 `k` 参数，遍历边时跳过 `edge_k < k` 的边

## 影响范围

- `datasets/dataset_builder.py`：edge_list 结构变更（**需清除 snapshot_cache**）
- `datasets/community_eval_builder.py`：解包 edge_list 适配 3-tuple
- `methods/_bfs_numba.py`：`_merge_sorted_dedup` 改 3 列 + max 去重；`_bfs_csr_numba` 加边剪枝
- `methods/tcs_streaming.py`：`_collect_edge_pairs`、`_build_csr`、`ingest`、`predict` 均需适配 3 列
- **F1 结果会变化**：edge-k 剪枝是额外过滤条件，recall 可能下降、precision 可能上升
