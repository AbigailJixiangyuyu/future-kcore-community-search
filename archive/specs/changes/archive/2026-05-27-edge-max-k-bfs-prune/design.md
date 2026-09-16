# Design: Edge Max-K 属性与 BFS 剪枝

## 背景

当前 BFS 在累积并图上搜索时，仅依据节点 TCS 分数决定是否访问邻居。但 k-core 社区预测的核心假设是：k-core 子图内的边比子图外的边更有预测价值。一条 `edge_k = 2` 的边不可能属于 k=5 的 k-core 子图，无论其端点的 TCS 分数如何。

## 方案决策

### edge_k 定义

`edge_k(u, v) = min(core_dict[u], core_dict[v])`，取自每条边所在快照的 `core_dict`。这是边能属于的最大 k-core 值的精确上界。

### 重复边合并：取 max(edge_k)

同一条边可能出现在多个快照中，各快照 coreness 不同导致 edge_k 不同。选择 **max** 而非 latest：
- 乐观策略：如果边曾经在某个快照中属于更高 k-core，保留这个信息
- 实现简单：sort 时以 `-edge_k` 作为 tiebreaker，dedup 保留第一行即最大值
- 与"latest wins"相比，避免了需要增量合并才能保证语义的问题

### 数据流变更

```
snapshot.edge_list: [(u,v)] → [(u,v,edge_k)]
_edge_pairs:        (E,2)    → (E,3)  [src, dst, edge_k]
CSR:                          adj_ptr + adj_nbr + adj_edge_k (新增)
BFS signature:                新增 adj_edge_k, k 参数
```

### _merge_sorted_dedup 改造

3 列合并，按 (col0, col1) 比较：
- `old < new`：输出 old 行
- `old > new`：输出 new 行
- `old == new`：输出 `max(old.col2, new.col2)` 所在行

### _collect_edge_pairs 改造

保持一次性 sort+dedup，无需增量合并：
1. 收集所有 (src, dst, edge_k) 三元组
2. `np.lexsort((-col2, col1, col0))`：按 (col0, col1) 升序，col2 降序
3. dedup 保留每组第一行（edge_k 最大的）

## 风险与权衡

1. **F1 变化**：edge-k 剪枝会跳过原来能访问到的某些路径，recall 可能下降、precision 可能上升。用户已确认接受。
2. **Snapshot cache 失效**：`edge_list` 结构从 2-tuple 变 3-tuple，需清除 `datasets/snapshot_cache/`。Sample cache 不受影响（不依赖 edge_list）。
3. **内存微增**：`_edge_pairs` 和 `adj_edge_k` 各增加 ~4 bytes/边。DBLP1 ~16.7M directed pairs，增加 ~67MB，可接受。
