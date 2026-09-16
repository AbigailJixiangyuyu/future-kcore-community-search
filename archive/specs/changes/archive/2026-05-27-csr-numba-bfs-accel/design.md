# Design: CSR + Numba BFS 加速

## 背景

当前 `StreamingTCS` 的邻接表用 `dict[原始节点ID → set[原始节点ID]]` 存储，predict 时 BFS 需要对每条边做 2 次 dict lookup（邻居查找 + node_idx 映射）。累积图有 1.82M 节点、8.3M 边，单次 BFS 遍历数十万节点，dict 开销巨大。

## 方案决策

### 阶段 A：CSR 邻接表

**数据结构**：用两个 numpy 数组替代 dict-of-set：

- `adj_ptr: np.ndarray[int64]`，形状 `(n+1,)`，`adj_ptr[i]` 到 `adj_ptr[i+1]` 是节点 i 的邻居在 `adj_nbr` 中的范围
- `adj_nbr: np.ndarray[int32]`，形状 `(2E,)`，存放所有邻居的整数索引

**关键约束**：

- 所有节点和邻居统一用整数索引（`node_idx` 映射在构建 CSR 时完成，predict 中不再需要 dict lookup）
- CSR 在 `__init__` 中构建一次，`ingest` 时增量追加（预分配足够空间或动态扩容）
- `ug_adj` 不再需要按时间戳深拷贝，改为维护单一 CSR，predict 直接使用当前版本。如需时间旅行（predict 指向更早时间戳），暂不支持或回退到旧逻辑——当前所有 predict 调用都指向 `current_t`，无需历史快照。

**ingest 增量更新**：将新边追加到 `adj_nbr` 末尾，更新对应节点的 `adj_ptr` 偏移。为避免频繁重分配，初始预留 2 倍边数空间。

**预期收益**：

- BFS 加速 ~3-6x（消除 dict lookup）
- ug_copy 78s → ~0s
- 内存 ~350MB → ~78MB

### 阶段 B：Numba JIT BFS

**实现**：写一个 `@numba.njit` 函数，输入 CSR 数组 + S_k + threshold + start_node，返回访问到的节点索引数组。

**约束**：

- 依赖阶段 A 的 CSR 数据结构
- Numba 不支持 Python dict/list 对象，所有输入必须是 numpy 数组或标量
- 首次调用有 ~1-2s JIT 编译开销（可用 `cache=True` 缓存）
- 返回 typed list，需转为 Python frozenset（在调用方处理）

**预期收益**：BFS 在阶段 A 基础上再加速 ~10-30x。

## 风险与权衡

1. **CSR 预分配 vs 动态扩容**：预分配过大浪费内存，过小需重分配。选择初始 2 倍当前边数，ingest 时若超出则 1.5x 扩容。
2. **Numba 兼容性**：Numba 0.65.0 对 `np.bool_` 数组支持良好，无已知问题。`cache=True` 可避免每次运行重编译。
3. **接口兼容**：`predict()` 签名和返回值不变，上层无需改动。`ingest()` 内部变更，外部调用不变。
