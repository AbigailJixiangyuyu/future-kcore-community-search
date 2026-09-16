# Proposal: CSR + Numba BFS 加速

## 为什么要做这个变更

DBLP 数据集上 `streaming_eval.py` 总耗时 919s，其中：

- **TCS predict BFS 占 63% wall time**（worker CPU 累计 3043s）。BFS 在 1.82M 节点、8.3M 边的累积图上用 Python deque + dict-of-list 遍历，单次 BFS 数百万次 Python 级别的 dict lookup。
- **`ug_adj` 深拷贝占 8.5%**（78s）。每次 ingest 将整个累积邻接表 dict 深拷贝一份，7 次 ingest 每次 ~11s。

当前邻接表用 `dict[原始ID → set[原始ID]]` 存储，内存 ~350MB，且每次 ingest 需要 `{node: list(nb)}` 全量拷贝。

## 变更内容

分两阶段对 `methods/tcs_streaming.py` 的数据结构和 BFS 内核进行加速：

**阶段 A — CSR 邻接表**：将累积邻接表从 `dict-of-set` 替换为 CSR（Compressed Sparse Row）格式（两个 numpy 数组 `adj_ptr` + `adj_nbr`），消除 dict lookup 和 ug_adj 深拷贝。

**阶段 B — Numba JIT BFS**：在 CSR 基础上，用 `@numba.njit` 将 BFS 内循环编译为原生代码，进一步消除 Python 解释器开销。

## 影响范围

- `methods/tcs_streaming.py`：主要改动文件（数据结构、`__init__`、`ingest`、`predict`）
- 新增一个 Numba JIT BFS 函数（可放在 `tcs_streaming.py` 内或独立模块）
- `streaming_eval.py`：无需改动（接口不变）
- `eval/worker.py`：无需改动（通过 fork 继承 TCS 对象）
- 运行时依赖新增：numba（已安装 0.65.0）
