# Design: TSG V1 — Temporal Structural Transformer 图预测

## 背景

email-Eu-core-temporal 数据集：986 个节点，76 个周快照。每快照节点集合不同（52~667），需固定为 986 节点集合并用 mask 处理不活跃节点。全节点对约 485K，小图可枚举。

## 方案决策

### 数据：固定节点集 + 预计算特征

- 所有快照统一映射到 986 节点空间，不活跃节点的边为空、特征置零
- 节点结构特征（degree、core、h-index 3 阶、clustering、triangle、Δdeg、Δcore）用 networkit 预计算并 pickle 缓存
- 多跳 token（K=2）用 BFS 预计算并缓存
- 邻接矩阵 `[N, N]` bool 数组预计算（~73MB），用于边特征在线计算
- 稀疏快照（snapshot 10/62/75）保留不做特殊处理，依赖模型学习

### 模型架构

**Hop-wise Structural Transformer**：
- 线性映射 feat_dim(10) → hidden_dim(64) + 可学习 hop PE
- 1 层 TransformerEncoder，4 heads，取 token[0] 作为 `z_v^τ`

**Temporal Transformer**：
- 可学习 time PE
- 1 层 TransformerEncoder，4 heads，取 last position 作为 `h_v^t`

**Edge Decoder**：
- 节点表示：`h_u + h_v`, `|h_u - h_v|`, `h_u ⊙ h_v` — 矩阵运算批量计算
- 边历史特征：MLP 编码 flatten 的 `[L × 8]` 特征（adjacency、CN、Jaccard、PA、AA、|Δdeg|、|Δcore|、min_core）
- 拼接后 MLP → 标量 score

**V1 边特征简化**：AA 和 Jaccard 计算成本较高（需共同邻居），V1 先用 adjacency、CN、|Δdeg|、|Δcore|、min_core、PA 六个特征，后续版本再加 AA/Jaccard。

### 训练

- 70/30 split：前 53 个快照训练（~48 个窗口），后 23 个测试
- BCEWithLogitsLoss，`pos_weight = num_neg / num_pos` 处理类别不平衡
- AdamW，lr=1e-3，50 epochs
- 每个 epoch 遍历所有训练窗口（无 mini-batch，窗口级 full-batch）

### 评估

- Oracle-M Top-M 推理
- 指标：Precision@M = Recall@M、ROC-AUC、AP、Edge Jaccard、Degree MAE

### 文件结构

```
tsg/__init__.py          # 包标记
tsg/config.py            # V1Config dataclass
tsg/features.py          # compute_all_features() → 缓存 + 加载
tsg/dataset.py           # GraphWindowDataset(Dataset)
tsg/model.py             # TSGModel(nn.Module)
tsg/train.py             # train() + evaluate()
tsg/evaluate.py          # top_m_infer() + compute_metrics()
run_tsg.py               # 入口：parse args → train → eval → report
```

## 风险与权衡

1. **全节点对枚举**：986 个节点 → 485K pairs × 8 特征 × L=5 = ~19M 浮点数/窗口。内存可接受但训练较慢。后续可引入负采样加速。
2. **特征预计算耗时**：986 节点 × 76 快照 × h-index 3 阶 + clustering + triangles，预估 2-5 分钟一次性计算。用 pickle 缓存避免重复。
3. **PyTorch 版本**：2.0.1 支持 `nn.TransformerEncoder`，无兼容风险。
4. **不修改现有代码**：完全独立的 `tsg/` 包，零侵入。
