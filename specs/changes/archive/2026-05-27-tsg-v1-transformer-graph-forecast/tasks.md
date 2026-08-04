# Tasks: TSG V1 — Temporal Structural Transformer 图预测

## 1. 基础框架与配置

- [x] 1.1 创建 `tsg/__init__.py`（空文件）和 `tsg/config.py`，定义 `V1Config` dataclass：L=5, K=2, H_INDEX_ORDERS=3, NODE_FEAT_DIM=10, EDGE_FEAT_DIM=6, hidden_dim=64, num_heads=4, num_hop_layers=1, num_temporal_layers=1, dropout=0.1, lr=1e-3, epochs=50, dataset_name, snapshot_path, split_ratio=0.7
- [x] 1.2 创建 `run_tsg.py` 入口脚本框架：argparse（--dataset, --epochs, --lr, --hidden-dim），加载配置，调用 train/eval（先用 placeholder）

## 2. 数据加载与节点映射

- [x] 2.1 在 `tsg/features.py` 中实现 `load_snapshots(config)` → 加载 pickle 缓存的快照，收集全量节点集，建立 node_id → 连续索引 [0, 985] 的映射，返回统一节点数的快照列表
- [x] 2.2 实现 `build_adjacency_matrix(snapshot, node_map, num_nodes)` → 返回 `[N, N]` bool numpy 数组；为每个快照调用并缓存全部 76 个邻接矩阵

## 3. 节点结构特征提取

- [x] 3.1 实现 `compute_node_features(snapshots, adj_matrices, node_map, config)` → 遍历每个快照，用 networkit 计算 degree、core number、clustering coefficient、triangle count，以及 3 阶 h-index（自行迭代计算），输出 `node_feat[76][986][10]`
- [x] 3.2 加入差分特征：Δdeg 和 Δcore（与前一快照的差，第一个快照置零）；实现 pickle 缓存逻辑（`tsg_feature_cache/` 目录），第二次运行直接加载

## 4. 多跳 token 构造

- [x] 4.1 实现 `build_hop_tokens(adj_matrices, node_feat, config)` → 对每个快照每个节点，用邻接矩阵乘法或 BFS 找 1-hop 和 2-hop 邻居，mean pooling 邻居特征，空 hop 置零，输出 `hop_tokens[76][986][3][10]`；加入 pickle 缓存

## 5. 边级历史特征

- [x] 5.1 实现 `compute_edge_features_for_window(adj_matrices, node_feat, snapshots, window_start, L, num_nodes, config)` → 对历史窗口内的每个快照计算全节点对的 6 维边特征（adjacency、CN、|Δdeg|、|Δcore|、min_core、PA），返回 `[L, num_pairs, 6]`

## 6. PyTorch Dataset

- [x] 6.1 在 `tsg/dataset.py` 中实现 `GraphWindowDataset(Dataset)`：`__init__` 接收预计算的 hop_tokens、adj_matrices、node_feat、snapshots、config；`__len__` 返回窗口数；`__getitem__(idx)` 返回 `(hop_tokens_window[L,N,K+1,feat_dim], edge_feat[L,num_pairs,6], labels[num_pairs], node_mask[L,N])`

## 7. 模型定义

- [x] 7.1 在 `tsg/model.py` 中实现 `HopTransformer(nn.Module)`：线性映射 feat_dim → hidden_dim，可学习 hop PE `[K+1, hidden_dim]`，1 层 TransformerEncoder，返回 token[0]
- [x] 7.2 实现 `TemporalTransformer(nn.Module)`：可学习 time PE `[max_L, hidden_dim]`，1 层 TransformerEncoder，返回 last position
- [x] 7.3 实现 `EdgeDecoder(nn.Module)`：接收 `h_u, h_v` 批量计算 `[h_u+h_v, |h_u-h_v|, h_u⊙h_v]`，MLP 编码 flatten 边历史特征，拼接后 MLP 输出标量 score
- [x] 7.4 实现 `TSGModel(nn.Module)` 组合以上三个模块，`forward` 接收 hop_tokens + edge_feat + node_mask，返回全节点对的 score 向量

## 8. 训练循环

- [x] 8.1 在 `tsg/train.py` 中实现 `train_epoch(model, dataset, optimizer, device, config)`：遍历训练窗口，BCEWithLogitsLoss（带 pos_weight），梯度裁剪，返回平均 loss
- [x] 8.2 实现 `train(model, train_dataset, test_dataset, device, config)`：完整训练循环，每个 epoch 评估测试集，打印 epoch loss + 指标，返回最优模型

## 9. 评估与推理

- [x] 9.1 在 `tsg/evaluate.py` 中实现 `top_m_infer(model, hop_tokens, edge_feat, node_mask, target_num_edges, device)` → 模型前向 + 取 top-M 分数的边索引
- [x] 9.2 实现 `compute_metrics(pred_edges, true_edges, all_scores, all_labels, num_nodes)` → 计算 Precision@M、Recall@M、ROC-AUC、AP、Edge Jaccard、Degree MAE

## 10. 入口整合与端到端测试

- [x] 10.1 完善 `run_tsg.py`：串联数据加载 → 特征预计算 → Dataset 构建 → 模型创建 → 训练 → 评估 → 打印结果表格
- [x] 10.2 在 email-Eu-core-temporal 上端到端运行，确认无报错，输出合理的指标数值
