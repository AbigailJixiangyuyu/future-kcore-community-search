# Proposal: TSG V1 — Temporal Structural Transformer 图预测

## 为什么要做这个变更

当前项目只有基于规则的预测方法（TCS scoring + BFS pruning、HCU baseline），没有学习能力。根据 `docs/temporal_structural_transformer_graph_forecasting_doc.md` 的技术方案，需要实现一个基于 Transformer 的动态图下一快照结构预测模型，为后续 k-core 社区预测提供基础边生成器。

## 变更内容

实现文档中定义的 Version 1 最小可运行版本，仅使用 email-Eu-core-temporal 数据集：

- 新建 `tsg/` 包，包含配置、特征工程、数据集、模型、训练、评估六个模块
- 新建 `run_tsg.py` 入口脚本
- 模型流水线：节点结构特征提取 → 多跳 token 构造 → Hop-wise Structural Transformer → Temporal Transformer → Pair-wise Edge Decoder
- 训练：BCEWithLogitsLoss，AdamW 优化器，滑动窗口枚举全节点对
- 推理：Oracle-M Top-M 边选择
- 评估：Precision@M、Recall@M、ROC-AUC、AP、Edge Jaccard、Degree MAE

## 影响范围

- 新增 `tsg/` 包（6 个模块 + `__init__.py`）
- 新增 `run_tsg.py` 入口
- 不修改任何现有文件（`datasets/dataset_builder.py`、`methods/`、`streaming_eval.py` 等保持不变）
- 运行时依赖新增：PyTorch（已安装 2.0.1+cu118）
- 可选新增 `requirements.txt` 记录 PyTorch 依赖
