# 历史专项分析

`analyze_historical_edge_coverage.py` 于 2026-09-16 从 `scripts/` 迁入，
用于统计真实未来社区内部边被历史边覆盖的比例，不参与当前训练和预测。
相关单元测试仍保留在 `tests/test_historical_edge_coverage.py`。

从项目根目录运行帮助：

```bash
python -m archive.analysis.analyze_historical_edge_coverage --help
```

实际分析依赖原有快照、样本缓存和相邻目录中的 Zebra 数据。
历史结论见 [覆盖率报告](../../docs/archive/historical-edge-coverage-20260908.md)。
