# Future k-core Community Search on Temporal Graphs

Given a query `(q, k, t)`, predict the community containing node `q` in
snapshot `t+1` using only snapshots through `t`. Snapshot indices are zero-based;
community evaluation uses `k=3..7`.

## Methods

- **Ours:** Predict next-snapshot node coreness on demand, select a community by
  BFS over the cumulative historical graph, then generate edges among the
  selected nodes. Edge generation does not perform a final k-core decomposition.
- **Zebra:** Predict links within a historical community candidate set, perform
  k-core decomposition, and return the component containing `q`.
- **EAGLE, SWIFT, TFWaveFormer, PRISM:** Additional link-prediction baselines
  with local adapters and source code under `third_party/`.

Community entry points live in `community/`:
`community/ours.py` selects BFS nodes,
`community/generated_edges.py` returns nodes and edges, and
`community/baselines/` contains the five link-prediction baselines.
Its `baseline_graph.py` holds shared candidate and community recovery code;
Zebra's runtime, progressive runner, and result storage live in separate
`zebra_*.py` modules behind the unchanged `community.baselines.zebra` command.
`training/` holds our training commands, `methods/` holds prediction/model
adapters, and `third_party/` contains upstream baseline source. Batch Ours
evaluation uses the complete node-and-edge pipeline.

## Setup And Data

Use Python 3 with PyTorch, NumPy, Networkit, and Numba:

```bash
python -m pip install torch numpy networkit numba
```

Baseline models have additional dependencies and, for SWIFT, a local CUDA/DGL
build; see `third_party/README.md`. Datasets, model checkpoints, and native
build outputs are not included in Git.

Place timestamp-sorted, integer `u,v,ts` records in
`data/<dataset>/<dataset>.csv`, with that header. Create indexed time windows
before training or evaluating:

```bash
python -m datasets.build_time_slices mooc 43200 86400
```

This command **rebuilds** the dataset's time-slice configuration and its
derived caches. Existing configurations can instead be migrated without
rebuilding caches using `python -u -m datasets.migrate_slice_storage`.
Snapshots and evaluation samples are cached under
`data/<dataset>/time_slices/step_<step>_window_<window>/` on first use.

| Dataset | Window | Evaluation k |
| --- | --- | --- |
| email-Eu-core-temporal | week | 3-7 |
| mooc | day | 3-7 |
| DBLP1 | year | 3-7 |
| sx-mathoverflow | 4 weeks | 3-7 |
| sx-askubuntu | 4 weeks | 3-7 |

## Train And Predict

Train a coreness model for a prepared slice configuration:

```bash
python -m training.ours data/mooc/time_slices/step_43200_window_86400
```

The default Ours checkpoint is saved under that slice directory's
`model_cache/`. Pass `--checkpoint` to use a different compatible checkpoint.
For CPU inference, replace `cuda:0` with `cpu`.

```bash
# Node selection only
python -m community.ours query 413 7 52 --device cuda:0

# Complete Ours community: selected nodes and generated edges
python -m community.generated_edges 413 7 52 \
  --slices-dir data/mooc/time_slices/step_43200_window_86400 \
  --checkpoint data/mooc/time_slices/step_43200_window_86400/model_cache/hybrid_coreness.pt \
  --device cuda:0

# Zebra (requires a locally trained Zebra checkpoint and prepared Zebra data)
python -m community.baselines.zebra query 413 7 52 --device cuda:0
```

Ours caches compatible historical streaming state automatically at the first
requested time. For other datasets, supply `--slices-dir` and a matching
checkpoint; Zebra also accepts `--zebra-dataset` and `--checkpoint`.

## Evaluate

```bash
python -m community.ours eval --device cuda:0 \
  --output outputs/ours_mooc.json
python -m community.baselines.zebra eval --device cuda:0 \
  --output outputs/zebra_mooc.json
```

Both evaluators report per-k and macro node precision, recall, F1 and Jaccard.
Ours also reports edge metrics, generated-edge case ratios, and the fraction
of nodes whose effective coreness was lowered. Edge ground truth is restricted
to edges inside the true next-snapshot community. `prepare_s` is time-slice
preparation; `query_s` is per-query prediction, including edge generation for
Ours. Ground-truth metric computation is excluded from both.

Training and evaluation require locally prepared data and compatible
checkpoints. Baseline training entry points are
`training.baselines.{prism,tfwaveformer}`; TFWaveFormer single-edge scoring
is `scripts.tfwaveformer_predict`. Other community baseline entry points are
`community.baselines.{eagle,swift,tfwaveformer,prism}` (run with `python -m`);
their source integration and build requirements are described in
`third_party/README.md`.
