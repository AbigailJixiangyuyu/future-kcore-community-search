# Project Overview

k-core community prediction in temporal networks. Given `(q, k, t)`, use
snapshots through t to predict the connected k-core component containing q in
the next snapshot.

## Tech Stack

Python 3, networkit, NumPy, PyTorch. Numba accelerates the active T-PPR
implementation and is also used by the archived StreamingTCS experiment.
Multiprocessing is retained by the archived StreamingTCS evaluator.

## Directory Structure

```
coreness-prediction/
├── train_coreness.py              # Hybrid coreness model training
├── hybrid_community.py            # On-demand coreness BFS prediction
├── zebra_community.py             # Zebra link-based community prediction
├── methods/
│   ├── hybrid_coreness.py          # Coreness model + community recovery
│   ├── tcs_representation.py       # Node-level temporal features
│   └── hcu.py                      # HCU baseline (historical community union)
├── datasets/
│   ├── dataset_builder.py         # Snapshot building + caching (networkit-based)
│   ├── community_eval_builder.py  # Community test sample generation + caching
├── data/
│   └── <dataset>/                 # Raw CSV and all generated data artifacts
│       ├── <dataset>.csv          # Raw edges: u,v,ts
│       └── time_slices/
│           └── step_<step>_window_<window>/
│               ├── slice_*.csv   # Generated sliding-window edge slices
│               ├── snapshot_cache/  # Cached k-core snapshots
│               ├── sample_cache/    # Cached test samples
│               └── community_eval/  # Persisted evaluation set (optional)
├── archive/streaming_tcs/         # Retired StreamingTCS code and evaluator
├── archive/analysis/             # Historical standalone analysis tools
├── archive/specs/                # Historical change specifications (archived)
└── docs/                          # Project documentation
```

## Active Methods

- **Hybrid coreness prediction** (`hybrid_community.py`): Traverse the cumulative graph from q, predict each new BFS frontier in batches, and expand only through nodes whose predicted next-snapshot coreness is at least k. Queries at the same time share node-level TCS, T-PPR influence, structure-feature, and predicted-coreness caches, so each touched node is featurized and inferred at most once per time slice. No k-core peeling is applied.
- **Zebra community prediction** (`zebra_community.py`): Predict links over the historical community candidate set, run k-core decomposition, and return q's connected component.
- **HCU** (`methods/hcu.py`): Union of q's historical k-core communities through the current snapshot.

StreamingTCS is retired. Its implementation and old comparison evaluator are
available only under `archive/streaming_tcs/` for reproducibility.

## TCS Formula

```
TCS(v,t,k) = 1 - (1/W) Σ w_i · (k - c_i) / max(k, c_i)
```
α=0.7, LOOKBACK=5. Fixed τ=0.15 or dynamic relax threshold.

## Data

CSV format `u,v,ts`. All subsequent sample generation, evaluation, and result
reporting must use `k in [3, 4, 5, 6, 7]`. Do not test or report `k >= 8`
unless the user explicitly requests a different range.

Required evaluation scope for subsequent work:

| Dataset | Window | Valid k |
|---------|--------|---------|
| email-Eu-core-temporal | week | 3-7 |
| mooc | day | 3-7 |
| DBLP1 | year | 3-7 |
| sx-mathoverflow | 4week | 3-7 |
| sx-askubuntu | 4week | 3-7 |

## Commands

```bash
python train_coreness.py data/mooc/time_slices/step_43200_window_86400
python hybrid_community.py query 413 7 52 --device cuda:0
python zebra_community.py query 413 7 52 --device cuda:0
python -m datasets.build_time_slices <dataset> <step_seconds> <window_seconds>
python -m datasets.community_eval_builder data/<dataset>/time_slices/step_<step>_window_<window>
```
