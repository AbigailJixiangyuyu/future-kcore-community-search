# Project Overview

k-core community prediction in temporal networks. Given `(q, k, t)`, use
snapshots through t to predict the connected k-core component containing q in
the next snapshot.

## Tech Stack

Python 3, networkit, NumPy, PyTorch. Numba and multiprocessing are retained only
by the archived StreamingTCS experiment.

## Directory Structure

```
coreness-prediction/
├── train_coreness.py              # Hybrid coreness model training
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
├── docs/                          # Project documentation
└── specs/                         # Change specifications and archives
```

## Active Methods

- **Hybrid coreness prediction** (`methods/hybrid_coreness.py`): Predict next-snapshot node coreness, filter by k, peel the cumulative graph, and return q's connected component.
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
python zebra_community.py query 413 7 52 --device cuda:0
python -m datasets.build_time_slices <dataset> <step_seconds> <window_seconds>
python -m datasets.community_eval_builder data/<dataset>/time_slices/step_<step>_window_<window>
```
