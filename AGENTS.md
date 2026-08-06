# Project Overview

k-core community prediction in temporal networks. Given `(q, k, t)`, use
snapshots through t to predict the connected k-core component containing q in
the next snapshot.

## Tech Stack

Python 3, networkit, NumPy, Numba. Multiprocessing via `concurrent.futures`.

## Directory Structure

```
coreness-prediction/
├── streaming_eval.py              # Main entry: streaming TCS vs HCU evaluation
├── methods/
│   ├── tcs_streaming.py           # StreamingTCS class (incremental score + ingest)
│   ├── _bfs_numba.py              # Numba-accelerated BFS kernels
│   └── hcu.py                     # HCU baseline (historical community union)
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
├── eval/
│   └── worker.py                  # Parallel eval workers for streaming methods
├── docs/                          # Project documentation
└── specs/                         # Change specifications and archives
```

## Active Methods

- **StreamingTCS** (`methods/tcs_streaming.py`): Time-decay weighted coreness stability scoring + BFS on the cumulative union graph. Supports fixed τ or dynamic relax threshold. 70/30 split, incremental ingest.
- **HCU** (`methods/hcu.py`): Union of q's historical k-core communities through the current snapshot.

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
python streaming_eval.py                     # Main experiment (streaming TCS vs HCU)
python streaming_eval.py --relax 5           # Dynamic relax threshold
python -m datasets.build_time_slices <dataset> <step_seconds> <window_seconds>
python -m datasets.community_eval_builder data/<dataset>/time_slices/step_<step>_window_<window>
```
