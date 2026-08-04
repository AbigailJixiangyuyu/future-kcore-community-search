# Project Overview

k-core community prediction in temporal networks. Given node q and threshold k, predict which vertices belong to the same k-core community as q in the next snapshot.

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
│   ├── community_eval_builder.py  # Test sample generation + caching
│   ├── snapshot_cache/            # Cached snapshots (auto-generated)
│   ├── sample_cache/              # Cached test samples (auto-generated)
│   └── *.csv                      # Data files (gitignored)
├── eval/
│   └── worker.py                  # Parallel eval workers for streaming methods
├── docs/                          # Project documentation
└── specs/                         # Change specifications and archives
```

## Active Methods

- **StreamingTCS** (`methods/tcs_streaming.py`): Time-decay weighted coreness stability scoring + BFS on cumulative union graph. Supports fixed τ or dynamic relax threshold. 70/30 split, incremental ingest.
- **HCU** (`methods/hcu.py`): Union of q's historical k-core communities across all past snapshots.

## TCS Formula

```
TCS(v,t,k) = 1 - (1/W) Σ w_i · (k - c_i) / max(k, c_i)
```
α=0.7, LOOKBACK=5. Fixed τ=0.15 or dynamic relax threshold.

## Data

CSV format `u,v,ts`. Per-dataset config in `datasets/dataset_builder.py`:

| Dataset | Window | Valid k |
|---------|--------|---------|
| email-Eu-core-temporal | week | 3-7 |
| mooc | day | 3-10 |
| DBLP1 | year | 3-5 |
| sx-mathoverflow | 4week | 3-10 |
| sx-askubuntu | 4week | 3-8 |

## Commands

```bash
python streaming_eval.py                     # Main experiment (streaming TCS vs HCU)
python streaming_eval.py --relax 5           # Dynamic relax threshold
python -m datasets.community_eval_builder datasets/<csv>  # Rebuild eval cache
```

