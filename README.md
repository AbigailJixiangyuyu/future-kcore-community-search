# k-core Community Prediction in Temporal Networks

Given a temporal network, a query node q and a threshold k, predict which vertices belong to the same k-core community as q in the next time snapshot.

## Quick Start

```bash
pip install networkit numpy

python streaming_eval.py                     # Streaming TCS vs HCU evaluation
python streaming_eval.py --relax 5           # With dynamic relax threshold
```

## Methods

**StreamingTCS** — Time-decay weighted coreness stability scoring. Incrementally maintains per-node scores and predicts communities via BFS on the cumulative union graph.

```
TCS(v, t, k) = 1 - (1/W) Σ w_i · (k - c_i) / max(k, c_i)
```

- α = 0.7 (decay), τ = 0.15 (fixed) or dynamic relax threshold
- 70/30 train-test split, incremental ingest during evaluation

**HCU (Historical Community Union)** — Baseline that unions q's k-core communities across all past snapshots.

## Evaluation Pipeline

1. Load edge list (`u, v, ts`) and build time-windowed snapshots (cached to `datasets/snapshot_cache/`)
2. Initialize StreamingTCS on first 70% snapshots
3. On each test snapshot: ingest new edges → predict communities for sampled (q, k) queries → evaluate against ground truth
4. Report F1, Precision, Recall, Size Ratio, Prediction Ratio per k

Test samples are generated with coreness-weighted sampling (seed=42) and cached to `datasets/sample_cache/`.

## Data Format

CSV with columns `u, v, ts` (no header). Per-dataset time windows:

| Dataset | Window | Valid k |
|---------|--------|---------|
| email-Eu-core-temporal | week | 3-7 |
| mooc | day | 3-10 |
| sx-mathoverflow | 4week | 3-10 |
| sx-askubuntu | 4week | 3-8 |
| DBLP1 | year | 3-5 |

## Project Structure

```
streaming_eval.py               # Main entry point
methods/
  tcs_streaming.py              # StreamingTCS class
  hcu.py                        # HCU baseline
datasets/
  dataset_builder.py            # Snapshot building + caching
  community_eval_builder.py     # Test sample generation + caching
  snapshot_cache/               # Auto-generated snapshot cache
  sample_cache/                 # Auto-generated sample cache
eval/
  worker.py                     # Parallel evaluation workers
```
