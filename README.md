# k-core Community Prediction in Temporal Networks

Given a temporal network and a query `(q, k, t)`, predict the connected k-core
component containing q in the next snapshot `G[t+1]`, using snapshots through
`G[t]`.

## Quick Start

```bash
pip install networkit numpy numba

python streaming_eval.py                     # Streaming TCS vs HCU evaluation
python streaming_eval.py --relax 5           # With dynamic relax threshold
```

## Methods

**StreamingTCS** — Time-decay weighted coreness stability scoring. It filters
unstable nodes and runs BFS on the cumulative union graph to predict q's
community.

```
TCS(v, t, k) = 1 - (1/W) Σ w_i · (k - c_i) / max(k, c_i)
```

- α = 0.7 (decay), τ = 0.15 (fixed) or dynamic relax threshold
- 70/30 train-test split, incremental ingest during evaluation

**HCU (Historical Community Union)** — Baseline that unions all historical
k-core components containing q through the current snapshot.

## Evaluation Pipeline

1. Build `slice_*.csv` files from the raw edge list with `datasets.build_time_slices`.
2. Load those slice files and construct cached k-core snapshots.
3. Initialize StreamingTCS on the first 70% snapshots.
4. For each current snapshot `G[t]`, ingest it and predict the k-core community
   containing q in `G[t+1]`.
5. Compare the predicted vertex set with the true connected k-core component;
   the main experiment evaluates samples whose true next community is non-empty.
6. Report F1, Precision, Recall, community size ratio, and prediction ratio per k.

Snapshots, test samples, and persisted evaluation sets are cached inside the
specific `time_slices/step_<step>_window_<window>/` directory that produced them.

## Data Format

CSV with a `u,v,ts` header. Each dataset keeps its source data and generated
artifacts in `data/<dataset>/`. Per-dataset time windows:

| Dataset | Window | Valid k |
|---------|--------|---------|
| email-Eu-core-temporal | week | 3-7 |
| mooc | day | 3-10 |
| sx-mathoverflow | 4week | 3-10 |
| sx-askubuntu | 4week | 3-8 |
| DBLP1 | year | 3-5 |

## Time Slices

Build sliding time slices with exactly three inputs: a dataset name, a step in
seconds, and a window length in seconds.

```bash
python -m datasets.build_time_slices email-Eu-core-temporal 604800 604800
```

This writes non-empty `slice_*.csv` files and `metadata.json` to
`data/email-Eu-core-temporal/time_slices/step_604800_window_604800/`. A slice
contains edges in `[start_ts, start_ts + window_seconds)`. When the step is
shorter than the window, an edge can appear in multiple overlapping slices.
Windows are aligned backwards from `last_timestamp + 1`, so the newest window
is complete. Leading windows that overlap the dataset are retained, so any
short observation interval occurs only at the beginning, not at the prediction
end of the timeline. Each dataset keeps one active slice configuration. A
successful rebuild replaces its previous `time_slices` directory, including
stale snapshot, sample, and evaluation caches derived from the old slices.

Build the required time slices before running the evaluation. With no slice
arguments, `streaming_eval.py` uses each dataset's configured default window as
both step and window length.

```bash
python streaming_eval.py
python streaming_eval.py --dataset mooc --step-seconds 86400 --window-seconds 172800
python streaming_eval.py --step-seconds 604800 --window-seconds 1209600
```

## Project Structure

```
streaming_eval.py               # Main entry point
methods/
  tcs_streaming.py              # StreamingTCS class
  hcu.py                        # HCU baseline
datasets/
  dataset_builder.py            # Snapshot building + caching
  community_eval_builder.py     # Community sample generation + caching
data/
  <dataset>/
    <dataset>.csv               # Raw temporal edges: u,v,ts
    time_slices/
      step_<step>_window_<window>/
        slice_*.csv             # Generated input snapshots
        metadata.json           # Slice boundaries and counts
        snapshot_cache/         # Derived k-core snapshots
        sample_cache/           # Derived test samples
        community_eval/         # Optional persisted evaluation set
eval/
  worker.py                     # Parallel evaluation workers
```
