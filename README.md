# k-core Community Prediction in Temporal Networks

Given a temporal network and a query `(q, k, t)`, predict the connected k-core
component containing q in the next snapshot `G[t+1]`, using snapshots through
`G[t]`.

## Quick Start

```bash
pip install networkit numpy torch

python train_coreness.py data/mooc/time_slices/step_43200_window_86400
python zebra_community.py query 413 7 52 --device cuda:0
```

### Zebra link-based community prediction

`zebra_community.py` keeps the community target unchanged while using Zebra to
predict the next graph. Given `(q, k, t)`, it unions q's connected k-core
communities in snapshots `0..t`, scores every unordered pair in that candidate
set at `t+1`, retains edges whose bidirectional mean probability is strictly
greater than `0.5`, and returns q's connected component after k-core
decomposition.

```bash
# t is the zero-based coreness snapshot index
python zebra_community.py query 413 7 52 --device cuda:0

# Non-empty MOOC ground-truth samples in the leakage-free Zebra test period
python zebra_community.py eval --device cuda:0 \
  --output outputs/zebra_mooc_community.json
```

The default MOOC paths use the converted `mooc-snapshot` data and its trained
checkpoint in the sibling `Zebra` repository. The evaluation start is derived
from Zebra's 85% time boundary; for the current 60-snapshot MOOC data it is
`t=52`, predicting snapshot 53 (Zebra timestamp 54). Predicted sparse graphs
are cached in `.zebra_cache/` using the checkpoint, mapping files, threshold,
time, and node set as the cache identity.

## Methods

**Hybrid coreness prediction** — Predicts every observable node's coreness in
the next snapshot. Community recovery retains nodes with predicted coreness at
least `k`, performs k-core peeling on the cumulative historical graph, and
returns the surviving connected component containing q.

**Zebra link prediction** — Predicts the next graph over q's historical
community candidate set, then performs k-core decomposition and returns q's
connected component.

**HCU (Historical Community Union)** — Baseline that unions all historical
k-core components containing q through the current snapshot.

The former **StreamingTCS** community search method is archived under
`archive/streaming_tcs/` and is not part of the active evaluation pipeline.

For node-level temporal features, `methods.tcs_representation` provides
`tcs_representation(snapshots, u, t, kmax)`. Snapshot preprocessing determines
and caches the dataset-level `kmax`; passing that fixed value keeps every
node's `[TCS(1), ..., TCS(kmax)]` representation the same width. The function
uses only snapshots through `t` and performs no neighbor aggregation.

For current-snapshot structural features, `methods.h_index_representation`
provides `structure_representation(snapshots, u, t, order, cmax)`. Snapshot
preprocessing caches h-index orders 1 through 3 and determines the dataset-level
maximum first-order h-index `hmax`, which is used as the fixed `cmax`. The
representation concatenates the
closed-neighborhood distributions for orders `1..order-1` and current
coreness, using buckets `0..hmax-1` plus a final `>=hmax` bucket. Because
`hmax` is the global maximum, the last bucket represents the exact maximum;
its width is `order * (hmax + 1)` for `order` in `1..4`.

`methods.t_ppr.TemporalPPR` selects influential historical time-nodes using an
inverse-time random walk with termination probability `alpha` and recency
decay `beta`. Each newer active snapshot group applies `beta` once to older
interactions, regardless of the number of edges in that group. Feature building
scans the snapshots once and incrementally maintains each node's internal Top-K
T-PPR state. All edges in one snapshot are updated simultaneously, so equal-time
results do not depend on CSV edge order.
The default internal width is 80, while the model receives normalized Top-L
attention weights with `L=20`. `top_neighbors(u, t, top_l)` remains available
as a full per-query reference implementation.

`train_coreness.py` builds causal `(u, t) -> coreness(u, t+1)` samples with a
70/15/15 chronological split and trains `HybridCorenessPredictor`. The model
applies a trainable time encoder and shared structural transformation before
T-PPR weighted aggregation, fuses the result with TCS, and predicts one of the
fixed classes `0..kmax`. Feature tensors and the final checkpoint are cached
inside the selected time-slice directory's `model_cache/`. When a per-time
sample limit is used, nodes are stratified by their observable current
coreness, including a group for historically seen but currently absent nodes;
future labels are never used for sampling.

For inference, `build_prediction_samples` and `prepare_feature_arrays` create
all observable node inputs at a query time, `predict_coreness_map` produces the
next-snapshot coreness mapping, and `community_from_predicted_coreness` applies
the `k` threshold, k-core peeling, and connected-component search containing
`q`.

## Community Recovery

1. Build `slice_*.csv` files from the raw edge list with `datasets.build_time_slices`.
2. Load those slice files and construct cached k-core snapshots.
3. Use either the hybrid model to predict next-snapshot node coreness or Zebra
   to predict next-snapshot links.
4. Recover the connected k-core component containing q from the corresponding
   predicted structure.
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
| mooc | day | 3-7 |
| sx-mathoverflow | 4week | 3-7 |
| sx-askubuntu | 4week | 3-7 |
| DBLP1 | year | 3-7 |
| wiki-talk-temporal | 7day | 3-7 |

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

Build the required time slices before training or evaluation.

```bash
python -m datasets.build_time_slices mooc 43200 86400
python train_coreness.py data/mooc/time_slices/step_43200_window_86400
python zebra_community.py eval --device cuda:0 \
  --output outputs/zebra_mooc_community.json
```

## Project Structure

```
train_coreness.py               # Hybrid coreness model training entry
zebra_community.py              # Zebra link-based community prediction
methods/
  tcs_representation.py         # Fixed-width node temporal features
  h_index_representation.py     # Multi-order structural features
  t_ppr.py                      # Temporal-neighbor influence queries
  hybrid_coreness.py            # Trainable fusion model + community recovery
  hcu.py                        # Historical community union baseline
datasets/
  dataset_builder.py            # Snapshot building + caching
  coreness_prediction_builder.py # Node samples and model feature tensors
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
archive/
  streaming_tcs/                # Retired StreamingTCS implementation and evaluator
```
