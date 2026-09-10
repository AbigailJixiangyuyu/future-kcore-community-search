# k-core Community Prediction in Temporal Networks

Given a temporal network and a query `(q, k, t)`, predict the connected k-core
component containing q in the next snapshot `G[t+1]`, using snapshots through
`G[t]`.

## Quick Start

```bash
pip install networkit numba numpy torch

python train_coreness.py data/mooc/time_slices/step_43200_window_86400
python hybrid_community.py query 413 7 52 --device cuda:0
python zebra_community.py query 413 7 52 --device cuda:0
```

Hybrid streaming state can be checkpointed before evaluation without
materializing model features. The default cache time is the 70% evaluation
boundary; later `query` and `eval` commands automatically load the newest
compatible cache at or before their query time.

```bash
python hybrid_community.py build-state-cache \
  --slices-dir data/wiki-talk-temporal/time_slices/step_259200_window_604800 \
  --checkpoint data/wiki-talk-temporal/time_slices/step_259200_window_604800/model_cache/hybrid_coreness.pt \
  --device cpu --time 343
```

### Coreness-guided edge generation

The independent `generated_edge_community.py` script first selects nodes by
q-rooted BFS through nodes with predicted coreness >= k on the full historical
graph. It then generates undirected edges only within that selected node set,
using reverse h-index and raw T-PPR scores. Each source's edge candidates are
the distinct vertices in its cached T-PPR Top-L records intersected with the
BFS set, excluding self; historical direct adjacency is not required.
Scores are summed across temporal records; uncached vertices are excluded.
For capacity-deficient sources, immediately when each source is visited, the generator
fills toward effective coreness using direct candidates and then two-hop
candidates (sum of raw-score products across paths). Effective cores start at
model predictions and only decrease, down to k. Each decrease is immediately
visible. Already-visited affected sources are revalidated before first-time
visits continue; unvisited sources simply read the latest values on first visit.
Initial visits and the priority revalidation queue each use descending current
effective core, breaking ties by node ID. Only neighbors with effective core
>= the target count as support. Directed selection ownership preserves an
undirected edge if either endpoint still selects it; invalid selections are
withdrawn immediately and lost support triggers revalidation.
The queue runs to exhaustion, without full-graph rounds or edge rebuilding.
Original model predictions stay unchanged. Valid deficit selections are retained
to avoid mutual withdrawal oscillations. If still insufficient at k, the partial
result is kept, with no degree/connectivity guarantee. JSON reports final
effective cores, node-processing/reprocessing counts, and individual core updates.
It returns these nodes and edges
directly, without k-core peeling, further connectivity filtering, or removing
isolates. Edge generation reuses the BFS predictions, not global inference.
It does not change the existing hybrid threshold-BFS command.
See [edge generation rules](docs/生成边.md).

```bash
python generated_edge_community.py 413 7 52 \
  --slices-dir data/mooc/time_slices/step_43200_window_86400 \
  --checkpoint results/fusion_ablation_20260907/mooc/concat.pt \
  --device cuda:0 \
  --output results/generated_edges/mooc_q413_k7_t52.json
```

Arguments are `q k t`, with zero-based `t` predicting snapshot `t+1`, and
`k` restricted to 3–7. An explicit compatible checkpoint is required.
JSON includes the full community node list, generated edges and graph sizes;
missing queries or queries with predicted coreness < k return an empty list. Output paths
must not already exist. This is a single-query prediction, not an evaluation
against future ground truth.

### Zebra link-based community prediction

`zebra_community.py` keeps the community target unchanged while using Zebra to
predict the next graph. Given `(q, k, t)`, it unions q's connected k-core
communities in snapshots `0..t`, scores only deduplicated historical edges
whose endpoints are in that candidate set at `t+1`, retains edges whose bidirectional mean probability is strictly
greater than `0.5`, and returns q's connected component after k-core
decomposition.

There is no candidate-history window or rho option in the active implementation.
An edge is eligible if it appeared anywhere in snapshots `0..t`, not necessarily
inside one of q's historical communities. Self-loops and repeated undirected
edges are excluded. First appearances after t cannot be scored. This reduces
decoder work but cannot recover never-before-seen future edges.
Queries at the same time still share node encoding and historical-edge scores.
Progressive evaluation reports historical candidate-edge counts, not all-pairs
counts. Result JSON identifies the full-history, historical-edges-only policy.

The previous sliding-window/all-pairs implementation is preserved unchanged as
`zebra_community_rho_backup.py`. Run it with the same arguments as before,
including `--candidate-window-rho 0.2`, to reproduce the rho experiments.
The historical email rho runner uses this backup, not the active implementation.

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
are cached in `.zebra_cache/` using the policy version, historical-edge index,
checkpoint, configuration, mapping files, threshold, time, and node set as
the cache identity. Previous all-pairs graph caches and progressive results
are not reused; existing checkpoints and experiment artifacts are preserved.

## Methods

**Hybrid coreness prediction** — Runs a layered BFS from q over the cumulative
historical graph. Each layer predicts next-snapshot coreness only for newly
reached nodes; nodes predicted at least `k` join the result and expose the next
layer. No additional k-core peeling is applied.

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
encodes the five most recent coreness values with a trainable token table and
GRU. The output head is an independent single linear layer. Bucket embeddings
and set attention encode the Top-L structural histories; Q-guided attention
uses T-PPR weights only as a score prior. Residual fusion predicts a correction to the
current-coreness persistence baseline. Class logits are converted to monotone
cumulative probabilities and trained with one mildly reweighted ordinal binary
cross-entropy objective; coreness is decoded as the number of passed thresholds.
Feature tensors and the final checkpoint are cached
inside the selected time-slice directory's `model_cache/`. When a per-time
sample limit is used, nodes are stratified by their observable current
coreness, including a group for historically seen but currently absent nodes;
future labels are never used for sampling.

The current architecture is documented in detail in
[`docs/current-coreness-model-architecture.md`](docs/current-coreness-model-architecture.md).

Structure pooling now uses only path B: set attention followed by
query-conditioned pooling and LayerNorm. T-PPR weights remain an attention
score prior. Path A and the pooling CLI selector have been removed.
Only checkpoints explicitly marked with `structure_pooling="b"` are compatible
with this pooling design. The historical node-only comparison is documented in
[`structure pooling results`](docs/archive/model-experiments/structure-pooling-ablation-results.md).

Fusion now uses only `[Q,S]`: `Linear(128,128)` followed by two ordinary
residual MLP blocks and the 32-dimensional state projection. Four-term
interaction fusion and its selector have been removed. Existing concat MLP
checkpoints remain compatible; four-term checkpoints are rejected.

The learned Lag table has been removed. History encoding now uses only a
no-Lag GRU; the experimental RoPE Transformer and its CLI options have been
removed. Transformer checkpoints are explicitly rejected. Missing coreness
history now uses token 0, with no separate
ABSENT embedding. Checkpoints containing Lag weights or a separate ABSENT
row are rejected and require retraining rather than silently changing predictions.
Always use a separate `--output` for experimental models.

The GRU receives only trainable coreness embeddings, with missing history
mapped to coreness 0. The unsuccessful structure-history input experiment and
its CLI flag have been removed; its
[results](docs/archive/model-experiments/structure-history-ablation-results.md) remain as a historical record.

See [history encoder results](docs/archive/model-experiments/history-encoder-ablation-results.md)
for the historical encoder comparison protocol and results; its runner has
been removed. The completed
[`Lag ablation`](docs/archive/model-experiments/lag-ablation-results.md) is retained as an experiment record.

The only output head is `Linear(32, kmax+1)` after
the existing 128-to-32 state projection. Its weights are initially copied from
the input coreness table, but are separate parameters and train independently.
The tied decoder, output-head selector and dedicated comparison scripts have
been removed. Checkpoints must explicitly declare `output_head_type="linear"`;
tied checkpoints and missing output-head metadata are rejected. Existing default
checkpoint files are not automatically replaced. Compatible local models are
`results/fusion_ablation_20260907/{email,mooc}/concat.pt`.
See [historical output-head results](docs/archive/model-experiments/output-head-ablation-results.md);
archived commands describe retired versions, not the current code.

For inference, `hybrid_community.py` builds one cumulative adjacency and T-PPR
state through the query time. It predicts `q` first, then deduplicates each BFS
frontier, creates features only for those nodes, and sends them to the model in
bounded batches. At one query time, TCS vectors, T-PPR Top-L influences,
structure features, and predicted coreness values are cached by node and shared
across all `(q, k)` queries. A node is therefore featurized and inferred at most
once per time slice while BFS still avoids untouched nodes. The CLI reports both
examined nodes and new model predictions.

## Community Recovery

1. Build `slice_*.csv` files from the raw edge list with `datasets.build_time_slices`.
2. Load those slice files and construct cached k-core snapshots.
3. Use either the hybrid model to predict next-snapshot node coreness on demand
   during BFS or Zebra to predict next-snapshot links.
4. Hybrid returns q's threshold-connected BFS region; Zebra recovers q's
   connected k-core component from its predicted graph.
5. Compare the predicted vertex set with the true connected k-core component;
   the main experiment evaluates samples whose true next community is non-empty.
6. Report F1, Precision, Recall, community size ratio, and prediction ratio per k.

Snapshots, test samples, and persisted evaluation sets are cached inside the
specific `time_slices/step_<step>_window_<window>/` directory that produced them.

The Hybrid and Zebra comparison on MOOC and WikiTalk is documented in
[`docs/community-evaluation-results.md`](docs/community-evaluation-results.md).

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
python hybrid_community.py eval --device cuda:0 \
  --output outputs/hybrid_mooc_community.json
python zebra_community.py eval --device cuda:0 \
  --output outputs/zebra_mooc_community.json
```

## Project Structure

```
train_coreness.py               # Hybrid coreness model training entry
hybrid_community.py             # On-demand coreness BFS query/evaluation
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
