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

Five link-prediction baselines (Zebra, EAGLE, SWIFT, TFWaveFormer, PRISM)
are included under `third_party/`; this repository's `methods/` adapters and
`*_community.py` entry points load them there by default. `scripts/baseline_training_batch.py`
also trains against the included sources. Data, checkpoints, and native build
artifacts are not included. SWIFT requires a CUDA build of its bundled DGL
source; see `third_party/README.md` for the exact source and build boundaries.

Hybrid streaming state can be checkpointed before evaluation without
materializing model features. The default cache time is the 70% evaluation
boundary; later `query` and `eval` commands automatically load the newest
compatible cache at or before their query time.
Normal Hybrid prediction also creates this cache automatically: on its first
requested time, load a compatible state if available; otherwise replay history
(or advance an older compatible state) and save the resulting target-time state.
A cache hit is not rewritten, and later incremental times do not each create a
file. The cache stores the cumulative graph and current T-PPR/TCS state, not
model features or predictions. Current-time model inputs are still constructed
after restoration. Cache reading and initial writing count toward `prepare_s`;
the normal cache directory is `model_cache/hybrid_state/` beneath the slice
directory. Passing `use_cache=False` to `advance_state` disables both automatic
reading and writing; explicit `build-state-cache` keeps control of its output.

```bash
python hybrid_community.py build-state-cache \
  --slices-dir data/mooc/time_slices/step_43200_window_86400 \
  --checkpoint results/current_snapshot_training_20260914_4ymmR4/mooc.pt \
  --device cpu --time 52
```

### Coreness-guided edge generation

Community batch evaluation now generates edges after each BFS query and reports
`edge_case1_ratio`, `edge_case2_ratio`, and `edge_case3_ratio` alongside F1/Jaccard.
These fractions describe final undirected edges by the case that created their
current lifetime; reverse-endpoint selections do not double-count or relabel an
edge, while complete deletion and recreation reset its attribution.
Per-K ratios average non-edgeless queries; macro ratios average valid K means.
Edgeless queries have zero counts and `null` ratios and are reported separately
as `edge_ratio_empty_samples` (versus `edge_ratio_valid_samples`).
`edge_case1_count`–`edge_case3_count` are cumulative audit counts, not macro
weights. JSON ratios are in [0,1]; multiply by 100 for percentage display.
`edge_generation_s` reports generation time; query time includes this step.
F1/Jaccard still use the unchanged BFS node set, with no peeling.

Batch evaluation also reports `edge_precision`, `edge_recall`, `edge_f1`,
and `edge_jaccard`. Predictions are all final generated edges; truth consists
only of edges in snapshot t+1 with both endpoints in q's true connected k-core
community. A predicted edge outside that community is incorrect even if it
exists elsewhere in the future snapshot. Undirected edges are deduplicated and
self-loops excluded. Per-K values average all queries, and macro values average
K=3–7 means. An empty prediction against nonempty truth scores zero for all four
metrics; both sets empty scores one, matching node-set evaluation.
These metrics are shared by the default T-PPR 20, T-PPR 40 and historical-neighbor
batch evaluators. Future edges are used only for evaluation, never prediction.
Truth extraction and metric calculation are excluded from community query time
(`elapsed_s` and `query_s`), but included in wall time. Pure single-query prediction
does not require a future snapshot.

Community timing outputs use `timing_version=2` and the monotonic
`time.perf_counter()` clock. The single-query and batch interfaces share these
definitions (all durations are seconds):

| Field | Scope |
|---|---|
| `load_s` | CLI predictor construction, including snapshot/model loading and index initialization |
| `state_update_s` | Historical state restoration/advancement performed during this preparation |
| `feature_materialize_s` | Feature-table construction performed during this preparation |
| `prepare_s` | Complete `prepare_time` call, including state/feature work and preparation overhead |
| `bfs_selection_s` | BFS, feature gathering, model inference, CPU/GPU transfers and prediction-cache lookup |
| `edge_generation_s` | Final edge generation, propagation and generation-result construction |
| `query_s`, `elapsed_s` | Identical: `bfs_selection_s + edge_generation_s`; excludes preparation and evaluation |
| `prediction_total_s` | `prepare_s + query_s`; excludes loading, evaluation and output serialization |
| `wall_s` | CLI entry into prediction/evaluation work through result readiness; includes loading and evaluation, excludes JSON serialization and output |

`state_update_s` and `feature_materialize_s` are components of `prepare_s`,
not additional terms to add to it. A reused time context reports zero for those
two components, but the preparation/cache lookup itself is still timed.
Ours shares coreness predictions and merged first-order T-PPR rows across queries
at the same time, clearing both when the time context changes.
Prepared time-level features and history state remain shared.
JSON identifies `cross_query_coreness_cache: true` and
`cross_query_tppr_score_cache: true`; Zebra remains independent across queries.
New/reused node prediction counts indicate the actual inference work. Returning predictions to CPU
synchronizes the inference results before the BFS timer ends.

Batch top-level durations are totals; `prepare_s` is charged once per queried
time, while per-K and macro `elapsed_s`/`query_s` are query means and then K means.
`amortized_prediction_s` is `(total prepare_s + total query_s) / sample_count`,
or `null` for no samples; it is a sample-weighted amortized duration, **not** a
cold independent-query latency or a K-macro mean. `sample_prepare_s` measures
evaluation sample preparation; `metric_s` measures per-query truth/metric work.
Both are excluded from `prediction_total_s`. Other evaluation bookkeeping and
progress logging remain included in `wall_s`.

The node-only `hybrid_community.py query` has `prediction_scope=nodes_only` and
zero edge-generation time. The complete `generated_edge_community.py` CLI
reports `nodes_and_edges`. When its `predict_community()` function receives an
already constructed predictor, loading is outside its scope: `load_s` is omitted
and `wall_scope` explicitly identifies function-level timing.

Historical JSON files are unchanged. Before version 2, single-query `elapsed_s`
included preparation, `bfs_selection_s` also included preparation, and batch
`wall_s` excluded loading/sample preparation. Do not compare these legacy
fields directly to version-2 fields. No cache is cleared or model warmed up
automatically for timing; report cache conditions when comparing cold and warm
queries. These timing changes apply to our shared evaluator and its candidate
variants, not the independent Zebra evaluator.

`case3_lowered_node_ratio` is the number of distinct nodes lowered by case 3
divided by all selected BFS community nodes. Repeated decreases of one node
count once. Empty communities have `null` ratios and are excluded from averaging;
nonempty but edgeless communities are included. Per-K query means and valid-K
macro means use this independent valid set, audited by `core_ratio_valid_samples`,
`core_ratio_empty_samples`, and cumulative `case3_lowered_node_count`.

The historical-neighbor and T-PPR capacity-40 ablation entry points were
removed on 2026-09-16; their retired commands are not supported by the current code.

The independent `generated_edge_community.py` script first selects nodes by
q-rooted BFS through nodes with predicted coreness >= k on the full historical
graph. It then generates undirected edges only within that selected node set,
using reverse h-index and raw T-PPR scores. Each source's edge candidates are
the distinct vertices in its cached T-PPR Top-L records intersected with the
BFS set, excluding self; historical direct adjacency is not required.
Scores are summed across temporal records; uncached vertices are excluded.
The default predictor reads raw T-PPR node/score arrays in batches, without
creating influence objects or normalizing unused weights. Merged first-order
score rows are cached for nodes requested by edge generation at the current
time, and reused across queries. Every query filters those rows by its
own BFS set; two-hop scores and propagation state remain query-specific.
The score cache is cleared when the time context changes, retains no old-time rows,
and uses at most O(M * top_l) entries for M distinct requested source nodes.
Two-hop score construction and sorting are deferred until case 3 still needs
support after checking existing edges and first-order candidates. Scores,
ID-ordered input rows, and rankings are cached within the query and reused.
Only first-order reverse relations are stored upfront; a core decrease expands
them by up to two reverse hops to recover the same affected sources, without
materializing unused two-hop scores or storing all two-hop dependencies.
Propagation, floating-point path-sum order, and tie-breaking rules are unchanged.
Initial row construction is included in query/edge-generation timing, not
moved into prepare_s. Historical-neighbor and capacity-40 comparison paths
retain their existing candidate adapters.
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

```bash
python generated_edge_community.py 413 7 52 \
  --slices-dir data/mooc/time_slices/step_43200_window_86400 \
  --checkpoint results/current_snapshot_training_20260914_4ymmR4/mooc.pt \
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
Online `query` and `eval` queries independently recompute node encoding and
historical-edge scores, sharing only prepared historical state.
Progressive evaluation reports historical candidate-edge counts, not all-pairs
counts. Result JSON identifies the full-history, historical-edges-only policy.

The previous sliding-window/all-pairs Zebra implementation and its email rho
runner were removed on 2026-09-16.

```bash
# t is the zero-based coreness snapshot index
python zebra_community.py query 413 7 52 --device cuda:0

# Non-empty MOOC ground-truth samples in the leakage-free Zebra test period
python zebra_community.py eval --device cuda:0 \
  --output outputs/zebra_mooc_community.json
```

The default MOOC paths look for converted `mooc-snapshot` data and its trained
checkpoint beneath `third_party/Zebra/`; these generated artifacts are not
included in Git and must be prepared locally or supplied via CLI options.
The evaluation start is derived
from Zebra's 85% time boundary; for the current 60-snapshot MOOC data it is
`t=52`, predicting snapshot 53 (Zebra timestamp 54).

`query` and `eval` share the online timing-v2 entry point:

- `prepare_s`: update Zebra memory/T-PPR once per time slice. At the first
  requested boundary, load a compatible history-state cache if present;
  otherwise replay from the origin and atomically save that boundary.
  Cache loading, validation, reconstruction and saving are included in
  `prepare_s`. `initial_prepare_s` identifies this first preparation, not a
  steady-state update.
- `query_s` / `elapsed_s`: candidate-community union search, per-query node
  encoding, historical-edge scoring and graph construction, k-core decomposition
  and connected-community extraction. Their stage times are also reported.
- Node embeddings and edge decisions are cleared before every query; no
  cross-query prediction reuse is allowed (`cross_query_cache: false`).
  Historical memory/T-PPR state remains prepared once per time slice.
- These two commands bypass disk prediction/embedding caches, including when
  `--cache-dir` is supplied. Existing artifacts are not removed.
- `load_s` includes model/data loading and construction of the historical-edge
  index. Sampling and truth-metric work are separate from prediction timing.
  CUDA is synchronized at stage boundaries; the clock is `perf_counter`.
- `prepare_mean_s` is total preparation / time-slice count; `query_mean_s` is
  total query time / sample count. `incremental_prepare_mean_s` excludes the
  first preparation. `prediction_total_s = prepare_s + query_s`, and
  `amortized_prediction_s` divides that total by sample count. Empty averages
  are `null`. Per-K averages and equal-K macro averages remain available.

The history cache defaults to `<slices-dir>/model_cache/zebra_state/`.
`--state-cache-dir PATH` selects another directory; `--state-cache-dir ""`
disables it. Only the first requested boundary is saved automatically; later
time slices advance in memory. Only an exact-boundary cache is loaded, never a
future state. Each preparation reports `state_cache_enabled`, `state_cache_hit`,
`state_cache_load_s` and `state_cache_save_s` (in `per_time_preparation` for eval).
The full state includes memory vectors, last-update times, pending-message
vectors/flags/times, all streaming T-PPR normalizers and ordered dictionaries,
the observed timestamp and execution mode. Query embeddings, predicted edges,
model weights and unused validation T-PPR copies are not stored.
Cache identities include the checkpoint, configuration, replay batch size,
mapping, replay data, edge features, implementation and runtime/device.
Invalid/corrupt caches emit a warning, reset history and are rebuilt by replay.
The cache uses tensor/primitive serialization, not pickled Numba objects.
Use only trusted locally generated cache files: older supported PyTorch versions
do not provide the restricted `weights_only` loader.

For Hybrid comparisons, use identical samples, time/query order, hardware and
thread settings. Compare steady-state updates separately from initialization;
warm-cache comparisons require both initial caches to exist before timing.
The September 14 `hybrid_zebra_timing_20260914_n12lr0` experiment predates this
Zebra state cache: its Zebra preparation includes a full initial replay and
must not be presented as a cache-hit result.
The retained `progressive-eval` is a legacy cached-embedding workflow and is
explicitly marked unsuitable for online timing comparisons. Low-level
`predict_graph` still supports its existing disk cache for non-timing callers.

## Methods

**Hybrid coreness prediction** — Runs a layered BFS from q over the cumulative
historical graph. Each layer predicts next-snapshot coreness only for newly
reached nodes; nodes predicted at least `k` join the result and expose the next
layer. No additional k-core peeling is applied.

**Zebra link prediction** — Predicts the next graph over q's historical
community candidate set, then performs k-core decomposition and returns q's
connected component.

For node-level temporal features, `methods.tcs_representation` provides
`tcs_representation(snapshots, u, t, kmax)`. Snapshot preprocessing determines
and caches the dataset-level `kmax`; passing that fixed value keeps every
node's `[TCS(1), ..., TCS(kmax)]` representation the same width. The function
uses only snapshots through `t` and performs no neighbor aggregation.

For current-snapshot structural features, `methods.h_index_representation`
provides `structure_representation(snapshots, u, t, order, cmax)`. Snapshot
preprocessing computes h-index orders 1 through 3 and determines the dataset-level
maximum first-order h-index `hmax`, which is used as the fixed `cmax`. The
representation concatenates the
closed-neighborhood distributions for orders `1..order-1` and current
coreness, using buckets `0..hmax-1` plus a final `>=hmax` bucket. Because
`hmax` is the global maximum, the last bucket represents the exact maximum;
its width is `order * (hmax + 1)` for `order` in `1..4`.

Snapshot preprocessing also persists **all nodes' four closed-neighborhood
distributions**, not just the underlying h-index/coreness values. Each snapshot
stores a node-to-row index and a float32 matrix of shape
`[snapshot_nodes, 4 * (hmax + 1)]`, with a version and bucket-width metadata.
`build_snapshots` automatically fills missing or incompatible distribution
tables in old caches, without rebuilding already-cached graph decompositions;
cache replacement is atomic. The first load of an old cache consequently takes
longer and uses additional disk space. Subsequent loads reuse the tables.

Once a valid complete distribution table is persisted, the three
`h_index_dicts` intermediates are omitted from the snapshot cache.
`max_h_index`, global `hmax`, coreness, communities and distribution values
remain unchanged. Existing caches are pruned atomically, with the policy
recorded as `h_index_storage` in the partitioned index; feature/state identities
are unchanged. Slices without a valid distribution table retain their
intermediates (including the not-yet-preprocessed Wiki distributions).
Normal training/prediction reads distributions without recomputing h-index.
Custom structural queries can reconstruct missing intermediates transiently
from the edges; rebuilding a full distribution table computes them once for
that table, without restoring persistent dictionaries.

For a task at observed time `t` predicting `t+1`, every T-PPR-selected neighbor
uses its structure in **G_t**, not the historical event snapshot stored in its
T-PPR record. Predicting snapshot 53 therefore uses structures from snapshot 52.
The record age `t-event_time`, influence weights and selected slots are unchanged.
Repeated records for the same neighbor share the current structure row.
Training, full-time inference and single-query feature building all use this rule.
The reference calculation remains available for manual snapshots and custom
widths/orders; it also reads G_t. A selected node absent from G_t has each
distribution concentrated in bucket zero, not an invalid padding slot.

Full-time inference now reads T-PPR records through `top_neighbor_arrays`,
without constructing per-slot `TemporalInfluence` objects. Nodes, timestamps,
scores and masks are gathered in bulk; weights are normalized over each row's
exact valid prefix. Inference builds only the current snapshot's float32 table,
with direct node-ID row lookup rather than `(node,time)` deduplication.
The table has a padding row, an absent-node row, and the current snapshot's rows.
It is shared by all queries at t and released with the context; no append-only
historical float32 table remains. Construction is charged to `prepare_s`.
Training, community prediction (Ours/Zebra), and evaluation sample generation
share a single field-partitioned disk snapshot store:
`core_dict` retains at most the model's history length (default five slices);
edges, float32 distributions and other snapshot metadata each retain at most
one slice. Snapshot views themselves hold no loaded payload. After full-time
input materialization the original distribution/graph payloads are released;
only the current float32 model structure table remains. Evaluation loads future
truth separately and drops it after that time's queries.
Streaming-only T-PPR keeps its current Top-20 state and a compact static node-ID
index, not all historical interactions or per-snapshot adjacency tables. TCS
keeps its recursive state, and BFS still requires the cumulative historical
graph. Thus memory is bounded by these necessary states and current/recent
features, **not independent of cumulative graph size**.

On first use, `snapshot_cache/partitioned/` is generated from the legacy cache;
this one-time migration still reads the complete legacy pickle and needs its
peak memory. Each written partition is read back and compared before atomic
index publication. Only then are `snapshots.pkl` and superseded generations
deleted. All later callers use the lazy store without a monolithic copy.
The store's identity uses logical window hashes and feature versions,
independent of legacy pickle files. Changed inputs are rejected, not silently
reused. An exclusive build lock prevents concurrent cache writers.

Snapshot store v3 saves each edge partition as `<time>-edges.npz`: a losslessly
compressed two-column endpoint array. Nonnegative node IDs use uint16, uint32
or uint64 according to their maximum; negative IDs use checked int64.
Edge order, orientation and multiplicity are unchanged. The redundant
`min(coreness[u], coreness[v])` column is reconstructed from that slice's
`core_dict`, preserving the existing list-of-triples API. This saves disk space,
not Python edge-list memory; edge reads now also require the slice's core table
and decompression.

Existing v2 stores are upgraded automatically under the build lock: only edges
are re-encoded and read back for exact comparison. Other partitions are
hard-linked (or copied if linking is unavailable), then the new index is
published atomically before old files are removed. Model inputs, feature
metadata and streaming-state identities do not change; no retraining is needed.

Community components are computed and persisted only for K=3–7. A snapshot
whose maximum coreness is below a requested K has no entry for that K (callers
already treat it as an empty community). This does **not** clip node coreness,
`max_core`, h-index values, distributions, or model labels. Existing stores
automatically drop out-of-range entries from `k_core_comps`, verify retained
content, and atomically publish the result; edges and other feature partitions
are reused. The index records `community_ks`, separately from model-feature
metadata, so valid evaluation samples and streaming states remain reusable.

Fresh builds stage individual graph snapshots to determine global feature
dimensions, then compute/write distribution tables one slice at a time.
Staging files are temporary; they are removed on completion or handled errors.
This bounds preprocessing retention to a small number of snapshots, but does
not change the memory needs of downstream algorithms.

To migrate existing caches **without adding missing distribution tables**:

```bash
python -m datasets.snapshot_store <time_slices_dir> [<another_time_slices_dir> ...]
```

Normal training/evaluation loads still complete missing distribution tables
when required. Storage-only migration of an old Wiki cache does not trigger
that potentially large preprocessing step. Stop readers of that dataset before
migrating: old snapshot views refer to generation files removed at completion.
This is an I/O/retention change, not a model change; current-snapshot checkpoints
need no retraining. Disk reads during state/feature preparation count toward
`prepare_s`; future-truth reads remain evaluation time.
History input assembly borrows the recent core dictionaries once per consecutive
sample-time group, rather than traversing the snapshot cache for every node/lag.
Only one history window is borrowed at a time, with no dictionary copies or
persistent extra cache. Current structure-table validation and materialization
reuse one cache fetch and one sorted node list.
Time deltas, weights and masks are filled with array operations.
Unknown nodes and invalid slots remain zero
and masked. The scalar `top_neighbors` API remains unchanged for edge generation
and other callers; maintained T-PPR state, ranking and capacity are unchanged.

This corrects the old feature semantics and requires retraining. Training uses
the new `hybrid_features_v9_float32_current_snapshot_*` cache namespace and never reuses
v7 historical-structure inputs. Checkpoints must explicitly contain
`feature_config.structure_time_reference="current_observed_snapshot"`; missing
or different metadata is rejected. Obsolete feature caches and incompatible
checkpoints were removed during cleanup; they are not silently converted.

Active numerical arrays use float32: structure distributions, TCS penalties
and representations, maintained T-PPR scores/normalizers, model features and
Zebra history-cache floating tensors. Legacy float64 structure files are cast
once, read back and compared with the former float32 model input, then
published atomically without recomputing graph features.
IDs/counts/index arrays remain integers. Python/JSON scalars and timing clocks
keep their native types; external Zebra runtime/Numba types are not modified
here (its cache adapter converts at the boundary).

Float32 T-PPR/TCS arithmetic can change rounding and near-tie rankings.
Training therefore uses a new v9 feature namespace; old v8 features are not
reused. Hybrid history caches require `numeric_dtype="float32"` and Zebra
history caches use version 2/new identities, so old numeric states are rebuilt.
Existing model weights remain loadable, but inference warns for checkpoints
without the float32 feature marker; re-evaluate their accuracy under the new
numeric pipeline. This change does not automatically retrain models.

Distribution precomputation belongs to snapshot preprocessing (or the one-time
cache upgrade), not `prepare_s`. End-to-end timing for a newly arriving snapshot
must include that preprocessing cost; moving the work outside `prepare_s` is
not itself an end-to-end speedup. The savings come from avoiding repeated
historical distribution calculations across later prediction times.

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

Structure pooling now uses only path B: set attention followed by
query-conditioned pooling and LayerNorm. T-PPR weights remain an attention
score prior. Path A and the pooling CLI selector have been removed.
Only checkpoints explicitly marked with `structure_pooling="b"` are compatible
with this pooling design.

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
its CLI flag have been removed.

The only output head is `Linear(32, kmax+1)` after
the existing 128-to-32 state projection. Its weights are initially copied from
the input coreness table, but are separate parameters and train independently.
The tied decoder, output-head selector and dedicated comparison scripts have
been removed. Checkpoints must explicitly declare `output_head_type="linear"`;
tied checkpoints and missing output-head metadata are rejected. Obsolete default
checkpoint files were deleted and not replaced. Pass `--checkpoint` explicitly
when predicting with the compatible local models:
`results/current_snapshot_training_20260914_4ymmR4/{email,mooc}.pt`.
The older fusion-ablation and Wiki checkpoints use retired feature semantics
and were deleted along with obsolete local experiment artifacts. Wiki requires
retraining before use with the current predictor. Old unversioned Hybrid state
caches were also deleted; the next run rebuilds them with the current identity.
Reusable snapshot/sample caches and Zebra state caches remain intact.

For inference, `hybrid_community.py` builds one cumulative adjacency and T-PPR
state through the query time. It predicts `q` first, then deduplicates each BFS
frontier, creates features only for those nodes, and sends them to the model in
bounded batches. At one query time, TCS vectors, T-PPR Top-L influences,
structure features, and predicted coreness values are cached by node and shared
across all `(q, k)` queries. A node is therefore featurized and inferred at most
once per time slice while BFS still avoids untouched nodes. The CLI reports both
examined nodes and new model predictions.

## Community Recovery

1. Index time windows in the main CSV with `datasets.build_time_slices`.
2. Read the indexed byte ranges and construct cached k-core snapshots.
3. Use either the hybrid model to predict next-snapshot node coreness on demand
   during BFS or Zebra to predict next-snapshot links.
4. Hybrid returns q's threshold-connected BFS region; Zebra recovers q's
   connected k-core component from its predicted graph.
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

This writes `metadata.json` (format `indexed_csv_v1`) to
`data/email-Eu-core-temporal/time_slices/step_604800_window_604800/`, without
duplicating records into `slice_*.csv`. The main CSV must have exactly the
`u,v,ts` header and one integer record per line, sorted by timestamp. Each
non-empty window records `[start_ts, end_ts)`, `[start_byte, end_byte)`, its row
count and an ordered SHA-256 of normalized records. The source size,
nanosecond modification time and SHA-256 are recorded too. Readers seek
directly into the main CSV, preserving row order, duplicates and orientation.
Overlapping windows reference the same source bytes instead of copying them.
Normal loading checks source size and modification time, including before
loading existing snapshot caches or taking the warm partitioned-cache fast
path; these checks are not a full hash rescan.
Treat the indexed source as immutable: editing it requires rebuilding both
the index and derived caches.
Windows are aligned backwards from `last_timestamp + 1`, so the newest window
is complete. Leading windows that overlap the dataset are retained, so any
short observation interval occurs only at the beginning, not at the prediction
end of the timeline. Each dataset keeps one active slice configuration. A
successful rebuild replaces its previous `time_slices` directory, including
stale snapshot, sample, and evaluation caches derived from the old slices.

To migrate **existing** configurations without changing windows or deleting
reusable caches, use the separate migration command:

```bash
python -u -m datasets.migrate_slice_storage
```

It discovers only existing `data/*/time_slices/*/metadata.json` configurations;
datasets without chosen windows are left alone. For each configuration it
compares every legacy slice's record count and ordered normalized SHA-256
against the source range before atomically publishing the index. It then
deletes only the verified legacy slice CSVs, preserving all cache directories
and a small `legacy-metadata.json` audit copy. Deleted slice CSVs can be
reconstructed from the unchanged source and window boundaries; they are not
kept in a backup. The migrated `file` fields are historical cleanup identifiers,
not reader inputs. Interruptions during cleanup are safely resumable by
rerunning the command. Failed verification leaves the legacy configuration
untouched; failures are logged, other configurations continue, and the process
exits nonzero if any failed. Do not run source edits, slice rebuilds or
training/evaluation concurrently with migration. Old CSV-backed configurations
remain readable until migrated.

Ours historical-state cache identities use `v2:` content keys: dataset/window
parameters, ordered window boundaries/counts/content hashes, and parsed
snapshot-feature metadata. Storage format, slice filenames, byte offsets,
source modification time and JSON formatting do not enter this identity.
Consequently a verified CSV-to-index migration does not invalidate caches
created with the v2 scheme. Indexed identities reuse certified hashes without
rescanning the main CSV; legacy CSV configurations must hash their slice
contents once at predictor construction. Old, unversioned state-cache keys
hashed only metadata file bytes and did not certify data contents. They are
not automatically aliased to v2: the first prediction rebuilds historical
state once and uses the existing automatic save mechanism. Model weights,
graph snapshots and training samples do not need rebuilding for this key
upgrade. Do not bypass source validation by manually restoring file timestamps.

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
datasets/
  dataset_builder.py            # Snapshot building + caching
  coreness_prediction_builder.py # Node samples and model feature tensors
  community_eval_builder.py     # Community sample generation + caching
data/
  <dataset>/
    <dataset>.csv               # Raw temporal edges: u,v,ts
    time_slices/
      step_<step>_window_<window>/
        metadata.json           # Window byte ranges, counts and source signature
        snapshot_cache/         # Derived k-core snapshots
        sample_cache/           # Derived test samples
        community_eval/         # Optional persisted evaluation set
```
