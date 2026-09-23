# Project Overview

k-core community prediction in temporal networks. Given `(q, k, t)`, use
snapshots through t to predict the connected k-core component containing q in
the next snapshot.

## Tech Stack

Python 3, networkit, NumPy, PyTorch. Numba accelerates the active T-PPR
implementation.

## Directory Structure

```
coreness-prediction/
├── train_coreness.py              # Hybrid coreness model training
├── hybrid_community.py            # On-demand coreness BFS prediction
├── zebra_community.py             # Zebra link-based community prediction
├── methods/
│   ├── hybrid_coreness.py          # Coreness model + community recovery
│   └── tcs_representation.py       # Node-level temporal features
├── datasets/
│   ├── dataset_builder.py         # Snapshot building + caching (networkit-based)
│   └── community_eval_builder.py  # Community test sample generation + caching
├── third_party/                    # Vendored Zebra, EAGLE, SWIFT, TFWaveFormer, PRISM sources
│   └── SWIFT/swift-dgl/            # Modified DGL source needed for SWIFT native build
├── data/
│   └── <dataset>/                 # Raw CSV and all generated data artifacts
│       ├── <dataset>.csv          # Raw edges: u,v,ts
│       └── time_slices/
│           └── step_<step>_window_<window>/
│               ├── metadata.json # Window byte ranges into the main CSV
│               ├── snapshot_cache/  # Cached k-core snapshots
│               ├── sample_cache/    # Cached test samples
│               └── community_eval/  # Persisted evaluation set (optional)
└── docs/                          # Project documentation
```

## Active Methods

- **Hybrid coreness prediction** (`hybrid_community.py`): Traverse the cumulative graph from q, predict each new BFS frontier in batches, and expand only through nodes whose predicted next-snapshot coreness is at least k. Queries at the same time share node-level TCS, T-PPR influence, structure-feature, and predicted-coreness caches, so each touched node is featurized and inferred at most once per time slice. No k-core peeling is applied.
- **Zebra community prediction** (`zebra_community.py`): Predict links over the historical community candidate set, run k-core decomposition, and return q's connected component.
- **Other link baselines** (`{eagle,swift,tfwaveformer,prism}_community.py`): Use the vendored model code under `third_party/` and the shared snapshot/community evaluation adapters.

Retired methods and historical specifications are kept locally under the
Git-ignored `archive/` directory, not distributed with the active experiments.

## TCS Formula

```
TCS(v,t,k) = 1 - (1/W) Σ w_i · (k - c_i) / max(k, c_i)
```
α=0.7, LOOKBACK=5. Fixed τ=0.15 or dynamic relax threshold.

## Data

CSV format `u,v,ts`. All subsequent sample generation, evaluation, and result
reporting must use `k in [3, 4, 5, 6, 7]`. Do not test or report `k >= 8`
unless the user explicitly requests a different range.

New time slices use `indexed_csv_v1`: the main CSV remains a single immutable
file and `metadata.json` stores window byte ranges, row counts and fingerprints.
Legacy CSV slices remain readable. Use `python -u -m datasets.migrate_slice_storage`
to verify and migrate existing configurations while preserving all caches.
The ordinary `build_time_slices` command remains a rebuild operation that
replaces old configurations and derived caches, not a migration command.
The partitioned snapshot cache fast path also validates the indexed source.
All active snapshot consumers, including training, Zebra and evaluation sample
generation, use the same lazy partitioned cache through `build_snapshots`.
Cache v3 is keyed by logical windows and feature versions. Edge partitions use
compressed two-column integer arrays (`*-edges.npz`) with checked ID widths;
the legacy edge triple API reconstructs min-core from `core_dict`. v2 caches
upgrade without rebuilding features or changing streaming-state identities.
Legacy `snapshots.pkl`
and superseded partitions are deleted only after verified atomic publication.
Use `python -m datasets.snapshot_store <time_slices_dir> ...` for storage-only
migration without adding missing dense distributions. Stop concurrent readers
before migration; normal loads complete missing features when needed.
Only K=3–7 community components are computed/stored (`community_ks` in the
partitioned index). Existing caches are pruned without changing the retained
communities or model features. Node coreness and `max_core` remain unrestricted;
do not clamp labels or structural values to the community evaluation range.
Completed distribution caches omit `h_index_dicts`; keep them only for slices
without valid distributions. Preserve `max_h_index`/`hmax` metadata.
Normal model inputs read distributions directly; custom/fallback computations
reconstruct missing h-index intermediates transiently from the snapshot graph.
Active numerical arrays/states use float32; IDs and indices retain integer
types. Structure storage migrates legacy float64 arrays without recomputation.
Float32 T-PPR/TCS training uses `hybrid_features_v9_float32_current_snapshot_*`;
old numeric history caches must not be reused. Python/JSON scalars and timing
clocks retain native types. External Zebra runtime types are preserved at the
adapter boundary; its persisted floating tensors are float32.
Ours state-cache identity `v2:` hashes logical window contents and feature
metadata, not physical slice storage. Old unversioned state caches are not
automatically trusted; only streaming state needs a one-time recomputation.

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
