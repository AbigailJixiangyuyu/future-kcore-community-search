# Baseline Sources

These directories contain the code used for the five link-prediction baselines
in this repository. The root-level `methods/` adapters connect the models to
the shared snapshot cache; `*_community.py` supplies the common query and
evaluation protocol. Source directories retain their original internal import
layout so training and checkpoint loading use the same implementations.

| Baseline | Included implementation | Local adaptations |
| --- | --- | --- |
| Zebra | `Zebra/{train.py,inference.py,model/,modules/,utils/,evaluation/}` | Indexed-CSV snapshot preprocessing, training split handling and inference support |
| EAGLE | `EAGLE/link_prediction/train_time.py`, its utilities and `snapshot_inference.py` | Snapshot-time training and inference |
| SWIFT | `SWIFT/` TGAT/TGN model, sampler and snapshot adapter; `swift-dgl/` build sources | Local native build, snapshot training and inference |
| TFWaveFormer | `TFWaveFormer/models/{TFWaveFormer,modules}.py` and `snapshot_inference.py` | Snapshot training and inference |
| PRISM | Snapshot runtime and its imported `PRISM/modules/` | PyTorch compatibility operators for snapshot inference |

Zebra was copied from the local checkout of `LuckyLYM/Zebra` at commit
`da10931`, **including uncommitted local changes** to `train.py`,
`utils/data_processing.py` and `utils/preprocess_time_slices.py`. The other
four sources were copied from local working directories without Git metadata;
their exact upstream revisions cannot be inferred from these directories.
Only this repository's baseline source/usage note is retained in Git; upstream
README files and unrelated documentation are deliberately excluded. One additional
integration edit to Zebra's indexed-CSV preprocessor locates this repository's
`datasets/indexed_slices.py` without relying on a sibling directory. SWIFT's
snapshot adapter and launcher received the corresponding path updates.

The SWIFT snapshot path depends on its locally modified DGL source. The
included `SWIFT/swift-dgl/` contains source and its own `LICENSE`, not compiled
libraries or the original project's examples and tests. Only the PyTorch DGL
backend is retained; the SWIFT launcher sets `DGLBACKEND=pytorch`. TVM, NCCL,
C++ test dependencies, unrelated dependency samples and other backend modules
are omitted because the validated local build does not use them.
From `SWIFT/`, install
the missing Python runtime dependency locally and build the native extensions:

```bash
python -m pip install --no-deps --target .local-deps psutil==5.9.8
bash build_local.sh
bash run_local.sh snapshot_adapter.py --help
```

Python, PyTorch, CUDA, GCC and other build dependencies remain external. The
supplied build defaults to CUDA 11.1 and SM 86; adjust the build configuration
for other machines. The nested METIS ignore rule was updated to keep the
patched GKlib source in Git.

The retained EAGLE snapshot code needs only the Python dependencies in its
`requirements.txt`; install a compatible PyTorch/CUDA wheel separately.
Vendored PRISM, EAGLE, SWIFT and Zebra files exclude unused upstream
node-classification or alternative-model implementations. Zebra retains
its diffusion embedding, which is used by both local training and inference.

Datasets, checkpoints, logs, temporary directories and compiled extensions are
excluded from Git. Copying model code does not bundle training data or model
weights. Before publicly redistributing these sources, verify each upstream
project's redistribution terms and preserve its required attribution. No
top-level license file was present in the four local source directories other
than the license bundled with SWIFT's DGL dependency.
