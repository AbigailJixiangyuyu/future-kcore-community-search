"""Bridge the bundled TFWaveFormer runtime without polluting models/utils imports."""

import importlib
from pathlib import Path
import sys
import types

import numpy as np

from datasets.dataset_builder import build_snapshots


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "TFWaveFormer"


def load_runtime(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    name = "_coreness_tfwaveformer"
    if not (root / "snapshot_inference.py").is_file():
        raise FileNotFoundError("TFWaveFormer snapshot adapter missing: {}".format(root))
    if name in sys.modules:
        if list(sys.modules[name].__path__) != [str(root)]:
            raise ValueError("a different TFWaveFormer root is already loaded")
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(name + ".snapshot_inference")


class SnapshotEdges:
    """Project the shared lazy snapshot store onto two-column integer edges."""

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, t):
        rows = self.snapshots[t]["edge_list"]
        array = np.asarray(rows)
        if not len(array):
            return np.empty((0, 2), dtype=np.int64)
        return array[:, :2]


def load_snapshot_edges(slices_dir):
    snapshots, _, _, _ = build_snapshots(slices_dir)
    return SnapshotEdges(snapshots)


def load_predictor(slices_dir, checkpoint, device="cpu", tfwaveformer_root=DEFAULT_ROOT):
    runtime = load_runtime(tfwaveformer_root)
    return runtime.TFWaveFormerSnapshotPredictor(
        load_snapshot_edges(slices_dir), checkpoint=checkpoint, device=device,
    )
