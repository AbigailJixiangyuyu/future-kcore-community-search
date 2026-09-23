"""Load PRISM snapshot runtime and project the shared lazy snapshot store."""

import importlib
from pathlib import Path
import sys
import types

import numpy as np

from datasets.dataset_builder import build_snapshots


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "PRISM"


def load_runtime(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    name = "_coreness_prism"
    if not (root / "snapshot_inference.py").is_file():
        raise FileNotFoundError("PRISM snapshot adapter missing: {}".format(root))
    if name in sys.modules:
        if list(sys.modules[name].__path__) != [str(root)]:
            raise ValueError("a different PRISM runtime is already loaded")
    else:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return importlib.import_module(name + ".snapshot_inference")


class SnapshotEdges:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, t):
        edges = np.asarray(self.snapshots[t]["edge_list"])
        return edges[:, :2] if len(edges) else np.empty((0, 2), dtype=np.int64)


def load_snapshots(slices_dir):
    snapshots, total_nodes, _, _ = build_snapshots(slices_dir)
    return SnapshotEdges(snapshots), total_nodes, snapshots


def load_predictor(slices_dir, checkpoint, device="cpu", root=DEFAULT_ROOT):
    runtime = load_runtime(root)
    edges, _, snapshots = load_snapshots(slices_dir)
    return runtime.PrismSnapshotPredictor(edges, checkpoint, device), snapshots
