"""Load EAGLE's frozen snapshot scorer against the shared snapshot store."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

from datasets.dataset_builder import build_snapshots


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "EAGLE"


def load_runtime(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    module_path = root / "snapshot_inference.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    name = "_coreness_eagle_snapshot_inference"
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != module_path:
            raise ValueError("a different EAGLE root is already loaded")
        return sys.modules[name]
    existing = sys.modules.get("link_prediction")
    if existing is not None and str(root / "link_prediction") not in list(existing.__path__):
        raise ValueError("link_prediction is loaded from a different project")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class SnapshotEdges:
    """Lazy two-column view; EAGLE only receives edges through prepared time."""

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, t):
        rows = np.asarray(self.snapshots[t]["edge_list"])
        if not len(rows):
            return np.empty((0, 2), dtype=np.int64)
        return rows[:, :2].astype(np.int64, copy=False)


def load_snapshot_edges(slices_dir):
    return SnapshotEdges(build_snapshots(slices_dir)[0])


def load_predictor(snapshots, *, config, checkpoint=None, device="cpu", eagle_root=DEFAULT_ROOT):
    runtime = load_runtime(eagle_root)
    return runtime.EagleSnapshotPredictor(
        SnapshotEdges(snapshots), config=config, checkpoint=checkpoint, device=device,
    )
