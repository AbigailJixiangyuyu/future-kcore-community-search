"""Bridge the project's lazy snapshot cache to SWIFT's native runtime."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

from datasets.dataset_builder import build_snapshots


SWIFT_ROOT = Path(__file__).resolve().parents[2] / "SWIFT"


class SnapshotEdges:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, t):
        rows = np.asarray(self.snapshots[t]["edge_list"])
        return rows[:, :2] if len(rows) else np.empty((0, 2), dtype=np.int64)


def load_snapshot_edges(slices_dir):
    return SnapshotEdges(build_snapshots(slices_dir)[0])


def load_runtime(root=SWIFT_ROOT):
    root = Path(root).resolve()
    module_name = "_coreness_swift_snapshot"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if Path(module.__file__).resolve().parent != root:
            raise ValueError("different SWIFT root is already loaded")
        return module
    source = root / "snapshot_adapter.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    # SWIFT's original modules use absolute sibling imports (modules, memorys,
    # layers). Restore sys.path after imports; reject collisions with another
    # installed package rather than silently using an unrelated model.
    for name in ("modules", "memorys", "layers", "smoke_test"):
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != root:
            raise ImportError("module collision in SWIFT runtime: " + name)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[module_name]
            raise
        return module
    finally:
        sys.path.remove(str(root))


def load_predictor(slices_dir, checkpoint, swift_root=SWIFT_ROOT):
    runtime = load_runtime(swift_root)
    return runtime.SnapshotPredictor(load_snapshot_edges(slices_dir), checkpoint)
