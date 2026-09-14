"""Read-only, field-partitioned snapshots with bounded in-process retention."""

import json
import os
import pickle
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path


VERSION = 1


def _group(key):
    if key == "core_dict":
        return "core"
    if key == "structure_distributions":
        return "structure"
    if key == "edge_list":
        return "edges"
    return "other"


def source_identity(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_snapshot_store(directory, snapshots, total_nodes, kmax, hmax,
                         source=None):
    """Publish an independent generation atomically; never overwrite old data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    keys = []
    edge_nodes = set()
    for time, snapshot in enumerate(snapshots):
        keys.append(list(snapshot))
        groups = {}
        for key, value in snapshot.items():
            groups.setdefault(_group(key), {})[key] = value
        for name, values in groups.items():
            with (generation / "{}-{}.pkl".format(time, name)).open("wb") as out:
                pickle.dump(values, out, protocol=pickle.HIGHEST_PROTOCOL)
        for edge in snapshot["edge_list"]:
            if edge[0] != edge[1]:
                edge_nodes.update(edge[:2])
    with (generation / "nodes.pkl").open("wb") as out:
        pickle.dump(sorted(edge_nodes), out, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = dict(version=VERSION, generation=generation.name, keys=keys,
                    total_nodes=total_nodes, kmax=kmax, hmax=hmax, source=source)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(directory),
                                     suffix=".tmp", delete=False) as out:
        json.dump(metadata, out)
        temporary = out.name
    os.replace(temporary, directory / "index.json")


class SnapshotView(Mapping):
    """A view owns no loaded arrays, graphs or dictionaries."""

    def __init__(self, store, time):
        self.store = store
        self.time = time

    def __getitem__(self, key):
        if key not in self.store.keys[self.time]:
            raise KeyError(key)
        return self.store._load(self.time, _group(key))[key]

    def __iter__(self):
        return iter(self.store.keys[self.time])

    def __len__(self):
        return len(self.store.keys[self.time])


class SnapshotStore(Sequence):
    """Keep five core dictionaries and at most one slice per other partition.

    Eviction drops the store's references, never mutating data held by a caller.
    No complete snapshot is reconstructed by indexing or iterating this sequence.
    """

    def __init__(self, directory, core_lookback=5):
        directory = Path(directory)
        self.metadata = json.loads((directory / "index.json").read_text())
        if self.metadata["version"] != VERSION:
            raise ValueError("Unsupported snapshot store version")
        self.directory = directory / self.metadata["generation"]
        self.keys = self.metadata["keys"]
        with (self.directory / "nodes.pkl").open("rb") as source:
            self.node_ids = pickle.load(source)
        self.set_core_lookback(core_lookback)
        self._cache = {name: OrderedDict() for name in self._limits}
        self.load_counts = {name: 0 for name in self._limits}

    def set_core_lookback(self, lookback):
        if lookback <= 0:
            raise ValueError("core lookback must be positive")
        self._limits = dict(core=int(lookback), edges=1, structure=1, other=1)
        if hasattr(self, "_cache"):
            for name, cache in self._cache.items():
                while len(cache) > self._limits[name]:
                    cache.popitem(last=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, time):
        if isinstance(time, slice):
            return [self[i] for i in range(*time.indices(len(self)))]
        time = int(time)
        if time < 0:
            time += len(self)
        if not 0 <= time < len(self):
            raise IndexError(time)
        return SnapshotView(self, time)

    def _load(self, time, name):
        cache = self._cache[name]
        if time not in cache:
            # Evict before reading to avoid an unnecessary double-snapshot peak.
            while len(cache) >= self._limits[name]:
                cache.popitem(last=False)
            with (self.directory / "{}-{}.pkl".format(time, name)).open("rb") as src:
                cache[time] = pickle.load(src)
            self.load_counts[name] += 1
        cache.move_to_end(time)
        return cache[time]

    def retain_for_prediction(self, time, lookback=5):
        """Discard unrelated evaluation/replay data before building G_t inputs."""
        self.set_core_lookback(lookback)
        for name, cache in self._cache.items():
            for cached_time in list(cache):
                keep = (max(0, time - lookback + 1) <= cached_time <= time
                        if name == "core" else cached_time == time)
                if not keep:
                    del cache[cached_time]

    def release_payloads(self):
        """After input materialization only recent coreness is still needed."""
        for name, cache in self._cache.items():
            if name != "core":
                cache.clear()

    def clear(self):
        for cache in self._cache.values():
            cache.clear()

    def resident_times(self):
        return {name: list(cache) for name, cache in self._cache.items()}


def open_snapshot_store(slices_dir):
    """Reuse partitioned caches, or perform a one-time legacy migration.

    Migration loads the legacy pickle once. Subsequent prediction runs never
    load it. The legacy cache is preserved for training and other evaluators.
    """
    from datasets.dataset_builder import build_snapshots

    slices_dir = Path(slices_dir)
    legacy = slices_dir / "snapshot_cache" / "snapshots.pkl"
    directory = slices_dir / "snapshot_cache" / "partitioned"
    index = directory / "index.json"
    valid = False
    if index.exists() and legacy.exists():
        metadata = json.loads(index.read_text())
        valid = (metadata.get("version") == VERSION
                 and metadata.get("source") == source_identity(legacy))
    if not valid:
        snapshots, total_nodes, kmax, hmax = build_snapshots(slices_dir)
        write_snapshot_store(directory, snapshots, total_nodes, kmax, hmax,
                             source_identity(legacy))
        del snapshots
    store = SnapshotStore(directory)
    return (store, store.metadata["total_nodes"], store.metadata["kmax"],
            store.metadata["hmax"])
