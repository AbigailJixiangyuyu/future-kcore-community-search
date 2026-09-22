"""Read-only, field-partitioned snapshots with bounded in-process retention."""

import json
import fcntl
import gzip
import hashlib
import os
import operator
import pickle
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


VERSION = 3
H_INDEX_STORAGE = "only_without_valid_distributions"
STRUCTURE_DTYPE = "float32"


def _float32_structure(table):
    values = table["values"]
    if not isinstance(values, np.ndarray) or values.dtype not in (
            np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Unsupported structure distribution dtype")
    return dict(table, values=values.astype(np.float32, copy=False))


def _edge_array(edges, cores):
    """Preserve order/orientation; never truncate IDs or silently discard data."""
    smallest = largest = 0
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("Expected (u, v, min_core) snapshot edges")
        u, v = operator.index(edge[0]), operator.index(edge[1])
        if edge[2] != min(cores[u], cores[v]):
            raise ValueError("Edge coreness cannot be reconstructed from core_dict")
        smallest = min(smallest, u, v)
        largest = max(largest, u, v)
    if smallest < 0:
        if smallest < -(2 ** 63) or largest >= 2 ** 63:
            raise ValueError("Signed node IDs exceed int64 range")
        dtype = np.dtype("<i8")
    else:
        if largest >= 2 ** 64:
            raise ValueError("Node IDs exceed uint64 range")
        dtype = np.dtype("<u2" if largest <= 65535 else
                         "<u4" if largest <= 4294967295 else "<u8")
    values = np.empty((len(edges), 2), dtype=dtype)
    for row, edge in enumerate(edges):
        values[row] = edge[:2]
    return values


def _read_edges(path, cores):
    with np.load(str(path), allow_pickle=False) as archive:
        values = archive["endpoints"]
    if (values.ndim != 2 or values.shape[1] != 2
            or values.dtype.kind not in ("u", "i")
            or values.dtype.itemsize not in (2, 4, 8)):
        raise ValueError("Invalid compact edge array")
    return [(int(u), int(v), min(cores[int(u)], cores[int(v)]))
            for u, v in values]


def _write_edges(path, edges, cores):
    values = _edge_array(edges, cores)
    with path.open("wb") as out:
        np.savez_compressed(out, endpoints=values)
        out.flush()
        os.fsync(out.fileno())
    _assert_equal(edges, _read_edges(path, cores))


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _link_partition(path, destination):
    if not path.is_file() or path.is_symlink():
        raise ValueError("Unexpected file in snapshot generation")
    try:
        os.link(path, destination)
    except OSError:
        shutil.copy2(path, destination)
        if _file_digest(path) != _file_digest(destination):
            raise ValueError("Snapshot partition copy verification failed")


def _community_subset(components):
    from datasets.dataset_builder import TARGET_KS
    return {k: value for k, value in components.items() if k in TARGET_KS}


def _assert_equal(expected, actual):
    """Validate serialization without relying on pickle byte layout."""
    import numpy as np
    if isinstance(expected, np.ndarray):
        if expected.dtype != actual.dtype:
            raise ValueError("Snapshot array dtype changed")
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError("Snapshot keys changed")
        for key in expected:
            _assert_equal(expected[key], actual[key])
    elif expected != actual:
        raise ValueError("Snapshot content changed")


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent),
                                         suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            json.dump(value, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_structure_partition(path):
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
        source.seek(0)
        if compressed:
            with gzip.GzipFile(fileobj=source, mode="rb") as stream:
                return pickle.load(stream)
        return pickle.load(source)


def _write_structure_partition(path, values):
    with path.open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", compresslevel=1) as stream:
            pickle.dump(values, stream, protocol=pickle.HIGHEST_PROTOCOL)
        destination.flush()
        os.fsync(destination.fileno())
    _assert_equal(values, _read_structure_partition(path))


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
                         source=None, legacy_source=None, expected_count=None):
    """Publish an independent generation atomically; never overwrite old data."""
    from datasets.dataset_builder import TARGET_KS
    from methods.h_index_representation import has_structure_distributions
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    try:
        keys = []
        edge_nodes = set()
        for time, snapshot in enumerate(snapshots):
            drop_h_index = has_structure_distributions(snapshot, hmax)
            groups = {}
            for key, value in snapshot.items():
                if key == "h_index_dicts" and drop_h_index:
                    continue
                if key == "k_core_comps":
                    value = _community_subset(value)
                if key == "structure_distributions":
                    value = _float32_structure(value)
                groups.setdefault(_group(key), {})[key] = value
            keys.append([key for key in snapshot
                         if key != "h_index_dicts" or not drop_h_index])
            for name, values in groups.items():
                if name == "edges":
                    _write_edges(generation / "{}-edges.npz".format(time),
                                 values["edge_list"], snapshot["core_dict"])
                    continue
                path = generation / "{}-{}.pkl".format(time, name)
                if name == "structure":
                    _write_structure_partition(path, values)
                    continue
                with path.open("wb") as out:
                    pickle.dump(values, out, protocol=pickle.HIGHEST_PROTOCOL)
                    out.flush()
                    os.fsync(out.fileno())
                with path.open("rb") as src:
                    _assert_equal(values, pickle.load(src))
            for edge in snapshot["edge_list"]:
                if edge[0] != edge[1]:
                    edge_nodes.update(edge[:2])
        if expected_count is not None and len(keys) != expected_count:
            raise ValueError("Snapshot count does not match time-slice manifest")
        with (generation / "nodes.pkl").open("wb") as out:
            pickle.dump(sorted(edge_nodes), out, protocol=pickle.HIGHEST_PROTOCOL)
            out.flush()
            os.fsync(out.fileno())
        with (generation / "nodes.pkl").open("rb") as src:
            _assert_equal(sorted(edge_nodes), pickle.load(src))
        metadata = dict(version=VERSION, generation=generation.name, keys=keys,
                        total_nodes=total_nodes, kmax=kmax, hmax=hmax, source=source,
                        legacy_source=legacy_source, community_ks=list(TARGET_KS),
                        h_index_storage=H_INDEX_STORAGE, structure_dtype=STRUCTURE_DTYPE)
        _atomic_json(directory / "index.json", metadata)
    except BaseException:
        shutil.rmtree(generation)
        raise


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
        if self.metadata["version"] not in (2, VERSION):
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
            if name == "edges" and self.metadata["version"] == VERSION:
                cores = self._load(time, "core")["core_dict"]
                cache[time] = {"edge_list": _read_edges(
                    self.directory / "{}-edges.npz".format(time), cores
                )}
            else:
                path = self.directory / "{}-{}.pkl".format(time, name)
                if name == "structure":
                    cache[time] = _read_structure_partition(path)
                else:
                    with path.open("rb") as src:
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


def _upgrade_edge_storage(directory):
    """Rewrite only edges; unchanged partitions are hard-linked then retired."""
    old = SnapshotStore(directory)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    try:
        for time in range(len(old)):
            _write_edges(generation / "{}-edges.npz".format(time),
                         old[time]["edge_list"], old[time]["core_dict"])
        for path in old.directory.iterdir():
            if path.name.endswith("-edges.pkl"):
                continue
            _link_partition(path, generation / path.name)
        metadata = dict(old.metadata, version=VERSION, generation=generation.name)
        _atomic_json(Path(directory) / "index.json", metadata)
    except BaseException:
        shutil.rmtree(generation)
        raise


def _restrict_community_storage(directory):
    """Drop unused K entries; preserve every other cached value verbatim."""
    from datasets.dataset_builder import TARGET_KS
    old = SnapshotStore(directory)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    try:
        for path in old.directory.iterdir():
            destination = generation / path.name
            if path.name.endswith("-other.pkl"):
                with path.open("rb") as src:
                    values = pickle.load(src)
                if "k_core_comps" in values:
                    values["k_core_comps"] = _community_subset(values["k_core_comps"])
                with destination.open("wb") as out:
                    pickle.dump(values, out, protocol=pickle.HIGHEST_PROTOCOL)
                    out.flush()
                    os.fsync(out.fileno())
                with destination.open("rb") as src:
                    _assert_equal(values, pickle.load(src))
            else:
                _link_partition(path, destination)
        metadata = dict(old.metadata, generation=generation.name,
                        community_ks=list(TARGET_KS))
        _atomic_json(Path(directory) / "index.json", metadata)
    except BaseException:
        shutil.rmtree(generation)
        raise


def _prune_h_index_storage(directory):
    """Remove intermediates only when a complete matching distribution exists."""
    from methods.h_index_representation import has_structure_distributions
    old = SnapshotStore(directory)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    keys = [list(items) for items in old.keys]
    removed_times = set()
    try:
        for time, snapshot in enumerate(old):
            if ("h_index_dicts" in snapshot
                    and has_structure_distributions(snapshot, old.metadata["hmax"])):
                removed_times.add(time)
                keys[time].remove("h_index_dicts")
                path = old.directory / "{}-other.pkl".format(time)
                with path.open("rb") as src:
                    values = pickle.load(src)
                values.pop("h_index_dicts")
                destination = generation / path.name
                with destination.open("wb") as out:
                    pickle.dump(values, out, protocol=pickle.HIGHEST_PROTOCOL)
                    out.flush()
                    os.fsync(out.fileno())
                with destination.open("rb") as src:
                    _assert_equal(values, pickle.load(src))
        rewritten = {"{}-other.pkl".format(t) for t in removed_times}
        for path in old.directory.iterdir():
            if path.name not in rewritten:
                _link_partition(path, generation / path.name)
        metadata = dict(old.metadata, generation=generation.name, keys=keys,
                        h_index_storage=H_INDEX_STORAGE)
        _atomic_json(Path(directory) / "index.json", metadata)
    except BaseException:
        shutil.rmtree(generation)
        raise


def _convert_structure_storage(directory):
    """Apply exactly the former model-input cast; leave all other data intact."""
    old = SnapshotStore(directory)
    generation = Path(tempfile.mkdtemp(prefix="generation-", dir=str(directory)))
    try:
        for path in old.directory.iterdir():
            destination = generation / path.name
            if path.name.endswith("-structure.pkl"):
                values = _read_structure_partition(path)
                values["structure_distributions"] = _float32_structure(
                    values["structure_distributions"]
                )
                _write_structure_partition(destination, values)
            else:
                _link_partition(path, destination)
        metadata = dict(old.metadata, generation=generation.name,
                        structure_dtype=STRUCTURE_DTYPE)
        _atomic_json(Path(directory) / "index.json", metadata)
    except BaseException:
        shutil.rmtree(generation)
        raise


def open_snapshot_store(slices_dir, precompute_structures=True):
    """Shared cache entry point; retire legacy data only after verified publication."""
    from datasets.dataset_builder import (
        _build_snapshot_inputs, load_time_slice_manifest, MAX_H_INDEX_ORDER,
        STRUCTURE_DISTRIBUTION_VERSION, TARGET_KS,
    )
    from datasets.indexed_slices import logical_slice_identity

    slices_dir = Path(slices_dir)
    manifest = load_time_slice_manifest(slices_dir)
    logical = logical_slice_identity(slices_dir, manifest)
    identity = hashlib.sha256(json.dumps(
        logical, sort_keys=True, separators=(",", ":")
    ).encode("utf8")).hexdigest()
    source = dict(logical_windows=identity, max_h_index_order=MAX_H_INDEX_ORDER,
                  structure_distribution_version=STRUCTURE_DISTRIBUTION_VERSION)
    cache = slices_dir / "snapshot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    legacy = cache / "snapshots.pkl"
    directory = cache / "partitioned"
    index = directory / "index.json"
    with (cache / ".snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = json.loads(index.read_text()) if index.exists() else {}
        if metadata and metadata.get("version") not in (1, 2, VERSION):
            raise ValueError("Unsupported snapshot store version")
        if metadata.get("community_ks") not in (None, list(TARGET_KS)):
            raise ValueError("Unsupported cached community K range; rebuild explicitly")
        valid = (metadata.get("version") in (2, VERSION)
                 and metadata.get("source") == source)
        if metadata.get("version") in (2, VERSION) and not valid:
            raise ValueError("Snapshot cache inputs changed; rebuild the cache explicitly")
        legacy_signature = source_identity(legacy) if legacy.exists() else None
        if valid and legacy_signature is not None and (
                metadata.get("legacy_source") != legacy_signature):
            raise ValueError("Unexpected legacy cache beside published snapshot store")
        incomplete = valid and any(
            "structure_distributions" not in keys for keys in metadata["keys"]
        )
        if not valid or (precompute_structures and incomplete):
            # Do not keep a second complete dataset in memory for fresh builds.
            with tempfile.TemporaryDirectory(prefix="build-", dir=str(cache)) as stage:
                snapshots, total_nodes, kmax, hmax = _build_snapshot_inputs(
                    slices_dir, manifest, Path(stage), precompute_structures,
                    existing=SnapshotStore(directory) if valid else None,
                )
                write_snapshot_store(directory, snapshots, total_nodes, kmax, hmax,
                                     source=source, legacy_source=legacy_signature,
                                     expected_count=len(manifest["slices"]))
        elif metadata.get("version") == 2:
            _upgrade_edge_storage(directory)
        published = json.loads(index.read_text())
        if published.get("community_ks") != list(TARGET_KS):
            _restrict_community_storage(directory)
        if published.get("h_index_storage") != H_INDEX_STORAGE:
            _prune_h_index_storage(directory)
        if published.get("structure_dtype") != STRUCTURE_DTYPE:
            _convert_structure_storage(directory)
        store = SnapshotStore(directory)
        feature_metadata = dict(
            snapshot_count=len(store), total_nodes=store.metadata["total_nodes"],
            kmax=store.metadata["kmax"], hmax=store.metadata["hmax"],
            max_h_index_order=MAX_H_INDEX_ORDER,
            structure_distribution_version=STRUCTURE_DISTRIBUTION_VERSION,
            structure_distribution_order=MAX_H_INDEX_ORDER + 1,
            structure_distribution_cmax=store.metadata["hmax"],
        )
        metadata_path = cache / "metadata.json"
        if (not metadata_path.exists()
                or json.loads(metadata_path.read_text()) != feature_metadata):
            _atomic_json(metadata_path, feature_metadata)
        if legacy.exists():
            if source_identity(legacy) != legacy_signature:
                raise ValueError("Legacy snapshot cache changed during migration")
            legacy.unlink()
        # Only cache-owned old generations; never follow symlink directories.
        for old in directory.glob("generation-*"):
            if old != store.directory and old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
        return (store, store.metadata["total_nodes"], store.metadata["kmax"],
                store.metadata["hmax"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Migrate existing snapshot caches without adding distribution tables."
    )
    parser.add_argument("slices_dirs", nargs="+")
    args = parser.parse_args()
    for path in args.slices_dirs:
        cache = Path(path) / "snapshot_cache"
        if not ((cache / "snapshots.pkl").exists()
                or (cache / "partitioned" / "index.json").exists()):
            parser.error("No existing snapshot cache: {}".format(path))
        store, *_ = open_snapshot_store(path, precompute_structures=False)
        print("Migrated {}: {} snapshots".format(path, len(store)), flush=True)
