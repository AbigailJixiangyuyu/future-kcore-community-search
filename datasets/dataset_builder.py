#!/usr/bin/env python3
"""Build k-core snapshots from generated temporal edge slices.

Pipeline:
    data/<dataset>/<dataset>.csv
        -> datasets.build_time_slices
        -> data/<dataset>/time_slices/step_<step>_window_<window>/
        -> build_snapshots
"""

import csv
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import networkit as nk

from datasets.indexed_slices import STORAGE_FORMAT, iter_indexed_edges, validate_source
from methods.h_index_representation import (
    MAX_H_INDEX_ORDER,
    STRUCTURE_DISTRIBUTION_KEY,
    STRUCTURE_DISTRIBUTION_VERSION,
    add_structure_distributions,
    compute_h_index_levels,
    has_structure_distributions,
)


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DAY = 86400
THREE_DAY = 3 * DAY
WEEK = 7 * DAY
MONTH = 30 * DAY
YEAR = 1

DEFAULT_TEST_RATIO = 0.3
TARGET_KS = [3, 4, 5, 6, 7]

DATASET_VALID_KS = {
    "email-Eu-core-temporal": [3, 4, 5, 6, 7],
    "sx-mathoverflow": [3, 4, 5, 6, 7],
    "sx-askubuntu": [3, 4, 5, 6, 7],
    "mooc": [3, 4, 5, 6, 7],
    "DBLP1": [3, 4, 5, 6, 7],
    "wiki-talk-temporal": [3, 4, 5, 6, 7],
    "sx-superuser": [3, 4, 5, 6, 7],
}

# Recommended default slice configuration for each dataset. These values are
# only used to locate already-generated time slices; they never re-bin raw data.
DATASET_WINDOW = {
    "email-Eu-core-temporal": WEEK,
    "sx-mathoverflow": 4 * WEEK,
    "sx-askubuntu": 4 * WEEK,
    "mooc": DAY,
    "DBLP1": YEAR,
    "wiki-talk-temporal": MONTH,
    "sx-superuser": 4 * WEEK,
}


def time_slices_dir(dataset_name, step_seconds, window_seconds):
    """Return the canonical directory for one generated slice configuration."""
    return (
        DATA_ROOT
        / dataset_name
        / "time_slices"
        / f"step_{step_seconds}_window_{window_seconds}"
    )


def load_time_slice_manifest(slices_dir):
    """Load and validate the manifest produced by ``build_time_slices``."""
    slices_dir = Path(slices_dir)
    manifest_path = slices_dir / "metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Time-slice metadata not found: {manifest_path}. "
            "Run datasets.build_time_slices first."
        )
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)

    required_keys = {"dataset", "step_seconds", "window_seconds", "slices"}
    if not required_keys.issubset(manifest):
        raise ValueError(f"Invalid time-slice metadata: {manifest_path}")
    if not isinstance(manifest["slices"], list) or not manifest["slices"]:
        raise ValueError(f"Time-slice metadata has no slices: {manifest_path}")
    storage = manifest.get("storage_format")
    if storage == STORAGE_FORMAT:
        validate_source(slices_dir, manifest)
    elif storage is not None:
        raise ValueError(f"Unsupported slice storage format: {storage}")
    return manifest


def _load_slice_edges(slice_path):
    """Read one standardized slice CSV without reordering or re-binning it."""
    with slice_path.open(newline="") as slice_file:
        reader = csv.DictReader(slice_file)
        required_columns = {"u", "v", "ts"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{slice_path} must have a u,v,ts header")
        try:
            return [(int(row["u"]), int(row["v"])) for row in reader]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid edge in {slice_path}") from error


def load_slice_edges(slices_dir, manifest, slice_info):
    if manifest.get("storage_format") == STORAGE_FORMAT:
        source = validate_source(slices_dir, manifest)
        return list(iter_indexed_edges(source, slice_info))
    filename = slice_info.get("file")
    if not filename:
        raise ValueError("Legacy slice has no file")
    return _load_slice_edges(Path(slices_dir) / filename)


def _build_snapshot(slice_edges):
    unique_nodes = sorted({node for edge in slice_edges for node in edge})
    node_to_id = {node: index for index, node in enumerate(unique_nodes)}
    id_to_node = {index: node for node, index in node_to_id.items()}

    filtered_edges = []
    seen = set()
    for u, v in slice_edges:
        if u == v:
            continue
        ui, vi = node_to_id[u], node_to_id[v]
        edge_key = (min(ui, vi), max(ui, vi))
        if edge_key not in seen:
            seen.add(edge_key)
            filtered_edges.append((u, v, ui, vi))

    graph = nk.Graph(len(unique_nodes))
    for _, _, ui, vi in filtered_edges:
        graph.addEdge(ui, vi)

    core_values = nk.centrality.CoreDecomposition(graph).run().scores()
    core_dict = {id_to_node[index]: int(core) for index, core in enumerate(core_values)}
    max_core = int(max(core_values)) if core_values else 0

    k_core_comps = {}
    for k in TARGET_KS:
        if k > max_core:
            continue
        cumulative_ids = {
            node_to_id[node] for node, core in core_dict.items() if core >= k
        }

        cumulative_list = sorted(cumulative_ids)
        local_id = {node_id: index for index, node_id in enumerate(cumulative_list)}
        subgraph = nk.Graph(len(cumulative_list))
        for _, _, ui, vi in filtered_edges:
            if ui in cumulative_ids and vi in cumulative_ids:
                subgraph.addEdge(local_id[ui], local_id[vi])

        components = []
        for component in nk.components.ConnectedComponents(subgraph).run().getComponents():
            original_nodes = frozenset(id_to_node[cumulative_list[node_id]] for node_id in component)
            if original_nodes:
                components.append(original_nodes)

        k_core_comps[k] = {
            "node_set": {id_to_node[node_id] for node_id in cumulative_ids},
            "components": components,
        }

    snapshot = {
        "edge_list": [(u, v, min(core_dict[u], core_dict[v])) for u, v, _, _ in filtered_edges],
        "core_dict": core_dict,
        "max_core": max_core,
        "k_core_comps": k_core_comps,
    }
    _add_h_index_features(snapshot)
    return snapshot


def _add_h_index_features(snapshot):
    """Add the fixed set of h-index levels used by structural features."""
    # Derived distributions cannot survive a rebuild of their source values.
    snapshot.pop(STRUCTURE_DISTRIBUTION_KEY, None)
    h_index_dicts = compute_h_index_levels(
        snapshot["edge_list"],
        snapshot["core_dict"],
        max_order=MAX_H_INDEX_ORDER,
    )
    snapshot["h_index_dicts"] = h_index_dicts
    snapshot["max_h_index"] = _first_order_hmax(snapshot)


def _first_order_hmax(snapshot):
    """Return the maximum first-order h-index in one snapshot."""
    first_order = snapshot["h_index_dicts"][1]
    return max((int(value) for value in first_order.values()), default=0)


def _has_h_index_features(snapshot):
    h_index_dicts = snapshot.get("h_index_dicts")
    return (
        isinstance(h_index_dicts, dict)
        and all(
            isinstance(h_index_dicts.get(order), dict)
            for order in range(1, MAX_H_INDEX_ORDER + 1)
        )
    )


def _prepare_structure_distributions(snapshots, hmax):
    """Upgrade legacy snapshots after the dataset-wide bucket width is known."""
    started = time.perf_counter()
    built = 0
    for snapshot in snapshots:
        if not has_structure_distributions(snapshot, hmax):
            add_structure_distributions(snapshot, hmax)
            built += 1
    if built:
        print(
            "[dataset_builder] Precomputed structure distributions for "
            "{} snapshots in {:.3f}s (hmax={})".format(
                built, time.perf_counter() - started, hmax
            ),
            flush=True,
        )
    return bool(built)


def _write_snapshot_cache(cache_path, snapshots, total_nodes, kmax, hmax):
    """Replace an upgraded cache only after the complete pickle is written."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_path.parent, prefix="snapshots-", suffix=".tmp",
            delete=False,
        ) as cache_file:
            temporary_path = Path(cache_file.name)
            pickle.dump(
                {"snapshots": snapshots, "total_nodes": total_nodes,
                 "kmax": kmax, "hmax": hmax},
                cache_file, protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_snapshots(slices_dir):
    """Open the single partitioned cache used by all active consumers."""
    from datasets.snapshot_store import open_snapshot_store
    return open_snapshot_store(slices_dir)


def _build_snapshot_inputs(slices_dir, manifest, staging, precompute_structures=True,
                           existing=None):
    """Build one k-core snapshot per indexed window (or legacy slice CSV).

    The input is a ``time_slices/step_<step>_window_<window>`` directory, not
    the raw dataset CSV. The resulting cache is scoped to that exact slice
    configuration so different window choices cannot share stale state.
    """
    slices_dir = Path(slices_dir)
    cache_dir = slices_dir / "snapshot_cache"
    cache_path = cache_dir / "snapshots.pkl"
    if existing is not None:
        def completed():
            for view in existing:
                snapshot = dict(view)
                if not has_structure_distributions(snapshot, existing.metadata["hmax"]):
                    add_structure_distributions(snapshot, existing.metadata["hmax"])
                yield snapshot
        return (completed(), existing.metadata["total_nodes"],
                existing.metadata["kmax"], existing.metadata["hmax"])

    if cache_path.exists():
        print(f"[dataset_builder] Loading cached snapshots from {cache_path}")
        with cache_path.open("rb") as cache_file:
            cached = pickle.load(cache_file)
        snapshots = cached["snapshots"]
        total_nodes = cached["total_nodes"]
        for snapshot in snapshots:
            if not _has_h_index_features(snapshot):
                _add_h_index_features(snapshot)
            max_h_index = _first_order_hmax(snapshot)
            if snapshot.get("max_h_index") != max_h_index:
                snapshot["max_h_index"] = max_h_index
        kmax = cached.get(
            "kmax", max((snapshot["max_core"] for snapshot in snapshots), default=0)
        )
        hmax = max(
            (snapshot["max_h_index"] for snapshot in snapshots),
            default=0,
        )
        def upgraded():
            for snapshot in snapshots:
                if precompute_structures and not has_structure_distributions(snapshot, hmax):
                    add_structure_distributions(snapshot, hmax)
                yield snapshot
                # Release upgraded dense arrays as the writer advances.
                snapshot.pop(STRUCTURE_DISTRIBUTION_KEY, None)
        return upgraded(), total_nodes, kmax, hmax

    total_node_ids = set()
    kmax = hmax = 0
    for position, slice_info in enumerate(manifest["slices"]):
        slice_edges = load_slice_edges(slices_dir, manifest, slice_info)
        snapshot = _build_snapshot(slice_edges)
        snapshot["slice_index"] = slice_info.get("index", position)
        snapshot["start_ts"] = slice_info.get("start_ts")
        snapshot["end_ts"] = slice_info.get("end_ts")
        # Dataset-wide hmax is needed before constructing distribution tables.
        # Stage only one snapshot at a time instead of retaining the dataset.
        with (staging / "{}.pkl".format(position)).open("wb") as out:
            pickle.dump(snapshot, out, protocol=pickle.HIGHEST_PROTOCOL)
        kmax = max(kmax, snapshot["max_core"])
        hmax = max(hmax, snapshot["max_h_index"])
        total_node_ids.update(node for edge in slice_edges for node in edge)

    total_nodes = len(total_node_ids)
    def finished():
        for position in range(len(manifest["slices"])):
            path = staging / "{}.pkl".format(position)
            with path.open("rb") as src:
                snapshot = pickle.load(src)
            add_structure_distributions(snapshot, hmax)
            yield snapshot
            path.unlink()
    return finished(), total_nodes, kmax, hmax
