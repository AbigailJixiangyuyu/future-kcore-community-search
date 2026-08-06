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
import pickle
from collections import defaultdict
from pathlib import Path

import networkit as nk


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DAY = 86400
THREE_DAY = 3 * DAY
WEEK = 7 * DAY
MONTH = 30 * DAY
YEAR = 1

DEFAULT_TEST_RATIO = 0.3
TARGET_KS = [1, 2, 3, 4, 5, 6]

DATASET_VALID_KS = {
    "email-Eu-core-temporal": [3, 4, 5, 6, 7],
    "sx-mathoverflow": [3, 4, 5, 6, 7, 8, 9, 10],
    "sx-askubuntu": [3, 4, 5, 6, 7, 8],
    "mooc": [3, 4, 5, 6, 7],
    "DBLP1": [3, 4, 5, 6, 7, 8, 9, 10],
    "wiki-talk-temporal": [3, 4, 5, 6, 7],
    "sx-superuser": [3, 4, 5, 6],
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

    nodes_by_core = defaultdict(set)
    for node, core in core_dict.items():
        nodes_by_core[core].add(node)

    k_core_comps = {}
    for k in range(1, max_core + 1):
        cumulative_ids = set()
        for core in range(k, max_core + 1):
            cumulative_ids.update(node_to_id[node] for node in nodes_by_core.get(core, set()))

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

    return {
        "edge_list": [(u, v, min(core_dict[u], core_dict[v])) for u, v, _, _ in filtered_edges],
        "core_dict": core_dict,
        "max_core": max_core,
        "k_core_comps": k_core_comps,
    }


def build_snapshots(slices_dir):
    """Build one k-core snapshot per generated time-slice CSV.

    The input is a ``time_slices/step_<step>_window_<window>`` directory, not
    the raw dataset CSV. The resulting cache is scoped to that exact slice
    configuration so different window choices cannot share stale state.
    """
    slices_dir = Path(slices_dir)
    manifest = load_time_slice_manifest(slices_dir)
    cache_dir = slices_dir / "snapshot_cache"
    cache_path = cache_dir / "snapshots.pkl"

    if cache_path.exists():
        print(f"[dataset_builder] Loading cached snapshots from {cache_path}")
        with cache_path.open("rb") as cache_file:
            cached = pickle.load(cache_file)
        return cached["snapshots"], cached["total_nodes"]

    snapshots = []
    total_node_ids = set()
    for position, slice_info in enumerate(manifest["slices"]):
        slice_filename = slice_info.get("file")
        if not slice_filename:
            raise ValueError(f"Slice {position} has no file in {slices_dir / 'metadata.json'}")
        slice_path = slices_dir / slice_filename
        if not slice_path.is_file():
            raise FileNotFoundError(f"Time-slice file not found: {slice_path}")

        slice_edges = _load_slice_edges(slice_path)
        snapshot = _build_snapshot(slice_edges)
        snapshot["slice_index"] = slice_info.get("index", position)
        snapshot["start_ts"] = slice_info.get("start_ts")
        snapshot["end_ts"] = slice_info.get("end_ts")
        snapshots.append(snapshot)
        total_node_ids.update(node for edge in slice_edges for node in edge)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(
            {"snapshots": snapshots, "total_nodes": len(total_node_ids)},
            cache_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[dataset_builder] Cached snapshots to {cache_path}")

    return snapshots, len(total_node_ids)
