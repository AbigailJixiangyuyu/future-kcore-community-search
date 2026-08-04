#!/usr/bin/env python3
"""
Unified Dataset Construction for Coreness Prediction
=====================================================
Provides consistent data loading, snapshot building, feature extraction,
and train/test splitting so that ALL prediction methods use the same data.

Usage (as a library):
    from datasets.dataset_builder import build_dataset
    data = build_dataset("datasets/email-Eu-core-temporal.csv")
    X_train, y_train = data["stage1"]["X_train"], data["stage1"]["y_train"]
    X_test,  y_test  = data["stage1"]["X_test"],  data["stage1"]["y_test"]
"""

import csv
import pickle
from collections import defaultdict
from pathlib import Path

import networkit as nk

SNAPSHOT_CACHE_DIR = Path(__file__).parent / "snapshot_cache"

# ── Time constants ─────────────────────────────────────────────────────────

DAY = 86400
THREE_DAY = 3 * DAY
WEEK = 7 * DAY
MONTH = 30 * DAY
YEAR = 1

# ── Default split parameters ───────────────────────────────────────────────

DEFAULT_TEST_RATIO = 0.3
DEFAULT_WINDOW = WEEK

# ── k values ───────────────────────────────────────────────────────────────

TARGET_KS = [1, 2, 3, 4, 5, 6]

DATASET_VALID_KS = {
    "email-Eu-core-temporal": [3, 4, 5, 6, 7],
    "sx-mathoverflow": [3, 4, 5, 6, 7, 8, 9, 10],
    "sx-askubuntu": [3, 4, 5, 6, 7, 8],
    "mooc": [3, 4, 5, 6, 7, 8, 9, 10],
    "DBLP1": [3, 4, 5, 6, 7, 8, 9, 10],
    "wiki-talk-temporal": [3, 4, 5, 6, 7, 8, 9, 10],
    "sx-superuser": [3, 4, 5, 6],

}

DATASET_WINDOW = {
    "email-Eu-core-temporal": WEEK,
    "sx-mathoverflow": 4 * WEEK,
    "sx-askubuntu": 4 * WEEK,
    "mooc": DAY,
    "DBLP1": YEAR,
    "wiki-talk-temporal": MONTH,
    "sx-superuser": 4 * WEEK,

}

# ── Feature names ──────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# 1. Edge loading (from community_evolution.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_edges(path):
    edges = []
    with open(path) as f:
        first_line = f.readline().strip()
        f.seek(0)
        has_header = _has_header(first_line)
        sep = _detect_separator(first_line, has_header)
        if has_header:
            reader = csv.DictReader(f)
            ukey, vkey, tkey = _find_keys(reader.fieldnames)
            for row in reader:
                edges.append((int(row[ukey]), int(row[vkey]), int(row[tkey])))
        else:
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) < 3:
                    continue
                edges.append((int(parts[0]), int(parts[1]), int(parts[2])))
    edges.sort(key=lambda x: x[2])
    return edges


def _has_header(first_line):
    parts = first_line.split()
    if len(parts) >= 3:
        try:
            int(parts[0]); int(parts[1]); int(parts[2])
            return False
        except ValueError:
            return True
    return "," in first_line and any(c.isalpha() for c in first_line)


def _detect_separator(first_line, has_header):
    line = first_line
    if has_header and "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None


def _find_keys(fieldnames):
    u = v = t = None
    for fn in fieldnames:
        fl = fn.lower().strip()
        if fl in ("u", "user", "user_id", "src", "source", "from"):
            u = fn
        elif fl in ("v", "item", "item_id", "dst", "dest", "target", "to"):
            v = fn
        elif fl in ("ts", "timestamp", "time", "t"):
            t = fn
    return u or fieldnames[0], v or fieldnames[1], t or fieldnames[2]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Snapshot building (from community_evolution.py)
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_cache_key(edge_path, window_sec):
    from pathlib import Path
    name = Path(edge_path).stem
    return f"snapshots_{name}_{window_sec}.pkl"


def build_snapshots(edge_path, window_sec, edges=None):
    cache_key = _snapshot_cache_key(edge_path, window_sec)
    cache_path = SNAPSHOT_CACHE_DIR / cache_key

    if cache_path.exists():
        print(f"[dataset_builder] Loading cached snapshots from {cache_path.name}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached["snapshots"], cached["total_nodes"]

    if edges is None:
        edges = load_edges(edge_path)
    if not edges:
        return [], 0

    total_nodes = len(set(u for e in edges for u in (e[0], e[1])))

    ts_min = edges[0][2]
    ts_max = edges[-1][2]
    n_windows = int((ts_max - ts_min) // window_sec) + 1

    edge_bins = [[] for _ in range(n_windows)]
    for u, v, t in edges:
        bi = int((t - ts_min) // window_sec)
        edge_bins[bi].append((u, v))

    snapshots = []
    for idx_seq, i in enumerate(range(n_windows)):
        snap_edges = edge_bins[i]

        if not snap_edges:
            continue

        unique_nodes = sorted(set(u for e in snap_edges for u in e))
        node_to_id = {v: idx for idx, v in enumerate(unique_nodes)}
        id_to_node = {idx: v for v, idx in node_to_id.items()}
        n_nodes = len(unique_nodes)

        filtered_edges = []
        seen = set()
        for u, v in snap_edges:
            if u == v:
                continue
            ui, vi = node_to_id[u], node_to_id[v]
            key = (min(ui, vi), max(ui, vi))
            if key not in seen:
                seen.add(key)
                filtered_edges.append((u, v, ui, vi))

        g = nk.Graph(n_nodes)
        for _, _, ui, vi in filtered_edges:
            g.addEdge(ui, vi)

        core_vals = nk.centrality.CoreDecomposition(g).run().scores()

        core_dict = {id_to_node[i]: int(c) for i, c in enumerate(core_vals)}
        max_core = int(max(core_vals))

        nodes_by_core = defaultdict(set)
        for node, c in core_dict.items():
            nodes_by_core[c].add(node)

        k_core_comps = {}
        for k in range(1, max_core + 1):
            cum_ids = set()
            for kk in range(k, max_core + 1):
                for nd in nodes_by_core.get(kk, set()):
                    cum_ids.add(node_to_id[nd])

            if not cum_ids:
                k_core_comps[k] = {
                    "node_set": set(),
                    "components": [],
                }
                continue

            cum_list = sorted(cum_ids)
            cum_local = {nid: li for li, nid in enumerate(cum_list)}
            sub_g = nk.Graph(len(cum_list))
            for _, _, ui, vi in filtered_edges:
                if ui in cum_ids and vi in cum_ids:
                    sub_g.addEdge(cum_local[ui], cum_local[vi])

            cc = nk.components.ConnectedComponents(sub_g).run()
            nk_components = cc.getComponents()
            components = []
            for comp in nk_components:
                orig = frozenset(id_to_node[cum_list[nid]] for nid in comp)
                if orig:
                    components.append(orig)

            cum_nodes = {id_to_node[nid] for nid in cum_ids}
            k_core_comps[k] = {
                "node_set": cum_nodes,
                "components": components,
            }

        edge_list = [(u, v, min(core_dict[u], core_dict[v])) for u, v, _, _ in filtered_edges]

        snapshots.append({
            "edge_list": edge_list,
            "core_dict": core_dict,
            "max_core": max_core,
            "k_core_comps": k_core_comps,
        })

    SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"snapshots": snapshots, "total_nodes": total_nodes}, f)
    print(f"[dataset_builder] Cached snapshots to {cache_path.name}")

    return snapshots, total_nodes
