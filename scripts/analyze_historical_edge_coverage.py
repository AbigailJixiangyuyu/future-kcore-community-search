"""Measure historical-edge recall inside cached future ground-truth communities."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def edges(snapshot):
    return {(min(u, v), max(u, v)) for u, v, *_ in snapshot["edge_list"] if u != v}


def measure(community, future, history, current):
    truth = {(u, v) for u, v in future if u in community and v in community}
    if not truth:
        raise ValueError("Nonempty k>=3 community must contain edges")
    covered = len(truth & history)
    return {
        "nodes": len(community),
        "future_edges": len(truth),
        "covered_edges": covered,
        "new_edges": len(truth) - covered,
        "coverage": covered / len(truth),
        "current_snapshot_coverage": len(truth & current) / len(truth),
        "fully_covered": covered == len(truth),
    }


def aggregate(rows):
    return {
        "count": len(rows),
        "macro_coverage": float(np.mean([r["coverage"] for r in rows])),
        "micro_coverage": sum(r["covered_edges"] for r in rows)
        / sum(r["future_edges"] for r in rows),
        "fully_covered_fraction": float(np.mean([r["fully_covered"] for r in rows])),
        "minimum_coverage": min(r["coverage"] for r in rows),
        "p10_coverage": float(np.quantile([r["coverage"] for r in rows], 0.1)),
        "current_snapshot_macro_coverage": float(
            np.mean([r["current_snapshot_coverage"] for r in rows])
        ),
    }


def analyze(root, dataset, slice_name, zebra_root):
    base = root / "data" / dataset / "time_slices" / slice_name
    with (base / "snapshot_cache/snapshots.pkl").open("rb") as source:
        snapshots = pickle.load(source)["snapshots"]
    split = int(len(snapshots) * 0.7)
    sample_path = base / "sample_cache" / (
        f"community_samples_v1_{dataset}_split{split}_k3_4_5_6_7_n20_e5_s42.pkl"
    )
    with sample_path.open("rb") as source:
        samples = pickle.load(source)
    zebra_name = {
        "email-Eu-core-temporal": "email-snapshot",
        "mooc": "mooc-snapshot",
        "wiki-talk-temporal": "wiki-talk-snapshot",
    }[dataset]
    converted = zebra_root / "data" / zebra_name
    with (converted / f"ml_{zebra_name}.csv").open() as source:
        timestamps = [float(row["ts"]) for row in csv.DictReader(source)]
    boundary = float(np.quantile(timestamps, 0.85))
    with (converted / "snapshot_mapping.csv").open() as source:
        start_t = max(
            int(row["slice_index"])
            for row in csv.DictReader(source)
            if float(row["zebra_ts"]) <= boundary
        )
    selected = [s for s in samples if s["t"] >= start_t and s["k"] in range(3, 8)]
    nonempty = [s for s in selected if s["community"]]
    by_time = {}
    for sample in nonempty:
        by_time.setdefault(sample["t"], []).append(sample)
    history, rows, unique_rows = set(), [], []
    current = edges(snapshots[0])
    for t in range(len(snapshots) - 1):
        history.update(current)
        future = edges(snapshots[t + 1])
        measured = {}
        for sample in by_time.get(t, []):
            community = frozenset(sample["community"])
            if community not in measured:
                measured[community] = measure(
                    community, future, history, current
                )
                unique_rows.append({"t": t, **measured[community]})
            rows.append({
                "query": sample["query"], "k": sample["k"], "t": t,
                **measured[community],
            })
        current = future
    signature = sorted((s["query"], s["k"], s["t"]) for s in nonempty)
    return {
        "dataset": dataset,
        "start_t": start_t,
        "end_t": max(s["t"] for s in nonempty),
        "sample_cache": str(sample_path),
        "empty_communities_excluded": len(selected) - len(nonempty),
        "query_set_sha256": hashlib.sha256(
            json.dumps(signature).encode()
        ).hexdigest(),
        "overall": aggregate(rows),
        "per_k": {str(k): aggregate([r for r in rows if r["k"] == k])
                  for k in range(3, 8)},
        "unique_time_community": aggregate(unique_rows),
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    datasets = {
        "email-Eu-core-temporal": "step_302400_window_604800",
        "mooc": "step_43200_window_86400",
        "wiki-talk-temporal": "step_259200_window_604800",
    }
    parser.add_argument(
        "--datasets", nargs="+", choices=datasets,
        default=["email-Eu-core-temporal", "mooc"],
        help="Uses full existing sample caches, not the wiki 1/6 subset.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        slices = datasets[dataset]
        result = analyze(root, dataset, slices, root.parent / "Zebra")
        (args.output_dir / f"{dataset}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
