#!/usr/bin/env python3
"""
Convert raw temporal edge CSV (u,v,ts) to snapshot-indexed CSV (u,i,ts,label,idx).

Usage:
    python -m datasets.convert_to_snapshot_csv <dataset_name>
    python -m datasets.convert_to_snapshot_csv email-Eu-core-temporal

Output: data/<dataset_name>/ml_<dataset_name>.csv
"""

import sys
from pathlib import Path

import pandas as pd

DAY = 86400
WEEK = 7 * DAY
MONTH = 30 * DAY
YEAR = 1

DATASET_WINDOW = {
    "email-Eu-core-temporal": WEEK,
    "sx-mathoverflow": 4 * WEEK,
    "sx-askubuntu": 4 * WEEK,
    "mooc": DAY,
    "DBLP1": YEAR,
    "wiki-talk-temporal": MONTH,
    "sx-superuser": 4 * WEEK,
}


def convert(dataset_name):
    src_path = Path("datasets") / f"{dataset_name}.csv"
    if not src_path.exists():
        print(f"Error: {src_path} not found")
        sys.exit(1)

    if dataset_name not in DATASET_WINDOW:
        print(f"Error: no window config for '{dataset_name}'")
        print(f"Available: {', '.join(DATASET_WINDOW.keys())}")
        sys.exit(1)

    window = DATASET_WINDOW[dataset_name]

    df = pd.read_csv(src_path)
    df = df.rename(columns={"v": "i"})

    df = df.sort_values("ts").reset_index(drop=True)

    ts_min = df["ts"].min()
    df["ts"] = ((df["ts"] - ts_min) // window) + 1

    df["label"] = 0.0
    df["idx"] = range(1, len(df) + 1)

    df = df[["u", "i", "ts", "label", "idx"]]

    out_dir = Path("data") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ml_{dataset_name}.csv"

    df.to_csv(out_path, index=True)
    print(f"Done: {out_path} ({len(df)} edges, {df['ts'].nunique()} snapshots)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m datasets.convert_to_snapshot_csv <dataset_name>")
        sys.exit(1)
    convert(sys.argv[1])
