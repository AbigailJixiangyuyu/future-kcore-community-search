#!/usr/bin/env python3
"""Build sliding temporal edge slices for one standardized dataset.

Usage:
    python -m datasets.build_time_slices <dataset> <step_seconds> <window_seconds>

The input must be ``data/<dataset>/<dataset>.csv`` with columns ``u,v,ts``.
Each output slice contains edges whose timestamps are in the half-open interval
``[start_ts, start_ts + window_seconds)``. Windows are aligned backwards from
``last_timestamp + 1`` so the newest window is complete. Leading windows that
overlap the dataset are retained even when they begin before the first observed
timestamp. A step shorter than the window creates overlapping slices.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _iter_edges(source_path):
    """Yield validated, time-ordered standardized edges from a CSV file."""
    with source_path.open(newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_columns = {"u", "v", "ts"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{source_path} must have a u,v,ts header")

        previous_ts = None
        for line_number, row in enumerate(reader, start=2):
            try:
                edge = (int(row["u"]), int(row["v"]), int(row["ts"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid edge at {source_path}:{line_number}") from error
            if previous_ts is not None and edge[2] < previous_ts:
                raise ValueError(
                    f"{source_path} is not sorted by ts at line {line_number}"
                )
            previous_ts = edge[2]
            yield edge


def _slice_name(index):
    return f"slice_{index:06d}.csv"


def _scan_time_bounds(source_path):
    """Return the validated edge count and inclusive timestamp bounds."""
    edge_count = 0
    first_timestamp = None
    last_timestamp = None
    for _, _, timestamp in _iter_edges(source_path):
        if first_timestamp is None:
            first_timestamp = timestamp
        last_timestamp = timestamp
        edge_count += 1

    if edge_count == 0:
        raise ValueError(f"Dataset is empty: {source_path}")
    return edge_count, first_timestamp, last_timestamp


def build_time_slices(dataset_name, step_seconds, window_seconds, data_root=DATA_ROOT):
    """Build non-empty sliding-window edge slices for one dataset.

    ``data_root`` is injectable for tests; the command-line interface always
    uses the repository's ``data/`` directory. A successful rebuild replaces
    the dataset's previous time slices and all caches derived from them.
    """
    if step_seconds <= 0 or window_seconds <= 0:
        raise ValueError("step_seconds and window_seconds must both be positive")

    dataset_dir = Path(data_root) / dataset_name
    source_path = dataset_dir / f"{dataset_name}.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    source_edge_count, first_timestamp, last_timestamp = _scan_time_bounds(source_path)
    observation_end = last_timestamp + 1
    latest_slice_start = observation_end - window_seconds
    if latest_slice_start < first_timestamp:
        raise ValueError(
            f"Dataset time span is shorter than one complete {window_seconds}-second "
            "window"
        )

    # Retain every backwards-aligned window that overlaps the observed range.
    # With overlapping windows this can produce more than one leading partial
    # window, preserving early events under the same regular cadence.
    backward_steps = (observation_end - first_timestamp - 1) // step_seconds
    first_slice_start = latest_slice_start - backward_steps * step_seconds
    planned_slice_count = backward_steps + 1

    config_name = f"step_{step_seconds}_window_{window_seconds}"
    output_parent = dataset_dir / "time_slices"
    output_dir = output_parent / config_name
    temporary_parent = dataset_dir / ".time_slices.tmp"
    temporary_dir = temporary_parent / config_name
    backup_parent = dataset_dir / ".time_slices.backup"
    if temporary_parent.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_parent}")
    if backup_parent.exists():
        if output_parent.exists():
            shutil.rmtree(backup_parent)
        else:
            backup_parent.rename(output_parent)

    temporary_parent.mkdir()
    temporary_dir.mkdir()
    active_writers = {}
    slices = {}
    written_edge_count = 0

    try:
        for u, v, timestamp in _iter_edges(source_path):
            # Close windows before writing an edge at their exclusive end.
            for index, writer in list(active_writers.items()):
                slice_end = first_slice_start + index * step_seconds + window_seconds
                if slice_end <= timestamp:
                    writer[0].close()
                    del active_writers[index]

            offset = timestamp - first_slice_start
            if offset < 0:
                continue
            first_index = max(0, (offset - window_seconds) // step_seconds + 1)
            last_index = min(planned_slice_count - 1, offset // step_seconds)
            for index in range(first_index, last_index + 1):
                if index not in active_writers:
                    slice_start = first_slice_start + index * step_seconds
                    slice_path = temporary_dir / _slice_name(index)
                    slice_file = slice_path.open("w", newline="")
                    writer = csv.writer(slice_file)
                    writer.writerow(("u", "v", "ts"))
                    active_writers[index] = (slice_file, writer)
                    slices[index] = {
                        "index": index,
                        "file": slice_path.name,
                        "start_ts": slice_start,
                        "end_ts": slice_start + window_seconds,
                        "edge_count": 0,
                    }

                active_writers[index][1].writerow((u, v, timestamp))
                slices[index]["edge_count"] += 1
                written_edge_count += 1
    except Exception:
        for slice_file, _ in active_writers.values():
            slice_file.close()
        shutil.rmtree(temporary_parent)
        raise

    for slice_file, _ in active_writers.values():
        slice_file.close()

    manifest = {
        "dataset": dataset_name,
        "source_file": source_path.name,
        "source_edge_count": source_edge_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "observation_end": observation_end,
        "window_alignment": "right",
        "step_seconds": step_seconds,
        "window_seconds": window_seconds,
        "planned_slice_count": planned_slice_count,
        "slice_count": len(slices),
        "written_edge_count": written_edge_count,
        "slices": [slices[index] for index in sorted(slices)],
    }
    with (temporary_dir / "metadata.json").open("w") as metadata_file:
        json.dump(manifest, metadata_file, indent=2)
        metadata_file.write("\n")

    replaced_existing = output_parent.exists()
    if replaced_existing:
        output_parent.rename(backup_parent)
    try:
        temporary_parent.rename(output_parent)
    except Exception:
        if replaced_existing:
            backup_parent.rename(output_parent)
        raise
    if replaced_existing:
        shutil.rmtree(backup_parent)
    return output_dir, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build sliding time slices from data/<dataset>/<dataset>.csv"
    )
    parser.add_argument("dataset", help="Dataset directory and standardized CSV name")
    parser.add_argument("step_seconds", type=int, help="Distance between slice starts")
    parser.add_argument("window_seconds", type=int, help="Inclusive-start, exclusive-end slice length")
    args = parser.parse_args()

    try:
        output_dir, manifest = build_time_slices(
            args.dataset, args.step_seconds, args.window_seconds
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Built {manifest['slice_count']} non-empty slices from "
        f"{manifest['source_edge_count']} edges: {output_dir}"
    )


if __name__ == "__main__":
    main()
