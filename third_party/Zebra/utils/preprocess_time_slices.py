#!/usr/bin/env python3
"""Convert snapshot CSV files into Zebra's temporal edge-list format."""

import argparse
import csv
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
REQUIRED_SLICE_COLUMNS = {"u", "v", "ts"}


def _validate_data_name(data_name):
  if not data_name or Path(data_name).name != data_name or data_name in {".", ".."}:
    raise ValueError("data name must be one non-empty path component")


def _load_manifest(slices_dir):
  manifest_path = slices_dir / "metadata.json"
  if not manifest_path.is_file():
    raise FileNotFoundError("Time-slice metadata not found: {}".format(manifest_path))
  with manifest_path.open() as manifest_file:
    manifest = json.load(manifest_file)
  slices = manifest.get("slices")
  if not isinstance(slices, list) or not slices:
    raise ValueError("Time-slice metadata must contain a non-empty slices list")
  return manifest, slices


def _iter_slice_edges(slice_path):
  with slice_path.open(newline="") as slice_file:
    reader = csv.DictReader(slice_file)
    if reader.fieldnames is None or not REQUIRED_SLICE_COLUMNS.issubset(reader.fieldnames):
      raise ValueError("{} must have a u,v,ts header".format(slice_path))
    for line_number, row in enumerate(reader, start=2):
      try:
        u = int(row["u"])
        v = int(row["v"])
        int(row["ts"])
      except (TypeError, ValueError) as error:
        raise ValueError(
          "Invalid edge at {}:{}".format(slice_path, line_number)
        ) from error
      yield u, v


def _iter_edges(slice_info, slice_path, indexed):
  if indexed is None:
    return _iter_slice_edges(slice_path)
  return indexed.iter_indexed_edges(slice_path, slice_info)


def _resolve_slices(slices_dir, manifest, slice_entries):
  resolved = []
  indexed = None
  if manifest.get("storage_format") == "indexed_csv_v1":
    module_path = Path(__file__).resolve().parents[3] / "datasets/indexed_slices.py"
    spec = importlib.util.spec_from_file_location("coreness_indexed_slices", module_path)
    indexed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexed)
    source = indexed.validate_source(slices_dir, manifest)
    return [(info, source, indexed) for info in slice_entries]
  seen_files = set()
  for position, slice_info in enumerate(slice_entries):
    if not isinstance(slice_info, dict) or not slice_info.get("file"):
      raise ValueError("Slice {} has no file entry".format(position))
    slice_filename = slice_info["file"]
    if Path(slice_filename).name != slice_filename:
      raise ValueError("Slice {} file must be a plain filename".format(position))
    if slice_filename in seen_files:
      raise ValueError("Duplicate slice file in metadata: {}".format(slice_filename))
    seen_files.add(slice_filename)
    slice_path = slices_dir / slice_filename
    if not slice_path.is_file():
      raise FileNotFoundError("Time-slice file not found: {}".format(slice_path))
    resolved.append((slice_info, slice_path, None))
  return resolved


def _scan_nodes(resolved_slices):
  nodes = set()
  for slice_info, slice_path, indexed in resolved_slices:
    edge_count = 0
    for u, v in _iter_edges(slice_info, slice_path, indexed):
      edge_count += 1
      if u != v:
        nodes.add(u)
        nodes.add(v)
    expected_count = slice_info.get("edge_count")
    if expected_count is not None and edge_count != expected_count:
      raise ValueError(
        "{} contains {} edges, metadata declares {}".format(
          slice_path, edge_count, expected_count
        )
      )
  if not nodes:
    raise ValueError("Time slices contain no non-self-loop edges")
  return {node: index for index, node in enumerate(sorted(nodes), start=1)}


def _write_node_mapping(path, node_mapping):
  with path.open("w", newline="") as mapping_file:
    writer = csv.writer(mapping_file)
    writer.writerow(("original_id", "zebra_id"))
    for original_id, zebra_id in node_mapping.items():
      writer.writerow((original_id, zebra_id))


def _write_edges_and_snapshot_mapping(
  edge_path, snapshot_mapping_path, resolved_slices, node_mapping
):
  total_input_edges = 0
  total_output_edges = 0
  slice_counts = []

  with edge_path.open("w", newline="") as edge_file, snapshot_mapping_path.open(
    "w", newline=""
  ) as snapshot_file:
    edge_writer = csv.writer(edge_file)
    snapshot_writer = csv.writer(snapshot_file)
    edge_writer.writerow(("u", "i", "ts", "label", "idx"))
    snapshot_writer.writerow(
      (
        "zebra_ts",
        "slice_index",
        "start_ts",
        "end_ts",
        "file",
        "input_edge_count",
        "output_edge_count",
        "self_loop_count",
        "duplicate_count",
      )
    )

    edge_idx = 1
    for zebra_ts, (slice_info, slice_path, indexed) in enumerate(resolved_slices, start=1):
      seen = set()
      input_count = 0
      output_count = 0
      self_loop_count = 0
      duplicate_count = 0

      for u, v in _iter_edges(slice_info, slice_path, indexed):
        input_count += 1
        if u == v:
          self_loop_count += 1
          continue
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen:
          duplicate_count += 1
          continue
        seen.add(edge_key)
        edge_writer.writerow(
          (node_mapping[u], node_mapping[v], zebra_ts, 0, edge_idx)
        )
        edge_idx += 1
        output_count += 1

      if output_count == 0:
        raise ValueError(
          "{} has no usable edges after removing self-loops and duplicates".format(
            slice_path
          )
        )

      snapshot_writer.writerow(
        (
          zebra_ts,
          slice_info.get("index", zebra_ts - 1),
          slice_info.get("start_ts", ""),
          slice_info.get("end_ts", ""),
          slice_info["file"],
          input_count,
          output_count,
          self_loop_count,
          duplicate_count,
        )
      )
      total_input_edges += input_count
      total_output_edges += output_count
      slice_counts.append(output_count)

  return {
    "input_edge_count": total_input_edges,
    "output_edge_count": total_output_edges,
    "slice_edge_counts": slice_counts,
  }


def build_zebra_dataset(slices_dir, data_name, output_root=DATA_ROOT):
  """Convert one time-slice directory into a Zebra dataset directory."""
  _validate_data_name(data_name)
  slices_dir = Path(slices_dir)
  output_root = Path(output_root)
  manifest, slice_entries = _load_manifest(slices_dir)
  resolved_slices = _resolve_slices(slices_dir, manifest, slice_entries)
  node_mapping = _scan_nodes(resolved_slices)

  output_root.mkdir(parents=True, exist_ok=True)
  output_dir = output_root / data_name
  if output_dir.exists():
    raise FileExistsError(
      "Output directory already exists; refusing to overwrite: {}".format(output_dir)
    )

  with tempfile.TemporaryDirectory(
    prefix=".{}-".format(data_name), dir=str(output_root)
  ) as temporary_name:
    temporary_dir = Path(temporary_name)
    edge_path = temporary_dir / "ml_{}.csv".format(data_name)
    node_mapping_path = temporary_dir / "node_mapping.csv"
    snapshot_mapping_path = temporary_dir / "snapshot_mapping.csv"

    _write_node_mapping(node_mapping_path, node_mapping)
    summary = _write_edges_and_snapshot_mapping(
      edge_path, snapshot_mapping_path, resolved_slices, node_mapping
    )
    conversion_metadata = {
      "dataset": data_name,
      "source_dataset": manifest.get("dataset"),
      "source_time_slices": str(slices_dir.resolve()),
      "snapshot_count": len(resolved_slices),
      "node_count": len(node_mapping),
      "input_edge_count": summary["input_edge_count"],
      "output_edge_count": summary["output_edge_count"],
      "timestamp_mapping": "manifest order mapped to consecutive integers starting at 1",
    }
    with (temporary_dir / "conversion_metadata.json").open("w") as metadata_file:
      json.dump(conversion_metadata, metadata_file, indent=2, sort_keys=True)
      metadata_file.write("\n")

    temporary_dir.rename(output_dir)

  counts = summary["slice_edge_counts"]
  print("[snapshot-conversion] output={}".format(output_dir))
  print(
    "[snapshot-conversion] snapshots={} nodes={} edges={}".format(
      len(resolved_slices), len(node_mapping), summary["output_edge_count"]
    )
  )
  print(
    "[snapshot-conversion] edges/snapshot min={} median={} max={}".format(
      min(counts), statistics.median(counts), max(counts)
    )
  )
  return output_dir, conversion_metadata


def build_parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input", required=True, help="Directory containing metadata.json and slice CSVs"
  )
  parser.add_argument(
    "--data", required=True, help="Zebra dataset name used for the output directory"
  )
  parser.add_argument(
    "--output-root",
    default=str(DATA_ROOT),
    help="Parent directory for the generated Zebra dataset",
  )
  return parser


def main():
  args = build_parser().parse_args()
  build_zebra_dataset(args.input, args.data, args.output_root)


if __name__ == "__main__":
  main()
