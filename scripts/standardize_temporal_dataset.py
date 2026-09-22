"""Convert temporal edge files to sorted integer u,v,ts CSV."""

import argparse
from array import array
import csv
from decimal import Decimal, ROUND_FLOOR
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def standardize(source, output, input_format, deduplicate_same_second=False,
                min_timestamp=None, node_mapping_output=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    mapping_path = (Path(node_mapping_output)
                    if node_mapping_output is not None else None)
    if input_format == "coin" and mapping_path is None:
        raise ValueError("coin requires node_mapping_output")
    if mapping_path is not None:
        if input_format != "coin":
            raise ValueError("node_mapping_output is only supported for coin")
        if mapping_path == output or mapping_path.exists():
            raise FileExistsError(mapping_path)
    node_ids = {}
    values = array("q")
    count = loops = fractional = descents = before_start = 0
    previous = None
    with source.open(newline="") as handle:
        if input_format in ("ml", "csv"):
            reader = csv.DictReader(handle)
            destination = "i" if input_format == "ml" else "v"
            if not {"u", destination, "ts"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Expected u,{destination},ts columns")
            rows = ((row["u"], row[destination], row["ts"]) for row in reader)
        elif input_format == "txt":
            rows = (line.split() for line in handle
                    if line.strip() and not line.lstrip().startswith("#"))
        elif input_format == "coin":
            reader = csv.reader(handle)
            header = next(reader)
            if header[:3] != ["day", "src", "dst"]:
                raise ValueError("Expected day,src,dst as first three columns")

            def coin_rows():
                for row in reader:
                    if len(row) != 4:
                        raise ValueError("Expected four fields in coin data row")
                    ts, src, dst, _ = row
                    if not src or not dst:
                        raise ValueError("Empty coin address")
                    # One namespace for both endpoints, deterministic first appearance.
                    for address in (src, dst):
                        if address not in node_ids:
                            node_ids[address] = len(node_ids)
                    yield node_ids[src], node_ids[dst], ts

            rows = coin_rows()
        else:
            raise ValueError("input_format must be ml, csv, txt or coin")
        for row in rows:
            u, v, timestamp = row
            u, v = int(u), int(v)
            timestamp = Decimal(timestamp)
            if not timestamp.is_finite():
                raise ValueError("Non-finite timestamp")
            count += 1
            descents += previous is not None and timestamp < previous
            previous = timestamp
            ts = int(timestamp.to_integral_value(rounding=ROUND_FLOOR))
            fractional += timestamp != ts
            if u == v:
                loops += 1
                continue
            if min_timestamp is not None and ts < min_timestamp:
                before_start += 1
                continue
            values.extend((u, v, ts))
    edges = np.frombuffer(values, dtype=np.int64).reshape(-1, 3)
    order = np.argsort(edges[:, 2], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    mapping_temporary = None
    mapping_created = False
    written = duplicates = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=output.parent, prefix=".standardize-",
            suffix=".csv", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("u", "v", "ts"))
            current_ts = None
            seen = set()
            for index in order:
                u, v, ts = edges[index]
                if deduplicate_same_second:
                    if ts != current_ts:
                        seen.clear()
                        current_ts = ts
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                writer.writerow((u, v, ts))
                written += 1
        if mapping_path is not None:
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", dir=mapping_path.parent,
                prefix=".node-mapping-", suffix=".csv", delete=False
            ) as handle:
                mapping_temporary = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(("node_id", "address"))
                writer.writerows((index, address)
                                 for address, index in node_ids.items())
            os.link(mapping_temporary, mapping_path)
            mapping_created = True
        # Atomic creation without overwriting any existing destination.
        os.link(temporary, output)
    except BaseException:
        if mapping_created:
            mapping_path.unlink()
        raise
    finally:
        if temporary is not None:
            temporary.unlink()
        if mapping_temporary is not None:
            mapping_temporary.unlink()
    return dict(source=str(source), output=str(output), input_records=count,
                output_records=written, removed_self_loops=loops,
                removed_same_second_duplicates=duplicates,
                removed_before_start=before_start,
                fractional_timestamps=fractional, time_descents=descents,
                mapped_nodes=len(node_ids),
                node_mapping_output=str(mapping_path) if mapping_path else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--format", required=True, choices=("ml", "csv", "txt", "coin"))
    parser.add_argument("--node-mapping-output",
                        help="Required for coin: save node_id,address mapping")
    parser.add_argument("--min-timestamp", type=int,
                        help="Keep floored timestamps >= this value")
    parser.add_argument(
        "--deduplicate-same-second", action="store_true",
        help="Keep the first record per undirected pair and floored second",
    )
    args = parser.parse_args()
    print(json.dumps(standardize(
        args.source, args.output, args.format, args.deduplicate_same_second,
        args.min_timestamp, args.node_mapping_output,
    )))
