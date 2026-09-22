"""Byte-range windows over an immutable, standardized temporal CSV."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


STORAGE_FORMAT = "indexed_csv_v1"


def source_signature(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def parse_record(line):
    """The indexed format deliberately requires one integer record per line."""
    try:
        fields = line.strip().split(b",")
        if len(fields) != 3:
            raise ValueError
        return tuple(int(value) for value in fields)
    except ValueError as error:
        raise ValueError("Indexed CSV requires three integer fields per line") from error


def canonical_record(edge):
    return ("%d,%d,%d\n" % edge).encode("ascii")


def index_windows(source, windows):
    """Scan once, recording exact half-open offsets, counts and ordered hashes."""
    source = Path(source)
    before = source_signature(source)
    events = {}
    digests = [hashlib.sha256() for _ in windows]
    for i, window in enumerate(windows):
        if window["start_ts"] >= window["end_ts"]:
            raise ValueError("Invalid window boundaries")
        events.setdefault(window["start_ts"], [[], []])[0].append(i)
        events.setdefault(window["end_ts"], [[], []])[1].append(i)
    boundaries = sorted(events)
    positions = {}
    active = set()
    cursor = count = 0
    first = previous = None
    source_hash = hashlib.sha256()
    with source.open("rb") as handle:
        header = handle.readline()
        if header.strip() != b"u,v,ts":
            raise ValueError("Indexed CSV requires the exact u,v,ts header")
        source_hash.update(header)
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            edge = parse_record(line)
            timestamp = edge[2]
            if previous is not None and timestamp < previous:
                raise ValueError("Source CSV is not sorted by timestamp")
            if first is None:
                first = timestamp
            previous = timestamp
            while cursor < len(boundaries) and boundaries[cursor] <= timestamp:
                boundary = boundaries[cursor]
                positions[boundary] = (offset, count)
                starts, ends = events[boundary]
                active.difference_update(ends)
                active.update(starts)
                cursor += 1
            normalized = canonical_record(edge)
            for i in active:
                digests[i].update(normalized)
            source_hash.update(line)
            count += 1
        for boundary in boundaries[cursor:]:
            positions[boundary] = (offset, count)
    if before != source_signature(source):
        raise ValueError("Source CSV changed while indexing")
    if not count:
        raise ValueError("Source CSV is empty")
    indexed = []
    for i, window in enumerate(windows):
        start_byte, first_row = positions[window["start_ts"]]
        end_byte, last_row = positions[window["end_ts"]]
        indexed.append(dict(
            window, start_byte=start_byte, end_byte=end_byte,
            edge_count=last_row - first_row, sha256=digests[i].hexdigest(),
        ))
    return indexed, dict(
        source_signature=dict(before, sha256=source_hash.hexdigest()),
        source_edge_count=count, first_timestamp=first, last_timestamp=previous,
    )


def validate_source(slices_dir, manifest):
    name = manifest["source_file"]
    if Path(name).name != name:
        raise ValueError("Source file must be a basename")
    source = Path(slices_dir).parent.parent / name
    signature = manifest["source_signature"]
    if source_signature(source) != {k: signature[k] for k in ("size_bytes", "mtime_ns")}:
        raise ValueError("Source CSV changed; rebuild its window index and derived caches")
    return source


def iter_indexed_edges(source, window):
    """Read only this window; preserve direction, duplicates, and row order."""
    if not 0 <= window["start_byte"] <= window["end_byte"] <= Path(source).stat().st_size:
        raise ValueError("Invalid indexed byte range")
    count = 0
    with Path(source).open("rb") as handle:
        handle.seek(window["start_byte"])
        while handle.tell() < window["end_byte"]:
            line = handle.readline()
            if not line or handle.tell() > window["end_byte"]:
                raise ValueError("Invalid indexed byte range")
            u, v, ts = parse_record(line)
            if not window["start_ts"] <= ts < window["end_ts"]:
                raise ValueError("Indexed record lies outside its time window")
            count += 1
            yield u, v
    if count != window["edge_count"]:
        raise ValueError("Indexed edge count mismatch")


def legacy_digest(path):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as handle:
        if handle.readline().strip() != b"u,v,ts":
            raise ValueError("Legacy slice must have u,v,ts header")
        for line in handle:
            digest.update(canonical_record(parse_record(line)))
            count += 1
    return count, digest.hexdigest()


def logical_slice_identity(slices_dir, manifest):
    """Content identity independent of paths, offsets, JSON layout and storage.

    Indexed CSVs are immutable and use hashes certified at index creation.
    Legacy slices have no certified hashes, so read their content once when
    constructing the predictor identity. This is not a per-query operation.
    """
    directory = Path(slices_dir)
    indexed = manifest.get("storage_format") == STORAGE_FORMAT
    if indexed:
        validate_source(directory, manifest)
    windows = []
    for position, item in enumerate(manifest["slices"]):
        if indexed:
            digest = item["sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid window content hash")
        else:
            filename = item["file"]
            if Path(filename).name != filename:
                raise ValueError("Legacy slice file must be a basename")
            count, digest = legacy_digest(directory / filename)
            if count != item["edge_count"]:
                raise ValueError("Legacy slice record count mismatch")
        windows.append({
            "index": item.get("index", position),
            "start_ts": item["start_ts"], "end_ts": item["end_ts"],
            "edge_count": item["edge_count"], "sha256": digest,
        })
    return {
        "version": "logical_windows_v1",
        "dataset": manifest["dataset"],
        "step_seconds": manifest["step_seconds"],
        "window_seconds": manifest["window_seconds"],
        "slices": windows,
    }


def atomic_manifest(path, manifest):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent),
                                         prefix=".metadata-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
