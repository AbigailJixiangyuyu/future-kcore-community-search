"""Migrate existing slice configurations without rebuilding any cached graphs.

Run: python -u -m datasets.migrate_slice_storage
Only existing data/*/time_slices/*/metadata.json configurations are processed.
"""

import argparse
import fcntl
import json
import re
import time
from pathlib import Path

from datasets.build_time_slices import DATA_ROOT
from datasets.indexed_slices import (
    STORAGE_FORMAT, atomic_manifest, index_windows, legacy_digest,
    source_signature, validate_source,
)


def legacy_path(directory, item):
    name = item.get("file")
    if not name or not re.fullmatch(r"slice_\d+\.csv", name):
        raise ValueError("Unsafe or missing legacy slice filename")
    path = directory / name
    if path.is_symlink():
        raise ValueError("Refusing to remove a symlinked legacy slice")
    return path


def migrate_configuration(directory):
    directory = Path(directory)
    with (directory / ".index-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _migrate_locked(directory)


def _migrate_locked(directory):
    manifest_path = directory / "metadata.json"
    old = json.loads(manifest_path.read_text())
    storage = old.get("storage_format")
    if storage not in (None, STORAGE_FORMAT):
        raise ValueError("Unknown storage format")
    paths = [legacy_path(directory, item) for item in old["slices"]
             if item.get("file")]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate legacy filenames")
    validated_stats = {}
    if storage == STORAGE_FORMAT:
        manifest = old
        validate_source(directory, manifest)
        # Resume cleanup safely after a previous interrupted migration.
        for item in manifest["slices"]:
            if not item.get("file"):
                continue
            path = legacy_path(directory, item)
            if path.exists():
                before = source_signature(path)
                if legacy_digest(path) != (item["edge_count"], item["sha256"]):
                    raise ValueError("Legacy slice differs from published index: " + str(path))
                if source_signature(path) != before:
                    raise ValueError("Legacy file changed during verification")
                validated_stats[path] = before
    else:
        source_name = old["source_file"]
        if Path(source_name).name != source_name:
            raise ValueError("Unsafe source filename")
        source = directory.parent.parent / source_name
        if len(paths) != len(old["slices"]):
            raise ValueError("Missing legacy filenames")
        print("  indexing source", source, flush=True)
        indexed, source_info = index_windows(source, old["slices"])
        for field in ("source_edge_count", "first_timestamp", "last_timestamp"):
            if source_info[field] != old[field]:
                raise ValueError("Source and legacy metadata differ: " + field)
        for number, (previous, current) in enumerate(zip(old["slices"], indexed), 1):
            path = legacy_path(directory, previous)
            before = source_signature(path)
            if previous["edge_count"] != current["edge_count"]:
                raise ValueError("Window record count differs: " + str(path))
            if legacy_digest(path) != (current["edge_count"], current["sha256"]):
                raise ValueError("Window contents/order differ: " + str(path))
            if source_signature(path) != before:
                raise ValueError("Legacy file changed during verification")
            validated_stats[path] = before
            if number % 25 == 0 or number == len(indexed):
                print("  verified slices %d/%d" % (number, len(indexed)), flush=True)
        manifest = dict(old, storage_format=STORAGE_FORMAT,
                        source_signature=source_info["source_signature"], slices=indexed)
        validate_source(directory, manifest)
        # Metadata only; retained for audit, not a backup of duplicated CSV data.
        backup = directory / "legacy-metadata.json"
        if not backup.exists():
            atomic_manifest(backup, old)
        atomic_manifest(manifest_path, manifest)

    # Only explicitly listed, verified files are removed. No directory deletion.
    released = removed = 0
    for path, signature in validated_stats.items():
        validate_source(directory, manifest)
        if source_signature(path) != signature:
            raise ValueError("Legacy slice changed before cleanup: " + str(path))
        path.unlink()
        released += signature["size_bytes"]
        removed += 1
    return {"configuration": str(directory), "slices": len(manifest["slices"]),
            "removed_csv_files": removed, "released_bytes": released}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    configurations = sorted(args.data_root.glob("*/time_slices/*/metadata.json"))
    print("START configurations=%d" % len(configurations), flush=True)
    failures = []
    released = 0
    for i, path in enumerate(configurations, 1):
        start = time.monotonic()
        print("[%d/%d] START %s" % (i, len(configurations), path.parent), flush=True)
        try:
            result = migrate_configuration(path.parent)
            released += result["released_bytes"]
            result["elapsed_s"] = round(time.monotonic() - start, 2)
            print("DONE " + json.dumps(result), flush=True)
        except Exception as error:
            failures.append(str(path.parent))
            print("FAILED %s: %s" % (path.parent, error), flush=True)
    print("SUMMARY " + json.dumps(
        {"total": len(configurations), "failed": failures, "released_bytes": released}
    ), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
