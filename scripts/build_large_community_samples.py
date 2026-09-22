"""Build evaluation sample caches after large snapshot builds finish."""

import argparse
import shutil
import time
from pathlib import Path

from datasets.community_eval_builder import (
    DEFAULT_MAX_PER_KT,
    DEFAULT_SEED,
    sample_qk_coreness_weighted,
)
from datasets.dataset_builder import (
    DATASET_VALID_KS,
    DEFAULT_TEST_RATIO,
    TARGET_KS,
    build_snapshots,
    load_time_slice_manifest,
)


def _builder_running(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as source:
            return b"scripts.build_large_snapshot_caches" in source.read()
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-pid", type=int, required=True)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("slices_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    for directory in args.slices_dirs:
        index = directory / "snapshot_cache" / "partitioned" / "index.json"
        print(f"WAIT {directory}", flush=True)
        while not index.is_file():
            if not _builder_running(args.snapshot_pid):
                raise RuntimeError(f"Snapshot build stopped before publishing {directory}")
            time.sleep(30)

        available = shutil.disk_usage(directory).free / 1024 ** 3
        if available < args.min_free_gib:
            raise RuntimeError(f"Only {available:.1f} GiB free before sampling {directory}")
        started = time.monotonic()
        snapshots, *_ = build_snapshots(directory)
        manifest = load_time_slice_manifest(directory)
        dataset = manifest["dataset"]
        split = int(len(snapshots) * (1 - DEFAULT_TEST_RATIO))
        samples = sample_qk_coreness_weighted(
            snapshots, split, DATASET_VALID_KS.get(dataset, TARGET_KS),
            DEFAULT_MAX_PER_KT, seed=DEFAULT_SEED, dataset_name=dataset,
            cache_dir=directory / "sample_cache",
        )
        cache_files = list((directory / "sample_cache").glob("*.pkl"))
        print(
            f"DONE {dataset}: {len(samples)} samples, "
            f"{sum(bool(s['community']) for s in samples)} nonempty, "
            f"cache={sum(p.stat().st_size for p in cache_files) / 1024 ** 2:.1f} MiB, "
            f"elapsed={time.monotonic() - started:.1f}s, "
            f"free={shutil.disk_usage(directory).free / 1024 ** 3:.1f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
