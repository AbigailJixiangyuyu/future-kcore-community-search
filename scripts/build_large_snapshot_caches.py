"""Build large snapshot caches sequentially with a free-space safety limit."""

import argparse
import shutil
import signal
import time
from pathlib import Path

from datasets.dataset_builder import build_snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dirs", nargs="+", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    minimum_bytes = int(args.min_free_gib * 1024 ** 3)

    def check_space(_signum=None, _frame=None):
        available = shutil.disk_usage(args.slices_dirs[0]).free
        if available < minimum_bytes:
            raise RuntimeError(
                f"Stopping snapshot build: {available / 1024 ** 3:.1f} GiB "
                f"free, below {args.min_free_gib:.1f} GiB limit"
            )

    signal.signal(signal.SIGALRM, check_space)
    signal.setitimer(signal.ITIMER_REAL, 30, 30)
    try:
        for directory in args.slices_dirs:
            check_space()
            started = time.monotonic()
            print(f"START {directory}", flush=True)
            snapshots, total_nodes, kmax, hmax = build_snapshots(directory)
            check_space()
            cache = directory / "snapshot_cache"
            size = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
            print(
                f"DONE {directory}: {len(snapshots)} snapshots, "
                f"{total_nodes} nodes, kmax={kmax}, hmax={hmax}, "
                f"cache={size / 1024 ** 3:.2f} GiB, "
                f"elapsed={time.monotonic() - started:.1f}s, "
                f"free={shutil.disk_usage(directory).free / 1024 ** 3:.1f} GiB",
                flush=True,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
