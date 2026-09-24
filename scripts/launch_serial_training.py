#!/usr/bin/env python3
"""Run Ours followed by all five baselines on one dataset and GPU."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import baseline_training_batch, launch_ours_training

DATASETS = sorted(set(launch_ours_training.DATASETS) & set(baseline_training_batch.DATASETS))


def _save(run_dir, state):
    temporary = run_dir / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(run_dir / "status.json")


def launch(dataset):
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="serial_training_" + dataset + "_",
                                        dir=results)).resolve()
    _save(run_dir, {"dataset": dataset, "status": "scheduled",
                    "stages": {"ours": {"status": "queued"},
                               "baselines": {"status": "queued"}}})
    with (run_dir / "driver.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(["nohup", sys.executable, "-u", __file__,
                                    "run", str(run_dir)], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print("{} pid={}".format(run_dir, process.pid), flush=True)


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    with (run_dir / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((run_dir / "status.json").read_text())
        if state["status"] != "scheduled":
            raise RuntimeError("run already started: {}".format(run_dir))
        state.update(status="running", pid=os.getpid())
        _save(run_dir, state)

        for stage in ("ours", "baselines"):
            info = state["stages"][stage]
            info["status"] = "running"
            _save(run_dir, state)
            try:
                if stage == "ours":
                    directory = launch_ours_training.launch([state["dataset"]], start=False)
                else:
                    directory = baseline_training_batch.launch(
                        [state["dataset"]], start=False,
                        split_rule=baseline_training_batch.CURRENT_SPLIT)
                info["directory"] = str(directory)
                _save(run_dir, state)
                if stage == "ours":
                    launch_ours_training.run(directory)
                else:
                    baseline_training_batch.run_batch(directory)
                result = json.loads((directory / ("status.json" if stage == "ours"
                                                 else "manifest.json")).read_text())
                info["status"] = result["status"]
            except Exception as error:
                traceback.print_exc()
                info.update(status="failed", error="{}: {}".format(type(error).__name__, error))
            info["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save(run_dir, state)
            print("{}: {}".format(stage, info["status"]), flush=True)

        state["status"] = ("complete" if all(
            info["status"] == "complete" for info in state["stages"].values()
        ) else "finished_with_failures")
        _save(run_dir, state)
        return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("launch").add_argument("dataset", choices=DATASETS)
    for command in ("run", "status"):
        commands.add_parser(command).add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.dataset)
    elif args.command == "run":
        return run(args.run_dir)
    else:
        print(json.dumps(json.loads((args.run_dir / "status.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
