#!/usr/bin/env python3
"""Launch the four missing Ours trainings sequentially on cuda:0."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = {
    "reddit": "data/reddit/time_slices/step_86400_window_259200",
    "lastfm": "data/lastfm/time_slices/step_604800_window_2419200",
    "sx-askubuntu": "data/sx-askubuntu/time_slices/step_604800_window_2419200",
    "sx-superuser": "data/sx-superuser/time_slices/step_604800_window_2419200",
}
MIN_FREE_BYTES = 4 * 1024 ** 3


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(run_dir, state):
    path = run_dir / "status.json"
    temporary = run_dir / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate(name, slices_dir, output, snapshot_count):
    from datasets.coreness_prediction_builder import prediction_split_config
    from methods.hybrid_coreness import load_hybrid_coreness_model
    import torch

    model, checkpoint = load_hybrid_coreness_model(output, device="cpu")
    expected = prediction_split_config(snapshot_count)
    if checkpoint.get("split_config") != expected:
        raise ValueError("{}: checkpoint split does not match 70/15/15".format(name))
    feature = checkpoint.get("feature_config", {})
    if (feature.get("split_rule") != expected["split_rule"]
            or feature.get("numeric_dtype") != "float32"):
        raise ValueError("{}: checkpoint feature protocol mismatch".format(name))
    metadata = json.loads((slices_dir / "snapshot_cache/metadata.json").read_text())
    if (model.kmax, model.hmax) != (metadata["kmax"], metadata["hmax"]):
        raise ValueError("{}: checkpoint dimensions do not match snapshots".format(name))
    if not all(torch.isfinite(tensor).all() for tensor in checkpoint["state_dict"].values()):
        raise ValueError("{}: checkpoint has nonfinite weights".format(name))
    for split in ("validation", "test"):
        if not all(math.isfinite(checkpoint["metrics"][split][key])
                   for key in ("loss", "accuracy", "mae")):
            raise ValueError("{}: nonfinite metrics".format(name))
    if not output.with_suffix(".json").is_file():
        raise ValueError("{}: missing metrics JSON".format(name))
    return {
        "checkpoint": str(output),
        "best_epoch": checkpoint["training_config"]["best_epoch"],
        "validation_mae": checkpoint["metrics"]["validation"]["mae"],
        "test_mae": checkpoint["metrics"]["test"]["mae"],
    }


def launch():
    from datasets.coreness_prediction_builder import prediction_split_config
    from datasets.dataset_builder import load_time_slice_manifest
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 is required for Ours training")
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 4 GiB free; refusing to launch")

    jobs = {}
    for name, relative in DATASETS.items():
        slices = ROOT / relative
        count = len(load_time_slice_manifest(slices)["slices"])
        jobs[name] = {
            "status": "queued",
            "slices_dir": str(slices),
            "split_config": prediction_split_config(count),
        }

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(
        prefix="ours_70_15_15_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_",
        dir=results,
    )).resolve()
    _save(run_dir, {
        "status": "scheduled", "created_at": _now(),
        "device": "cuda:0", "order": list(jobs), "jobs": jobs,
    })
    with (run_dir / "driver.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            ["nohup", sys.executable, "-u", __file__, "run", str(run_dir)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    print("{} pid={}".format(run_dir, process.pid), flush=True)


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    with (run_dir / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((run_dir / "status.json").read_text())
        if state["status"] != "scheduled":
            raise RuntimeError("run already started: {}".format(run_dir))
        state.update(status="running", driver_pid=os.getpid(), started_at=_now())
        _save(run_dir, state)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        for name in state["order"]:
            job = state["jobs"][name]
            slices = Path(job["slices_dir"])
            output = run_dir / (name + ".pt")
            log_path = run_dir / (name + ".log")
            command = [
                sys.executable, "-u", "-m", "training.ours", str(slices),
                "--output", str(output), "--device", "cuda:0",
                "--train-ratio", "0.7", "--val-ratio", "0.15",
            ]
            job.update(command=command, log=str(log_path), started_at=_now())
            try:
                free = shutil.disk_usage(run_dir).free
                if free < MIN_FREE_BYTES:
                    raise RuntimeError("less than 4 GiB free ({} bytes)".format(free))
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                    job.update(status="running", pid=process.pid)
                    _save(run_dir, state)
                    print("{} {} started pid={} free={} bytes".format(
                        _now(), name, process.pid, free), flush=True)
                    code = process.wait()
                job["returncode"] = code
                if code:
                    raise RuntimeError("training exited with code {}".format(code))
                job.update(_validate(name, slices, output,
                                     job["split_config"]["snapshot_count"]))
                job["status"] = "complete"
            except Exception as error:
                traceback.print_exc()
                job.update(status="failed", error="{}: {}".format(
                    type(error).__name__, error))
            job["finished_at"] = _now()
            _save(run_dir, state)
            print("{} {} {}".format(_now(), name, job["status"]), flush=True)

        state.update(status="complete" if all(
            job["status"] == "complete" for job in state["jobs"].values()
        ) else "finished_with_failures", finished_at=_now())
        _save(run_dir, state)
        print("{} batch {}".format(_now(), state["status"]), flush=True)
        return 0 if state["status"] == "complete" else 1


def status(run_dir):
    state = json.loads((Path(run_dir) / "status.json").read_text())
    print("batch={} pid={}".format(state["status"], state.get("driver_pid", "-")))
    for name in state["order"]:
        job = state["jobs"][name]
        print("{}: {} pid={} {}".format(name, job["status"], job.get("pid", "-"),
                                         job.get("error", job.get("checkpoint", ""))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("launch")
    for action in ("run", "status"):
        subparsers.add_parser(action).add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "launch":
        launch()
    elif args.command == "status":
        status(args.run_dir)
    else:
        return run(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
