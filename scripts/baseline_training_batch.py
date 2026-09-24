#!/usr/bin/env python3
"""Run selected snapshot baseline trainings sequentially on one GPU."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "third_party"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS = {
    "lastfm": ("lastfm", "step_604800_window_2419200"),
    "reddit": ("reddit", "step_86400_window_259200"),
    "sx-askubuntu": ("sx-askubuntu", "step_604800_window_2419200"),
    "sx-superuser": ("sx-superuser", "step_604800_window_2419200"),
    "mooc": ("mooc", "step_43200_window_86400"),
    "email": ("email-Eu-core-temporal", "step_302400_window_604800"),
    "wiki-talk-temporal": ("wiki-talk-temporal", "step_259200_window_604800"),
    "sx-stackoverflow": ("sx-stackoverflow", "step_604800_window_2419200"),
    "tgbl-coin": ("tgbl-coin", "step_86400_window_86400"),
}
DEFAULT_DATASETS = ("lastfm", "reddit", "sx-askubuntu", "sx-superuser")
METHODS = ("eagle", "zebra", "swift", "tfwaveformer", "prism")
MIN_FREE_BYTES = 4 * 1024 ** 3

from datasets.baseline_split import CURRENT_SPLIT, LEGACY_SPLIT, training_boundaries


def _write_json(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _update(path, **fields):
    payload = json.loads(Path(path).read_text())
    payload.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
    _write_json(path, payload)
    return payload


def _run(command, cwd, *, env=None):
    print("command:", json.dumps(list(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(cwd), env=env, check=True)


def _train(job, job_path):
    method, dataset = job["method"], job["dataset"]
    split_rule = job.get("split_rule", LEGACY_SPLIT)
    train_end, fit_end = training_boundaries(job["snapshot_count"], split_rule)
    slices = Path(job["slices_dir"])
    batch = Path(job["batch_dir"])
    output = batch / "artifacts" / method / dataset
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               NUMBA_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")

    if method == "eagle":
        from scripts.launch_eagle_time_training import prepare_run

        run, metadata = prepare_run(dataset, BASELINES / "EAGLE", output.parent,
                                    epochs=50, gpu=0, split_rule=split_rule)
        _update(job_path, artifact=str(run), checkpoint=metadata["checkpoint"])
        _run(metadata["command"][1:], run, env=env)
        if not Path(metadata["checkpoint"]).is_file():
            raise FileNotFoundError("EAGLE did not produce its trained checkpoint")
    elif method == "zebra":
        zebra_root = BASELINES / "Zebra"
        zebra_data = "{}-snapshot-{}-{}".format(
            dataset, "70_15_15" if split_rule == CURRENT_SPLIT else "7x3", batch.name)
        _run([sys.executable, "-u", zebra_root / "utils/preprocess_time_slices.py",
              "--input", slices, "--data", zebra_data], ROOT, env=env)
        _update(job_path, zebra_dataset=zebra_data,
                artifact=str(zebra_root / "data" / zebra_data))
        _run([sys.executable, "-u", "train.py", "--data", zebra_data,
              "--snapshot-count", job["snapshot_count"], "--n_epoch", "50",
              "--bs", "200", "--lr", "0.0001", "--patience", "5",
              "--n_runs", "1", "--n_degree", "10", "--n_layer", "2",
              "--n_head", "2", "--node_dim", "100", "--time_dim", "100",
              "--memory_dim", "100", "--drop_out", "0.3",
              "--message_function", "identity", "--memory_updater", "gru",
              "--aggregator", "last", "--tppr_strategy", "streaming",
              "--topk", "20", "--alpha_list", "0.1", "0.1",
              "--beta_list", "0.5", "0.95", "--gpu", "0", "--save_best",
              "--snapshot-split-rule", split_rule],
             zebra_root, env=env)
        sidecars = list((zebra_root / "saved_checkpoints").glob(
            zebra_data + "*.pth.split.json"))
        if len(sidecars) != 1 or not Path(str(sidecars[0])[:-len(".split.json")]).is_file():
            raise FileNotFoundError("Zebra checkpoint and snapshot split sidecar were not produced")
        _update(job_path, checkpoint=str(sidecars[0])[:-len(".split.json")])
    elif method == "swift":
        _run(["bash", BASELINES / "SWIFT/run_local.sh",
              BASELINES / "SWIFT/snapshot_adapter.py", slices,
              "--output", output, "--model", "TGAT", "--epochs", "5",
              "--train-end", train_end + 1, "--val-end", fit_end + 1,
              "--split-rule", split_rule],
             ROOT, env=env)
        if not (output / "best.pt").is_file():
            raise FileNotFoundError("SWIFT did not produce its trained checkpoint")
        _update(job_path, artifact=str(output), checkpoint=str(output / "best.pt"))
    elif method == "tfwaveformer":
        _run([sys.executable, "-u", "-m", "training.baselines.tfwaveformer", slices,
              "--output-dir", output, "--device", "cuda:0", "--epochs", "30",
              "--patience", "5", "--threads", "2", "--train-end-t", train_end,
              "--val-end-t", fit_end, "--split-rule", split_rule], ROOT, env=env)
        if not (output / "best.pt").is_file():
            raise FileNotFoundError("TFWaveFormer did not produce its trained checkpoint")
        _update(job_path, artifact=str(output), checkpoint=str(output / "best.pt"))
    elif method == "prism":
        _run([sys.executable, "-u", "-m", "training.baselines.prism", slices,
              "--output-dir", output, "--device", "cuda:0", "--epochs", "10",
              "--train-end", train_end, "--val-end", fit_end,
              "--split-rule", split_rule],
             ROOT, env=env)
        if not (output / "best.pt").is_file():
            raise FileNotFoundError("PRISM did not produce its trained checkpoint")
        _update(job_path, artifact=str(output), checkpoint=str(output / "best.pt"))
    else:
        raise ValueError("unknown method: " + method)


def worker(job_path):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    batch = Path(job["batch_dir"])
    try:
        free = shutil.disk_usage(batch).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError("insufficient disk space: {} bytes free".format(free))
        _update(job_path, status="running", pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat())
        print("running on cuda:0: {} {} ({} bytes free)".format(
            job["method"], job["dataset"], free), flush=True)
        _train(job, job_path)
        _update(job_path, status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        print("complete: {} {}".format(job["method"], job["dataset"]), flush=True)
    except BaseException as error:
        traceback.print_exc()
        _update(job_path, status="failed", error="{}: {}".format(
            type(error).__name__, error), finished_at=datetime.now(timezone.utc).isoformat())
        raise


def run_batch(batch):
    batch = Path(batch).resolve()
    manifest_path = batch / "manifest.json"
    manifest = _update(manifest_path, status="running", supervisor_pid=os.getpid())
    for path in manifest["jobs"]:
        job = json.loads(Path(path).read_text())
        with Path(job["log"]).open("ab", buffering=0) as log:
            process = subprocess.run(
                [sys.executable, "-u", __file__, "worker", str(path)],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if process.returncode and json.loads(Path(path).read_text())["status"] != "failed":
            _update(path, status="failed", error="worker exited {}".format(
                process.returncode), finished_at=datetime.now(timezone.utc).isoformat())
        print("{} {}: {}".format(job["method"], job["dataset"], process.returncode),
              flush=True)
    jobs = [json.loads(Path(path).read_text()) for path in manifest["jobs"]]
    _update(manifest_path, status="complete" if all(
        job["status"] == "complete" for job in jobs) else "finished_with_failures",
        finished_at=datetime.now(timezone.utc).isoformat())


def _select(requested, available, label):
    selected = tuple(requested) if requested else tuple(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError("unknown {}: {}".format(label, ", ".join(unknown)))
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate {} selection".format(label))
    return selected


def launch(dataset_names=None, method_names=None, *, start=True,
           split_rule=CURRENT_SPLIT):
    from datasets.baseline_eval import load_test_samples
    from datasets.baseline_split import test_start_t
    from datasets.dataset_builder import load_time_slice_manifest
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 is required for all training jobs")

    dataset_names = _select(dataset_names, DATASETS, "dataset")
    method_names = _select(method_names, METHODS, "method")
    counts = {}
    for name in dataset_names:
        source_name, configuration = DATASETS[name]
        slices = ROOT / "data" / source_name / "time_slices" / configuration
        manifest = load_time_slice_manifest(slices)
        count = len(manifest["slices"])
        training_boundaries(count, split_rule)
        load_test_samples(slices, count)
        counts[name] = (slices, count, test_start_t(count))
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    label = "baseline_70_15_15_" if split_rule == CURRENT_SPLIT else "baseline_7x3_"
    batch = Path(tempfile.mkdtemp(prefix=label +
                   datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=results))
    (batch / "jobs").mkdir()
    (batch / "logs").mkdir()
    jobs = []
    for method in method_names:
        for dataset, (slices, count, start) in counts.items():
            name = "{}__{}".format(method, dataset)
            path = batch / "jobs" / (name + ".json")
            log = batch / "logs" / (name + ".log")
            _write_json(path, dict(method=method, dataset=dataset,
                        snapshot_count=count, test_start_t=start,
                        slices_dir=str(slices), batch_dir=str(batch),
                        log=str(log), split_rule=split_rule, status="scheduled"))
            jobs.append(path)
    manifest_path = batch / "manifest.json"
    _write_json(manifest_path, dict(protocol=split_rule,
                datasets=list(dataset_names), methods=list(method_names),
                jobs=[str(job) for job in jobs], status="scheduled",
                created_at=datetime.now(timezone.utc).isoformat()))
    if start:
        with (batch / "supervisor.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(["nohup", sys.executable, "-u", __file__,
                                        "run-batch", str(batch)], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print("{} pid={}".format(batch, process.pid), flush=True)
    return batch


def status(batch):
    batch = Path(batch)
    manifest = json.loads((batch / "manifest.json").read_text())
    for path in manifest["jobs"]:
        job = json.loads(Path(path).read_text())
        print("{} {:15} {:17} pid={} {}".format(
            job["status"], job["method"], job["dataset"], job.get("pid", "-"),
            job.get("error", job.get("checkpoint", ""))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                               default=list(DEFAULT_DATASETS))
    launch_parser.add_argument("--methods", nargs="+", choices=sorted(METHODS),
                               default=list(METHODS))
    sub.add_parser("run-batch").add_argument("batch")
    sub.add_parser("worker").add_argument("job")
    sub.add_parser("status").add_argument("batch")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.datasets, args.methods)
    elif args.command == "run-batch":
        run_batch(args.batch)
    elif args.command == "worker":
        worker(args.job)
    else:
        status(args.batch)


if __name__ == "__main__":
    main()
