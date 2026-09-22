"""Export snapshot-time events and launch isolated EAGLE-Time nohup jobs."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.dataset_builder import load_slice_edges, load_time_slice_manifest
from datasets.baseline_split import training_boundaries


DATASETS = {
    "mooc": ("mooc", 43200, 86400),
    "email": ("email-Eu-core-temporal", 302400, 604800),
    "lastfm": ("lastfm", 604800, 2419200),
    "reddit": ("reddit", 86400, 259200),
    "sx-askubuntu": ("sx-askubuntu", 604800, 2419200),
    "sx-superuser": ("sx-superuser", 604800, 2419200),
}


def prepare_run(name, eagle_root, output_root, epochs=100, gpu=0):
    dataset, step, window = DATASETS[name]
    slices_dir = ROOT / "data" / dataset / "time_slices" / (
        "step_{}_window_{}".format(step, window)
    )
    manifest = load_time_slice_manifest(slices_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(
        prefix="{}-{}-".format(name, datetime.now().strftime("%Y%m%d_%H%M%S")),
        dir=str(output_root),
    ))
    csv_path = run / "snapshot_events.csv"
    counts, sources, destinations = [], [], []
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("u", "v", "ts"))
        for t, info in enumerate(manifest["slices"]):
            if info["index"] != t:
                raise ValueError("Snapshot indices must be consecutive from zero")
            edges = sorted({
                (min(u, v), max(u, v))
                for u, v in load_slice_edges(slices_dir, manifest, info) if u != v
            })
            counts.append(len(edges))
            writer.writerows((u, v, t + 1) for u, v in edges)
            sources.extend(u for u, _ in edges)
            destinations.extend(v for _, v in edges)
    timestamps = np.repeat(np.arange(1, len(counts) + 1), counts)
    train_end_t, fit_end_t = training_boundaries(len(counts))
    val_time, test_time = train_end_t + 1, fit_end_t + 1
    masks = (timestamps <= val_time,
             (timestamps > val_time) & (timestamps <= test_time),
             timestamps > test_time)
    if not all(mask.any() for mask in masks):
        raise ValueError("Nonempty chronological train/validation/test splits required")
    # Match the external loader's shared-ID factorization for reproducibility.
    _, ids = pd.factorize(np.asarray(sources + destinations, dtype=np.int64), sort=False)
    with (run / "training_node_mapping.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("original_id", "training_id"))
        writer.writerows((int(node), index) for index, node in enumerate(ids))

    config = dict(branch="time", structure_topk=100, alpha=0.9, beta=0.8,
                  time_topk=15, time_channels=100, num_layers=1, yita=None,
                  fit_end_t=fit_end_t)
    (run / "inference_config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset_name = name + "-snapshot"
    checkpoint = run / "saved_time_models" / "learn_time" / dataset_name / (
        "topk_15_flag_last_lr_0.001_wd_5e-05_bs_200.pth"
    )
    command = [
        "nohup", sys.executable, "-u", str(eagle_root / "link_prediction/train_time.py"),
        "--dataset_name", dataset_name, "--data_path", str(csv_path),
        "--split-times", str(val_time), str(test_time),
        "--gpu", str(gpu), "--topk", "15", "--topk_sample_flag", "last",
        "--hidden_dims", "100", "--num_layers", "1", "--batch_size", "200",
        "--num_epochs", str(epochs), "--patience", "5",
        "--lr", "0.001", "--weight_decay", "5e-5", "--seed", "2024",
        "--train_only",
    ]
    metadata = dict(
        dataset=dataset, slices_dir=str(slices_dir), protocol="eagle_snapshot_v1",
        snapshot_count=len(counts), snapshot_edge_counts=counts,
        events=len(timestamps), nodes=len(ids),
        destination_nodes=len(set(destinations)),
        train_events=int(masks[0].sum()), val_events=int(masks[1].sum()),
        held_out_test_events=int(masks[2].sum()),
        split_rule="snapshot_55_15_30_v1; equal times stay together",
        train_end_t=train_end_t, test_start_t=fit_end_t,
        val_time=float(val_time), test_time=float(test_time), fit_end_t=fit_end_t,
        first_test_target_t=int(timestamps[masks[2]].min()) - 1,
        time_unit="snapshot_index", history_policy="strictly_before_target_snapshot",
        snapshot_events_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((slices_dir / "metadata.json").read_bytes()).hexdigest(),
        train_script_sha256=hashlib.sha256(
            (eagle_root / "link_prediction/train_time.py").read_bytes()).hexdigest(),
        checkpoint=str(checkpoint), command=command,
        inference_config=str(run / "inference_config.json"),
        log=str(run / "train.log"), run_dir=str(run), status="prepared",
        hybrid_calibrated=False,
    )
    (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run, metadata


def launch(run, metadata):
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                       PYTHONUNBUFFERED="1")
    with (run / "train.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            metadata["command"], cwd=str(run), env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (run / "train.pid").write_text(str(process.pid) + "\n")
    metadata.update(pid=process.pid, status="launched")
    (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--eagle-root", type=Path, default=ROOT.parent / "EAGLE")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    eagle_root = args.eagle_root.resolve()
    output_root = (args.output_root or eagle_root / "log/snapshot-training").resolve()
    for name in args.datasets:
        run, metadata = prepare_run(name, eagle_root, output_root, args.epochs, args.gpu)
        if not args.prepare_only:
            launch(run, metadata)
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
