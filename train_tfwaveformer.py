"""Train TFWaveFormer for next-snapshot links using the shared snapshot cache."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

from methods.tfwaveformer import DEFAULT_ROOT, load_runtime, load_snapshot_edges
from datasets.baseline_split import training_boundaries


def sample_negatives(nodes, positive_edges, count, rng):
    """Sample undirected nonedges with replacement, using observed nodes only."""
    nodes = np.asarray(sorted(nodes), dtype=np.int64)
    positive = {tuple(pair) for pair in positive_edges.tolist()}
    node_set = set(nodes.tolist())
    excluded = sum(u in node_set and v in node_set for u, v in positive)
    if len(nodes) * (len(nodes) - 1) // 2 <= excluded:
        raise ValueError("no negative pairs among historically observed nodes")
    result = []
    for _ in range(100):
        candidates = rng.choice(nodes, size=(max(64, 2 * (count - len(result))), 2))
        candidates.sort(axis=1)
        result.extend((int(u), int(v)) for u, v in candidates
                      if u != v and (int(u), int(v)) not in positive)
        if len(result) >= count:
            return np.asarray(result[:count], dtype=np.int64)
    raise ValueError("negative sampling exhausted; observed graph is too dense")


def split_boundaries(count, train_end_t=None, val_end_t=None):
    default_train, default_val = training_boundaries(count)
    train_end = default_train if train_end_t is None else train_end_t
    val_end = default_val if val_end_t is None else val_end_t
    if not 1 <= train_end < val_end < count - 1:
        raise ValueError("require 1 <= train_end_t < val_end_t < snapshot_count - 1")
    return train_end, val_end


def _atomic_json(path, payload):
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _save_checkpoint(path, payload):
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def run_epoch(runtime, model, snapshots, config, train_end, val_end, *,
              device, batch_size, seed, optimizer=None, max_edges=None):
    history = runtime.SnapshotHistory(config.max_input_sequence_length)
    model[0].set_neighbor_sampler(history)
    results = {"train": {"loss_sum": 0., "count": 0, "snapshots": 0},
               "val": {"loss_sum": 0., "count": 0, "snapshots": 0}}
    val_scores, val_labels = [], []
    history.observe(snapshots[0], 0)
    for target in range(1, val_end + 1):
        positive = runtime.edge_array(snapshots[target], canonical=True)
        split = "train" if target <= train_end else "val"
        training = split == "train" and optimizer is not None
        model.train(training)
        # Validation sampling is independent of epoch and training RNG consumption.
        rng = np.random.RandomState((seed if split == "train" else 2026) + target)
        selected = positive
        if max_edges is not None and len(selected) > max_edges:
            selected = selected[rng.choice(len(selected), size=max_edges, replace=False)]
        if len(selected):
            negative = sample_negatives(history.mapping, positive, len(selected), rng)
            pairs = np.concatenate((selected, negative))
            labels = np.concatenate((np.ones(len(selected), dtype=np.float32),
                                     np.zeros(len(selected), dtype=np.float32)))
            order = rng.permutation(len(pairs))
            pairs, labels = pairs[order], labels[order]
            results[split]["snapshots"] += 1
            for start in range(0, len(pairs), batch_size):
                edges = pairs[start:start + batch_size]
                y = torch.as_tensor(labels[start:start + batch_size], device=device)
                with torch.set_grad_enabled(training):
                    logits = runtime.edge_logits(model, history, edges)
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("non-finite loss at target {}".format(target))
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if not torch.isfinite(norm).item():
                            raise FloatingPointError("non-finite gradient")
                        optimizer.step()
                results[split]["loss_sum"] += loss.item() * len(edges)
                results[split]["count"] += len(edges)
                if split == "val":
                    val_scores.append(logits.detach().sigmoid().cpu().numpy())
                    val_labels.append(labels[start:start + batch_size])
        # No target edge is visible to any minibatch of this snapshot.
        history.observe(positive, target)
    for split, values in results.items():
        if not values["count"]:
            raise ValueError("{} split has no labeled examples".format(split))
        values["loss"] = values.pop("loss_sum") / values["count"]
    labels, scores = np.concatenate(val_labels), np.concatenate(val_scores)
    results["val"]["ap"] = float(average_precision_score(labels, scores))
    results["val"]["auc"] = float(roc_auc_score(labels, scores))
    return results, history.digest.hexdigest()


def train(snapshots, runtime, config, output_dir, *, train_end_t=None, val_end_t=None,
          epochs=100, patience=5, batch_size=128, learning_rate=1e-4,
          weight_decay=0., seed=2024, device="cpu", max_edges=None, source=None):
    for name, value in (("epochs", epochs), ("patience", patience), ("batch_size", batch_size)):
        runtime.integer(value, name, 1)
    if max_edges is not None:
        runtime.integer(max_edges, "max_edges", 1)
    runtime.integer(seed, "seed")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    train_end, val_end = split_boundaries(len(snapshots), train_end_t, val_end_t)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        protocol=runtime.PROTOCOL_VERSION, source=source, config=asdict(config),
        train_end_t=train_end, val_end_t=val_end, fit_end_t=val_end,
        first_test_target_t=val_end + 1, snapshot_count=len(snapshots),
        train_target_range=[1, train_end], val_target_range=[train_end + 1, val_end],
        split_rule="chronological snapshot boundaries; snapshot 0 is history only",
        attributes="zero_float32", undirected="mean_logits",
        negative_sampling="uniform observed-node nonedges with replacement, 1:1",
        epochs=epochs, patience=patience, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, seed=seed,
        device=str(device), max_edges_per_snapshot=max_edges,
        test_scored=False, status="running",
    )
    _atomic_json(output_dir / "run.json", metadata)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = runtime.build_model(config, runtime.SnapshotHistory(config.max_input_sequence_length), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_ap, stale = -1., 0
        for epoch in range(epochs):
            metrics, prefix = run_epoch(
                runtime, model, snapshots, config, train_end, val_end,
                device=device, batch_size=batch_size, seed=seed + epoch,
                optimizer=optimizer, max_edges=max_edges,
            )
            improved = metrics["val"]["ap"] > best_ap
            if improved:
                best_ap, stale = metrics["val"]["ap"], 0
                checkpoint = dict(
                    protocol=runtime.PROTOCOL_VERSION, config=asdict(config), fit_end_t=val_end,
                    fit_history_sha256=prefix, epoch=epoch, val_metrics=metrics["val"],
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    training=metadata.copy(),
                )
                _save_checkpoint(output_dir / "best.pt", checkpoint)
                metadata.update(best_epoch=epoch, best_val_ap=best_ap)
            else:
                stale += 1
            record = dict(epoch=epoch, improved=improved, **metrics)
            with (output_dir / "epochs.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            metadata.update(completed_epochs=epoch + 1)
            _atomic_json(output_dir / "run.json", metadata)
            if stale >= patience:
                break
        metadata.update(status="completed", checkpoint=str((output_dir / "best.pt").resolve()))
        _atomic_json(output_dir / "run.json", metadata)
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        _atomic_json(output_dir / "run.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir", type=Path)
    parser.add_argument("--tfwaveformer-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="new directory; existing directories are never overwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--train-end-t", type=int)
    parser.add_argument("--val-end-t", type=int)
    parser.add_argument("--max-edges-per-snapshot", type=int, help="optional subsampling for smoke runs; default uses all edges")
    parser.add_argument("--feature-dim", type=int, default=172)
    parser.add_argument("--time-feat-dim", type=int, default=100)
    parser.add_argument("--channel-embedding-dim", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-input-sequence-length", type=int, default=32)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    runtime = load_runtime(args.tfwaveformer_root)
    config = runtime.SnapshotConfig(**{
        name: getattr(args, name) for name in runtime.SnapshotConfig.__dataclass_fields__
    })
    slices = args.slices_dir.resolve()
    output = args.output_dir or slices / "tfwaveformer_runs" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    metadata = train(
        load_snapshot_edges(slices), runtime, config, output,
        train_end_t=args.train_end_t, val_end_t=args.val_end_t,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed,
        device=args.device, max_edges=args.max_edges_per_snapshot, source=str(slices),
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
