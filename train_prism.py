"""Chronological PRISM training on the shared next-snapshot edge task."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from methods.prism import load_runtime, load_snapshots
from datasets.baseline_split import training_boundaries


def split_boundaries(n, train_end=None, val_end=None):
    # Community eval samples use current t >= floor(.7*N). Leave that range
    # entirely outside both training and validation targets by default.
    default_train, default_val = training_boundaries(n)
    train_end = default_train if train_end is None else train_end
    val_end = default_val if val_end is None else val_end
    if not 1 <= train_end < val_end < n - 1:
        raise ValueError("require 1 <= train_end < val_end < snapshot_count - 1")
    return train_end, val_end


def sample_negatives(nodes, positives, n, rng):
    nodes = np.asarray(sorted(nodes), dtype=np.int64)
    positive = set(map(tuple, positives))
    node_set = set(nodes.tolist())
    if len(nodes) * (len(nodes) - 1) // 2 <= sum(u in node_set and v in node_set
                                                  for u, v in positive):
        raise ValueError("no historical-node nonedges available")
    result = []
    for _ in range(max(1000, n * 100)):
        left, right = rng.randint(len(nodes), size=2)
        if left == right:
            continue
        a, b = nodes[left], nodes[right]
        edge = tuple(sorted((int(a), int(b))))
        if edge not in positive:
            result.append(edge)
            if len(result) == n:
                return np.asarray(result, dtype=np.int64).reshape(-1, 2)
    raise ValueError("negative sampling exhausted on a dense snapshot")


def fit(snapshots, capacity, output, config, *, train_end, val_end,
        epochs=10, batch_size=128, device="cpu", seed=2026, max_edges=None):
    runtime = load_runtime()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = runtime.PrismSnapshotModel(capacity, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    best = float("inf")
    for epoch in range(epochs):
        history = runtime.SnapshotHistory(capacity)
        model.reset_history()
        initial = history.observe(snapshots[0], 0)
        model.observe(initial, history, 0)
        totals = {"train": [0., 0], "val": [0., 0]}
        for target in range(1, val_end + 1):
            split = "train" if target <= train_end else "val"
            training = split == "train"
            model.train(training)
            positives = runtime.edge_array(snapshots[target])
            rng = np.random.RandomState((seed + epoch if training else 2026) + target)
            selected = positives
            if max_edges is not None and len(selected) > max_edges:
                selected = selected[rng.choice(len(selected), max_edges, replace=False)]
            if len(selected) and len(history.mapping) > 1:
                negatives = sample_negatives(history.mapping, positives, len(selected), rng)
                pairs = np.concatenate((selected, negatives))
                labels = np.concatenate((np.ones(len(selected), dtype=np.float32),
                                         np.zeros(len(selected), dtype=np.float32)))
                order = rng.permutation(len(pairs))
                pairs, labels = pairs[order], labels[order]
                # One optimizer step per target keeps all candidate batches on exactly
                # the same history state. Never ingest target positives mid-scoring.
                if training:
                    optimizer.zero_grad()
                for i in range(0, len(pairs), batch_size):
                    with torch.set_grad_enabled(training):
                        prob = model.probabilities(pairs[i:i + batch_size], history)
                        label = torch.as_tensor(labels[i:i + batch_size], device=device)
                        loss = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), label)
                        if training:
                            (loss * (len(label) / len(pairs))).backward(
                                retain_graph=i + batch_size < len(pairs))
                    totals[split][0] += float(loss.detach()) * len(label)
                    totals[split][1] += len(label)
                if training:
                    optimizer.step()
            # Truncated BPTT across boundaries, but preserve gradient from the
            # immediately preceding snapshot's PRISM memory update.
            model.state = model.state.detach()
            observed = history.observe(positives, target)
            with torch.set_grad_enabled(target < train_end):
                model.observe(observed, history, target)
        val = totals["val"][0] / max(1, totals["val"][1])
        report = {"epoch": epoch + 1, "train_loss": totals["train"][0] /
                  max(1, totals["train"][1]), "val_loss": val,
                  "train_edges": totals["train"][1] // 2,
                  "val_edges": totals["val"][1] // 2}
        print(json.dumps(report), flush=True)
        if val < best:
            best = val
            torch.save({"protocol": runtime.PROTOCOL, "config": asdict(config),
                        "capacity": capacity, "train_end_t": train_end,
                        "val_end_t": val_end, "fit_end_t": val_end,
                        "prefix_digest": history.digest.hexdigest(),
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       str(output / "best.pt"))
    (output / "run.json").write_text(json.dumps({"train_end_t": train_end,
                        "val_end_t": val_end, "best_val_loss": best,
                        "max_edges_per_snapshot": max_edges}, indent=2) + "\n")
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--val-end", type=int)
    parser.add_argument("--max-edges-per-snapshot", type=int,
                        help="smoke-only scoring cap; history still observes complete snapshots")
    parser.add_argument("--mem-dim", type=int, default=100)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--m-pass", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.max_edges_per_snapshot is not None
                                                   and args.max_edges_per_snapshot < 1):
        parser.error("epochs, batch-size and scoring cap must be positive")
    runtime = load_runtime()
    snapshots, capacity, _ = load_snapshots(args.slices_dir)
    train_end, val_end = split_boundaries(len(snapshots), args.train_end, args.val_end)
    config = runtime.Config(args.mem_dim, args.time_dim, args.emb_dim,
                            args.m_pass, args.neighbors)
    output = args.output_dir or str(Path(args.slices_dir) / "prism_runs" /
                                    datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    print(fit(snapshots, capacity, output, config, train_end=train_end,
              val_end=val_end, epochs=args.epochs, batch_size=args.batch_size,
              device=args.device, max_edges=args.max_edges_per_snapshot))


if __name__ == "__main__":
    main()
