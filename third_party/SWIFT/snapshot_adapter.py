"""Snapshot-boundary link prediction with SWIFT's native CUDA sampler/models.

Only observed snapshots enter History. Query scoring never updates TGN MailBox.
Run through run_local.sh (the bundled DGL runtime requires its environment).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from memorys import MailBox
from modules import GeneralModel
from sampler.sampler_gpu import Sampler_GPU
from smoke_test import make_csr, model_params


def canonical(rows):
    result = set()
    for u, v, *_ in rows:
        u, v = int(u), int(v)
        if u != v:
            result.add((min(u, v), max(u, v)))
    return sorted(result)


class History:
    def __init__(self):
        self.ids = {}
        self.edges = []
        self.times = []
        self.edge_set = set()
        self.digest = hashlib.sha256()
        self.t = -1

    def observe(self, t, rows):
        if t != self.t + 1:
            raise ValueError("snapshots must be observed in order")
        pairs = canonical(rows)
        self.digest.update(np.asarray([t, len(pairs)], dtype="<i8").tobytes())
        for u, v in pairs:
            self.digest.update(np.asarray([u, v], dtype="<i8").tobytes())
            for node in (u, v):
                if node not in self.ids:
                    self.ids[node] = len(self.ids)
            self.edges.append((self.ids[u], self.ids[v]))
            self.times.append(t + 1)
            self.edge_set.add((u, v))
        self.t = t
        return pairs

    def graph(self):
        pairs = np.asarray(self.edges, dtype=np.int32).reshape(-1, 2)
        return make_csr(pairs[:, 0], pairs[:, 1],
                        np.asarray(self.times, dtype=np.float32), len(self.ids) + 1)


class SnapshotEngine:
    def __init__(self, kind="TGAT", layers=1, fanout=3, batch_size=128,
                 state=None):
        if kind not in ("TGAT", "TGN") or layers not in (1, 2):
            raise ValueError("supported models: TGAT/TGN with 1 or 2 layers")
        if min(fanout, batch_size) < 1 or not torch.cuda.is_available():
            raise ValueError("positive fanout/batch_size and SWIFT CUDA required")
        self.kind, self.layers, self.fanout, self.batch_size = kind, layers, fanout, batch_size
        self.history = History()
        sampling, memory, gnn, training = model_params(kind, layers)
        sampling["neighbor"] = [fanout] * layers
        self.model = GeneralModel(8, 4, sampling, memory, gnn, training).cuda()
        if state is not None:
            self.model.load_state_dict(state)
        self.mailbox = None
        self.sampler = None
        self._rebuild()

    def _rebuild(self):
        self.sampler = Sampler_GPU(self.history.graph(), [self.fanout] * self.layers,
                                   self.layers)
        if self.kind == "TGN":
            old = self.mailbox
            self.mailbox = MailBox(self.model.memory_param, len(self.history.ids) + 1, 4)
            self.mailbox.move_to_gpu()
            if old is not None:
                for name in ("node_memory", "node_memory_ts", "mailbox",
                             "mailbox_ts", "next_mail_pos"):
                    target, source = getattr(self.mailbox, name), getattr(old, name)
                    target[:len(source)].copy_(source)

    def _blocks(self, roots, query_ts):
        ts = np.full(len(roots), query_ts, dtype=np.float32)
        sampled = self.sampler.sample_layer(torch.as_tensor(roots, dtype=torch.int32,
                                                            device="cuda"),
                                            torch.from_numpy(ts).cuda())
        blocks = self.sampler.gen_mfgs(sampled)
        for layer in blocks:
            for block in layer:
                block.edata["f"] = torch.zeros((block.num_edges(), 4), device="cuda")
        for block in blocks[0]:
            block.srcdata["h"] = torch.zeros((block.num_src_nodes(), 8), device="cuda")
        if self.mailbox is not None:
            self.mailbox.prep_input_mails(blocks[0])
        return blocks, ts

    def logits(self, pairs, target_t):
        """Differentiable symmetric logits. History must end at target_t-1."""
        if target_t != self.history.t + 1:
            raise ValueError("target must immediately follow observed history")
        if not pairs:
            return torch.empty(0, device="cuda")
        a, b = np.asarray(pairs, dtype=np.int32).T
        values = []
        for src, dst in ((a, b), (b, a)):
            roots = np.concatenate((src, dst, dst)).astype(np.int32)
            blocks, _ = self._blocks(roots, target_t + 1)
            values.append(self.model(blocks)[0].squeeze(1))
        return (values[0] + values[1]) * 0.5

    def score_edges(self, pairs, target_t, batch_size=None):
        if target_t != self.history.t + 1:
            raise ValueError("target must immediately follow observed history")
        pairs = list(pairs)
        mapped = [(self.history.ids.get(int(u), len(self.history.ids)),
                   self.history.ids.get(int(v), len(self.history.ids))) for u, v in pairs]
        previous = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                scores = [torch.sigmoid(self.logits(mapped[i:i + (batch_size or self.batch_size)],
                                                   target_t)).cpu().numpy()
                          for i in range(0, len(mapped), batch_size or self.batch_size)]
        finally:
            self.model.train(previous)
        return np.concatenate(scores).astype(np.float32) if scores else np.empty(0, np.float32)

    def observe(self, t, rows):
        """Replay real events against previous graph, then publish this snapshot."""
        if t != self.history.t + 1:
            raise ValueError("snapshots must be observed in order")
        pairs = canonical(rows)
        for u, v in pairs:
            for node in (u, v):
                if node not in self.history.ids:
                    self.history.ids[node] = len(self.history.ids)
        # New IDs require a larger CSR and mailbox before replay; no current
        # edges are present in the sampler until the snapshot is committed.
        self._rebuild()
        self.model.eval()
        if self.mailbox is not None:
            with torch.no_grad():
                src_ids, dst_ids, src_mem, dst_mem = [], [], [], []
                for left in range(0, len(pairs), self.batch_size):
                    chunk = pairs[left:left + self.batch_size]
                    a, b = np.asarray([(self.history.ids[u], self.history.ids[v])
                                       for u, v in chunk], dtype=np.int32).T
                    roots = np.concatenate((a, b, a)).astype(np.int32)
                    blocks, ts = self._blocks(roots, t + 1)
                    self.model(blocks)
                    updater = self.model.memory_updater
                    src_ids.append(a)
                    dst_ids.append(b)
                    src_mem.append(updater.last_updated_memory[:len(chunk)].clone())
                    dst_mem.append(updater.last_updated_memory[len(chunk):2 * len(chunk)].clone())
                if pairs:
                    a, b = np.concatenate(src_ids), np.concatenate(dst_ids)
                    roots = np.concatenate((a, b, a))
                    ts = np.full(len(roots), t + 1, dtype=np.float32)
                    memory = torch.cat((torch.cat(src_mem), torch.cat(dst_mem)))
                    nid = torch.from_numpy(roots[:2 * len(pairs)]).cuda()
                    mem_ts = torch.full((2 * len(pairs),), t + 1,
                                        dtype=torch.float32, device="cuda")
                    self.mailbox.update_mailbox(nid, memory, roots, ts,
                        torch.zeros((len(pairs), 4), device="cuda"), None)
                    self.mailbox.update_memory(nid, memory, roots, mem_ts)
        self.history.observe(t, pairs)
        self._rebuild()


class SnapshotPredictor:
    protocol = "swift_snapshot_v1"

    def __init__(self, snapshots, checkpoint):
        self.snapshots = snapshots
        saved = torch.load(checkpoint, map_location="cpu")  # trusted local checkpoint
        if saved.get("protocol") != self.protocol:
            raise ValueError("incompatible SWIFT snapshot checkpoint")
        self.fit_end_t = int(saved["fit_end_t"])
        self.expected_digest = saved["digest"]
        self.engine = SnapshotEngine(**saved["config"], state=saved["model"])

    def prepare_time(self, t):
        if not self.fit_end_t <= t < len(self.snapshots) - 1:
            raise ValueError("t must be at or after the fitted prefix and before final snapshot")
        if t < self.engine.history.t:
            raise ValueError("history cannot be rewound; create a new predictor")
        for i in range(self.engine.history.t + 1, t + 1):
            self.engine.observe(i, self.snapshots[i])
            if i == self.fit_end_t and self.engine.history.digest.hexdigest() != self.expected_digest:
                raise ValueError("checkpoint fitted prefix differs from supplied snapshots")

    def score_edges(self, edges, target_t=None, batch_size=None):
        t = self.engine.history.t
        if t < self.fit_end_t or (target_t is not None and target_t != t + 1):
            raise ValueError("prepare_time(t) before scoring target t+1")
        return self.engine.score_edges(edges, t + 1, batch_size)


def negatives(positives, nodes, rng, occupied=None):
    occupied = set(positives) if occupied is None else occupied
    if len(nodes) < 3:
        raise ValueError("at least three observed nodes required")
    choices = np.asarray(nodes, dtype=np.int64)
    result = []
    for u, v in positives:
        destination = None
        for _ in range(32):
            w = int(choices[rng.randint(len(choices))])
            if w != u and (min(u, w), max(u, w)) not in occupied:
                destination = w
                break
        if destination is None:
            options = [w for w in nodes if w != u and
                       (min(u, w), max(u, w)) not in occupied]
            if not options:
                raise ValueError("no negative destination among observed nodes")
            destination = options[rng.randint(len(options))]
        result.append((u, destination))
    return result


def split_bounds(count, train_end=None, val_end=None):
    train_end = int(count * .55) if train_end is None else train_end
    val_end = int(count * .7) + 1 if val_end is None else val_end
    if not 2 <= train_end < val_end < count:
        raise ValueError("need nonempty train (targets 1..), validation, and test")
    return train_end, val_end


def train(snapshots, output, kind="TGAT", layers=1, fanout=3,
          epochs=1, batch_size=128, train_end=None, val_end=None,
          max_edges_per_snapshot=None, seed=2026):
    """Train targets [1, train_end); validate [train_end, val_end)."""
    if epochs < 1 or (max_edges_per_snapshot is not None and max_edges_per_snapshot < 1):
        raise ValueError("epochs and optional smoke edge limit must be positive")
    train_end, val_end = split_bounds(len(snapshots), train_end, val_end)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    engine = SnapshotEngine(kind, layers, fanout, batch_size)
    best, reports = -1., []
    for epoch in range(epochs):
        engine = SnapshotEngine(kind, layers, fanout, batch_size,
                                state=engine.model.state_dict())
        optimizer = torch.optim.Adam(engine.model.parameters(), lr=.001)
        labels, scores = [], []
        for target in range(val_end):
            if target == 0:
                engine.observe(0, snapshots[0])
                continue
            rows = canonical(snapshots[target])
            if max_edges_per_snapshot is not None:
                rows = rows[:max_edges_per_snapshot]
            nodes = sorted(engine.history.ids)
            positive = [(u, v) for u, v in rows if u in engine.history.ids and
                        v in engine.history.ids]
            rng = np.random.RandomState(seed + target + (epoch if target < train_end else 0))
            # Exclude all target positives, including smoke-cap omissions.
            occupied = set(canonical(snapshots[target])) if max_edges_per_snapshot is not None else set(rows)
            negative = negatives(positive, nodes, rng, occupied) if positive else []
            pairs = positive + negative
            mapped = [(engine.history.ids[u], engine.history.ids[v]) for u, v in pairs]
            training = target < train_end
            engine.model.train(training)
            # Every batch sees the same observed prefix. No target event is replayed
            # into the sampler or mailbox until scoring the whole target finishes.
            for left in range(0, len(pairs), batch_size):
                chunk = mapped[left:left + batch_size]
                positive_count = max(0, min(len(positive) - left, len(chunk)))
                truth = torch.tensor([1.] * positive_count +
                                     [0.] * (len(chunk) - positive_count), device="cuda")
                with torch.set_grad_enabled(training):
                    logits = engine.logits(chunk, target)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if not training:
                    scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
                    labels.extend(truth.cpu().tolist())
            engine.observe(target, snapshots[target])
        ap = float(average_precision_score(labels, scores)) if labels else float("nan")
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
        reports.append({"epoch": epoch + 1, "val_ap": ap, "val_auc": auc})
        print(json.dumps(reports[-1]), flush=True)
        if np.isfinite(ap) and ap > best:
            best = ap
            # Replaying the prefix with the selected model gives a reproducible
            # inference state; no ephemeral training mailbox is serialized.
            torch.save({"protocol": SnapshotPredictor.protocol,
                        "model": {k: v.detach().cpu() for k, v in engine.model.state_dict().items()},
                        "config": dict(kind=kind, layers=layers, fanout=fanout,
                                       batch_size=batch_size),
                        "fit_end_t": val_end - 1,
                        "digest": engine.history.digest.hexdigest()}, output / "best.pt")
    (output / "report.json").write_text(json.dumps({"train_end": train_end,
        "val_end": val_end, "smoke_edge_limit": max_edges_per_snapshot,
        "epochs": reports}, indent=2) + "\n")
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("TGAT", "TGN"), default="TGAT")
    parser.add_argument("--layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fanout", type=int, default=3)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--val-end", type=int)
    parser.add_argument("--max-edges-per-snapshot", type=int, help="smoke test only")
    args = parser.parse_args()
    from pathlib import Path as _Path
    import sys
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from methods.swift_snapshot import load_snapshot_edges
    train(load_snapshot_edges(args.slices_dir), args.output, args.model, args.layers,
          args.fanout, args.epochs, args.batch_size, args.train_end, args.val_end,
          args.max_edges_per_snapshot)


if __name__ == "__main__":
    main()
