"""In-memory validation using SWIFT's native CUDA sampler and original models.

This is deliberately separate from train.py's disk-bucket/prefetch pipeline.
Run with: bash run_local.sh smoke_test.py
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from modules import GeneralModel
from memorys import MailBox
from sampler.sampler_gpu import Sampler_GPU


def load_edges(path):
    frame = pd.read_csv(path, dtype={"u": str, "v": str})
    if not {"u", "v", "ts"}.issubset(frame.columns):
        raise ValueError("CSV must contain u,v,ts")
    if frame[["u", "v", "ts"]].isnull().any().any() or len(frame) < 20:
        raise ValueError("Need at least 20 complete events")
    times = pd.to_numeric(frame.ts, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("Timestamps must be finite")
    order = np.argsort(times, kind="stable")
    frame = frame.iloc[order]
    times = times[order]
    # SWIFT stores timestamps as float32. Remove the epoch offset, but do not
    # silently merge distinct timestamps when converting a custom dataset.
    times = times - times[0] + 1
    rounded = times.astype(np.float32)
    if np.any((np.diff(times) > 0) & (np.diff(rounded) == 0)):
        raise ValueError("Time resolution is lost in float32; rescale timestamps")
    if not np.isfinite(rounded).all():
        raise ValueError("Time range exceeds float32")
    codes, nodes = pd.factorize(pd.concat([frame.u, frame.v], ignore_index=True))
    size = len(frame)
    if len(nodes) < 3:
        raise ValueError("At least three nodes are needed")
    src, dst = codes[:size].astype(np.int32), codes[size:].astype(np.int32)
    return src, dst, rounded, [str(node) for node in nodes]


def make_csr(src, dst, times, num_nodes):
    rows = [[] for _ in range(num_nodes)]
    for eid, (u, v, ts) in enumerate(zip(src, dst, times)):
        rows[u].append((ts, v, eid))
        rows[v].append((ts, u, eid))
    indptr, indices, timestamps, eids = [0], [], [], []
    for row in rows:
        row.sort()
        for ts, node, eid in row:
            indices.append(node)
            timestamps.append(ts)
            eids.append(eid)
        indptr.append(len(indices))
    return dict(indptr=np.asarray(indptr, dtype=np.int32),
                indices=np.asarray(indices, dtype=np.int32),
                ts=np.asarray(timestamps, dtype=np.float32),
                eid=np.asarray(eids, dtype=np.int32))


def validate_sampler(sampler, graph):
    roots = torch.arange(len(graph["indptr"]) - 1, dtype=torch.int32, device="cuda")
    checks = 0
    for time in (0.0, 1.0, 17.0, float(graph["ts"].max()) + 1):
        query = torch.full(roots.shape, time, dtype=torch.float32, device="cuda")
        result = sampler.sample_layer(roots, query)[0]
        neighbors, targets, timestamps, eids, _, _, _ = [
            value.cpu().numpy() for value in result]
        for node in roots.cpu().tolist():
            lo, hi = graph["indptr"][node:node + 2]
            available = np.arange(lo, hi)[graph["ts"][lo:hi] < time]
            expected = available[-sampler.fan_nums[0]:][::-1]
            mask = targets == node
            np.testing.assert_array_equal(eids[mask], graph["eid"][expected])
            np.testing.assert_array_equal(neighbors[mask], graph["indices"][expected])
            np.testing.assert_array_equal(timestamps[mask], graph["ts"][expected])
            checks += 1
    return checks


def model_params(kind, layers):
    sampling = dict(layer=layers, neighbor=[3] * layers, history=1)
    memory = dict(type="node" if kind == "TGN" else "none", dim_out=16,
                  dim_time=8, deliver_to="self", mail_combine="last",
                  memory_update="gru", mailbox_size=1, combine_node_feature=True)
    gnn = dict(arch="transformer_attention", layer=layers, att_head=2,
               dim_time=8, dim_out=16)
    train = dict(dropout=0.1, att_dropout=0.1)
    return sampling, memory, gnn, train


def run_model(kind, layers, edges, graph, output, epochs, batch_size):
    torch.manual_seed(2026)
    np.random.seed(2026)
    dgl.seed(2026)
    src, dst, times, nodes = edges
    parameters = model_params(kind, layers)
    model = GeneralModel(8, 4, *parameters).cuda()
    initial = {key: value.detach().cpu().clone()
               for key, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler = Sampler_GPU(graph, [3] * layers, layers)
    mailbox = MailBox(parameters[1], len(nodes), 4) if kind == "TGN" else None
    if mailbox is not None:
        mailbox.move_to_gpu()
    generator = torch.Generator().manual_seed(17)
    node_features = torch.randn(len(nodes), 8, generator=generator).cuda()
    edge_features = torch.randn(len(src), 4, generator=generator).cuda()
    # Keep equal timestamps on the same side of a split.
    train_end = int(np.searchsorted(times, times[int(len(src) * .7)], side="left"))
    val_end = int(np.searchsorted(times, times[int(len(src) * .85)], side="left"))
    if not 0 < train_end < val_end < len(src):
        raise ValueError("Timestamp groups leave an empty chronological split")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    gradient_seen = False

    def execute(start, end, training, seed):
        nonlocal gradient_seen
        model.train(training)
        rng = np.random.RandomState(seed)
        predictions, losses = [], []
        for left in range(start, end, batch_size):
            right = min(left + batch_size, end)
            u, v, ts = src[left:right], dst[left:right], times[left:right]
            negative = rng.randint(0, len(nodes) - 1, size=len(u)).astype(np.int32)
            negative += negative >= v
            roots = np.concatenate([u, v, negative]).astype(np.int32)
            query_ts = np.tile(ts, 3).astype(np.float32)
            sampled = sampler.sample_layer(torch.from_numpy(roots).cuda(),
                                           torch.from_numpy(query_ts).cuda())
            blocks = sampler.gen_mfgs(sampled)
            for layer in blocks:
                for block in layer:
                    assert bool((block.edata["dt"] > 0).all())
                    block.edata["f"] = edge_features[block.edata["ID"].long()]
            for block in blocks[0]:
                block.srcdata["h"] = node_features[block.srcdata["ID"].long()].clone()
            if mailbox is not None:
                mailbox.prep_input_mails(blocks[0])
            with torch.set_grad_enabled(training):
                pos, neg = model(blocks)
                assert pos.shape == neg.shape == (len(u), 1)
                loss = (loss_fn(pos, torch.ones_like(pos)) +
                        loss_fn(neg, torch.zeros_like(neg)))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            assert bool(torch.isfinite(parameter.grad).all())
                            gradient_seen |= bool((parameter.grad != 0).any())
                    optimizer.step()
            predictions.append(torch.cat([pos, neg], dim=1).detach().cpu())
            losses.append(float(loss.detach()) * len(u))
            if mailbox is not None:
                updater = model.memory_updater
                mailbox.update_mailbox(
                    updater.last_updated_nid, updater.last_updated_memory,
                    roots, query_ts, edge_features[left:right], None)
                mailbox.update_memory(
                    updater.last_updated_nid, updater.last_updated_memory,
                    roots, updater.last_updated_ts)
        scores = torch.cat(predictions).sigmoid().numpy()
        labels = np.tile([1, 0], len(scores))
        return scores, dict(loss=sum(losses) / (end - start),
                            ap=float(average_precision_score(labels, scores.ravel())),
                            auc=float(roc_auc_score(labels, scores.ravel())))

    def state():
        return dict(model=copy.deepcopy(model.state_dict()),
                    mailbox=None if mailbox is None else {
                        name: getattr(mailbox, name).clone() for name in
                        ("node_memory", "node_memory_ts", "mailbox",
                         "mailbox_ts", "next_mail_pos")})

    def restore(saved):
        model.load_state_dict(saved["model"])
        if mailbox is not None:
            for name, value in saved["mailbox"].items():
                getattr(mailbox, name).copy_(value)

    checkpoint = output / "{}-{}.pt".format(kind, layers)
    best, history = -1, []
    for epoch in range(epochs):
        if mailbox is not None:
            mailbox.reset()
        _, train_metrics = execute(0, train_end, True, epoch)
        _, val_metrics = execute(train_end, val_end, False, 100)
        history.append(dict(train=train_metrics, val=val_metrics))
        print(kind, layers, "epoch", epoch + 1, history[-1], flush=True)
        if val_metrics["ap"] > best:
            best = val_metrics["ap"]
            torch.save(dict(state=state(), parameters=parameters,
                            node_features=node_features, edge_features=edge_features,
                            node_ids=nodes, val_end=val_end), checkpoint)
    saved = torch.load(checkpoint, map_location="cuda")
    restore(saved["state"])
    scores, metrics = execute(val_end, len(src), False, 200)
    # A new model object plus serialized mailbox must reproduce streaming test.
    model = GeneralModel(8, 4, *parameters).cuda()
    restore(saved["state"])
    reloaded_scores, _ = execute(val_end, len(src), False, 200)
    reload_error = float(np.max(np.abs(scores - reloaded_scores)))
    assert np.allclose(scores, reloaded_scores, rtol=1e-5, atol=1e-6)
    changed = any(not torch.equal(initial[key], value.detach().cpu())
                  for key, value in model.named_parameters())
    assert changed and gradient_seen
    return dict(model=kind, layers=layers, history=history, test=metrics,
                parameters_changed=changed, nonzero_finite_gradient=gradient_seen,
                reload_max_error=reload_error, checkpoint=str(checkpoint),
                splits=[train_end, val_end - train_end, len(src) - val_end])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="+", choices=["TGAT", "TGN"],
                        default=["TGAT", "TGN"])
    parser.add_argument("--layers", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "local-results")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not torch.cuda.is_available():
        parser.error("This native SWIFT smoke test requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="smoke-", dir=str(args.output.resolve())))
    data_path = args.data_path
    if data_path is None:
        rng = np.random.RandomState(42)
        src = np.arange(217) % 24
        dst = (src + rng.randint(1, 24, len(src))) % 24
        data_path = output / "synthetic.csv"
        pd.DataFrame(dict(u=100 + src * 10, v=100 + dst * 10,
                          ts=np.arange(1, len(src) + 1))).to_csv(data_path, index=False)
    edges = load_edges(data_path)
    graph = make_csr(*edges[:3], len(edges[3]))
    checks = validate_sampler(Sampler_GPU(graph, [3], 1), graph)
    report = dict(torch=torch.__version__, dgl=dgl.__version__,
                  gpu=torch.cuda.get_device_name(0), data=str(data_path.resolve()),
                  events=len(edges[0]), nodes=len(edges[3]),
                  sampler_reference_checks=checks,
                  scope="native CUDA sampler and in-memory models, not disk pipeline")
    report["runs"] = [run_model(kind, layers, edges, graph, output,
                                args.epochs, args.batch_size)
                      for kind in args.models for layers in args.layers]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
