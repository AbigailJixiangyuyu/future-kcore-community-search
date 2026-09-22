#!/usr/bin/env python3
"""SWIFT next-snapshot community query/evaluation, using historical edges only."""
import argparse
from collections import deque
import json

from datasets.community_eval_builder import set_metrics
from datasets.baseline_eval import load_test_samples, require_fit_boundary, query_set_sha256
from datasets.dataset_builder import build_snapshots
from methods.swift_snapshot import load_predictor


class SwiftCommunity:
    """Candidates: union of q's historical k-core components; edges: observed only."""
    protocol = "historical_q_k_core_union__historical_edges"

    def __init__(self, snapshots, predictor, threshold=.5, batch_size=128):
        if not 0 <= threshold <= 1 or batch_size <= 0:
            raise ValueError("invalid threshold/batch size")
        self.snapshots, self.predictor = snapshots, predictor
        self.threshold, self.batch_size = threshold, batch_size
        self.t = -1
        self.edges = set()
        self.communities = {k: {} for k in range(3, 8)}

    def prepare_time(self, t):
        if not self.predictor.fit_end_t <= t < len(self.snapshots) - 1 or t < self.t:
            raise ValueError("invalid or backwards query time")
        self.predictor.prepare_time(t)
        for i in range(self.t + 1, t + 1):
            snapshot = self.snapshots[i]
            for u, v, *_ in snapshot["edge_list"]:
                u, v = int(u), int(v)
                if u != v:
                    self.edges.add((min(u, v), max(u, v)))
            for k in range(3, 8):
                for component in snapshot.get("k_core_comps", {}).get(k, {}).get("components", ()):
                    members = set(component)
                    for q in members:
                        self.communities[k].setdefault(q, set()).update(members)
        self.t = t

    def candidate(self, q, k):
        if k not in range(3, 8):
            raise ValueError("community evaluation supports only k=3..7")
        return set(self.communities[k].get(q, ()))

    def predict(self, q, k, t):
        if t != self.t:
            raise ValueError("prepare_time(t) before querying")
        candidate = self.candidate(q, k)
        if q not in candidate:
            return frozenset()
        pairs = sorted(edge for edge in self.edges if edge[0] in candidate
                       and edge[1] in candidate)
        neighbors = {node: set() for node in candidate}
        for left in range(0, len(pairs), self.batch_size):
            chunk = pairs[left:left + self.batch_size]
            scores = self.predictor.score_edges(chunk, target_t=t + 1,
                                                 batch_size=self.batch_size)
            for (u, v), score in zip(chunk, scores):
                if score >= self.threshold:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        degree = {node: len(adj) for node, adj in neighbors.items()}
        removed = set()
        queue = deque(node for node, d in degree.items() if d < k)
        while queue:
            node = queue.popleft()
            if node in removed:
                continue
            removed.add(node)
            for other in neighbors[node]:
                degree[other] -= 1
                if other not in removed and degree[other] < k:
                    queue.append(other)
        if q in removed:
            return frozenset()
        visited = {q}
        queue = deque([q])
        while queue:
            for other in neighbors[queue.popleft()] - removed - visited:
                visited.add(other)
                queue.append(other)
        return frozenset(visited)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=128)
    sub = parser.add_subparsers(dest="command", required=True)
    query = sub.add_parser("query")
    query.add_argument("q", type=int)
    query.add_argument("k", type=int)
    query.add_argument("t", type=int)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("samples", help="prebuilt community_eval/<dataset>.pkl")
    evaluate.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    snaps = build_snapshots(args.slices_dir)[0]
    predictor = load_predictor(args.slices_dir, args.checkpoint)
    method = SwiftCommunity(snaps, predictor, args.threshold, args.batch_size)
    if args.command == "query":
        method.prepare_time(args.t)
        result = method.predict(args.q, args.k, args.t)
        print(json.dumps({"q": args.q, "k": args.k, "t": args.t,
                          "community": sorted(result), "candidate_protocol": method.protocol}))
    else:
        require_fit_boundary(method.predictor.fit_end_t, len(snaps))
        samples = load_test_samples(args.slices_dir, len(snaps), args.samples)
        samples = sorted((s for s in samples if method.predictor.fit_end_t <= s["t"]
                          and 3 <= s["k"] <= 7), key=lambda s: s["t"])
        if args.limit is not None:
            samples = samples[:args.limit]
        results = []
        for sample in samples:
            t = int(sample["t"])
            method.prepare_time(t)
            prediction = method.predict(int(sample["query"]), int(sample["k"]), t)
            results.append(set_metrics(prediction, sample["community"]))
        print(json.dumps({"count": len(results), "candidate_protocol": method.protocol,
                          "query_set_sha256": query_set_sha256(samples),
                          "threshold": method.threshold,
                          "mean": {key: sum(item[key] for item in results) / len(results)
                                   for key in ("precision", "recall", "f1", "jaccard")}
                          if results else {}}))


if __name__ == "__main__":
    main()
