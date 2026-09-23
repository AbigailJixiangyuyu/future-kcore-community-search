"""Checkpoint-backed PRISM next-snapshot community query and evaluation."""

import argparse
from collections import deque
import json

import networkit as nk

from datasets.community_eval_builder import set_metrics
from datasets.baseline_eval import (baseline_training_split, evaluation_start_t,
                                    load_test_samples, non_empty_samples,
                                    require_fit_boundary, query_set_sha256)
from methods.prism import load_predictor
from zebra_community import historical_community_union


def recover_component(candidate, edges, scores, q, k, threshold):
    if q not in candidate:
        return frozenset()
    nodes = sorted(candidate)
    lookup = {node: i for i, node in enumerate(nodes)}
    graph = nk.Graph(len(nodes), weighted=False, directed=False)
    for (u, v), score in zip(edges, scores):
        if score >= threshold:
            graph.addEdge(lookup[int(u)], lookup[int(v)])
    cores = nk.centrality.CoreDecomposition(graph).run().scores()
    start = lookup[q]
    if cores[start] < k:
        return frozenset()
    visited = {start}
    pending = deque([start])
    while pending:
        for neighbor in graph.iterNeighbors(pending.popleft()):
            if neighbor not in visited and cores[neighbor] >= k:
                visited.add(neighbor)
                pending.append(neighbor)
    return frozenset(nodes[i] for i in visited)


class PrismCommunityPredictor:
    candidate_protocol = "q_historical_kcore_union_and_historical_edges_through_t"

    def __init__(self, slices_dir, checkpoint, device="cpu", threshold=0.5,
                 batch_size=128):
        if not 0 <= threshold <= 1 or batch_size < 1:
            raise ValueError("threshold must be in [0,1] and batch size positive")
        self.predictor, self.snapshots = load_predictor(slices_dir, checkpoint, device)
        self.threshold = threshold
        self.batch_size = batch_size

    def predict(self, q, k, t):
        if k not in range(3, 8):
            raise ValueError("valid community k is 3..7")
        self.predictor.prepare_time(t)
        candidate = historical_community_union(self.snapshots, q, k, t)
        if not candidate:
            return frozenset()
        edges = self.predictor.history.edge_pairs(candidate)
        scores = self.predictor.score_edges(edges, target_t=t + 1,
                                            batch_size=self.batch_size)
        return recover_component(candidate, edges, scores, q, k, self.threshold)


def evaluate(predictor, samples, *, start_t=None):
    count = len(predictor.snapshots)
    start_t = evaluation_start_t(count) if start_t is None else start_t
    if not predictor.predictor.fit_end_t <= start_t < count - 1:
        raise ValueError("evaluation start must follow validation and precede the final snapshot")
    selected = non_empty_samples(
        sample for sample in samples if sample["k"] in range(3, 8)
        and int(sample["t"]) >= start_t
    )
    result = []
    for sample in sorted(selected, key=lambda s: s["t"]):
        community = predictor.predict(int(sample["query"]), int(sample["k"]),
                                      int(sample["t"]))
        metrics = set_metrics(community, sample["community"])
        result.append({"q": int(sample["query"]), "k": int(sample["k"]),
                       "t": int(sample["t"]), **metrics})
    return {"count": len(result), "start_t": start_t,
            "mean_f1": sum(x["f1"] for x in result) /
            len(result) if result else None, "results": result,
            "sample_scope": "shared_community_eval_non_empty_only",
            "query_set_sha256": query_set_sha256(selected),
            "candidate_protocol": predictor.candidate_protocol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-t", type=int, help="first current snapshot to evaluate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", nargs=3, type=int, metavar=("Q", "K", "T"))
    group.add_argument("--eval-samples", help="existing community evaluation pickle")
    args = parser.parse_args()
    predictor = PrismCommunityPredictor(args.slices_dir, args.checkpoint,
                                         args.device, args.threshold, args.batch_size)
    if args.query:
        q, k, t = args.query
        print(json.dumps({"community": sorted(predictor.predict(q, k, t)),
                          "candidate_protocol": predictor.candidate_protocol}))
    else:
        require_fit_boundary(predictor.predictor.fit_end_t, len(predictor.snapshots))
        samples = load_test_samples(args.slices_dir, len(predictor.snapshots), args.eval_samples)
        result = evaluate(predictor, samples, start_t=args.start_t)
        result["training_split"] = baseline_training_split(len(predictor.snapshots))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
