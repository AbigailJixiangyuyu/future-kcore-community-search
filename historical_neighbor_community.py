#!/usr/bin/env python3
"""Historical-neighbor candidate ablation; keep the default T-PPR method intact."""

import argparse
import json
from pathlib import Path

from generated_edge_community import predict_community
from hybrid_community import (
    EVALUATION_KS, _add_common_arguments, _build_predictor, _evaluate,
)
from methods.coreness_edge_generation import generate_predicted_edges
from methods.t_ppr import TemporalInfluence


METADATA = {
    "method": "coreness_generated_edges_historical_neighbors",
    "edge_candidates": "historical_direct_neighbors_within_bfs_set",
    "history": "all_snapshots_through_t",
    "candidate_score": "cached_raw_tppr_sum_missing_zero",
    "candidate_tie_break": "node_id_ascending",
    "second_hop_candidates": "historical_neighbors_of_historical_neighbors_within_bfs_set",
}


class HistoricalNeighborPredictor:
    """Delegate model/BFS unchanged; replace only edge candidate construction."""

    def __init__(self, predictor):
        self.predictor = predictor

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def generate_edges(self, t, nodes, k, context=None):
        context = context or self.predictor.prepare_time(t)
        if context.time != t or context is not self.predictor._context:
            raise ValueError("context is stale or does not match the query time")
        if self.predictor.t_ppr_index.current_time != t:
            raise ValueError("T-PPR state does not match the query time")
        selected = set(nodes)
        if not selected.issubset(context.adjacency):
            raise ValueError("selected nodes must belong to historical graph")
        if not selected.issubset(context.coreness_cache):
            raise ValueError("selected nodes must have cached BFS predictions")

        def candidates(node):
            # Use the same raw cached scores as the default generator.
            # Zero-score historical candidates must remain selectable.
            scores = {}
            for record in self.predictor.t_ppr_index.top_neighbors(node):
                scores[record.node] = scores.get(record.node, 0.) + record.score
            return [
                TemporalInfluence(v, t, scores.get(v, 0.), 0.)
                for v in sorted(set(context.adjacency[node]) & selected - {node})
            ]

        return generate_predicted_edges(selected, context.coreness_cache, candidates, k)


def build_predictor(args):
    predictor, total_nodes = _build_predictor(args)
    return HistoricalNeighborPredictor(predictor), total_nodes


def main(argv=None, *, predictor_builder=None, metadata=None, description=None):
    """Shared comparison CLI; defaults remain the historical-neighbor variant."""
    predictor_builder = predictor_builder or build_predictor
    metadata = METADATA if metadata is None else metadata
    parser = argparse.ArgumentParser(description=description or __doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("query", "eval"):
        sub = commands.add_parser(command)
        _add_common_arguments(sub)
        sub.add_argument("--output", help="JSON destination (must not exist)")
        if command == "query":
            sub.add_argument("q", type=int)
            sub.add_argument("k", type=int, choices=EVALUATION_KS)
            sub.add_argument("t", type=int)
        else:
            sub.add_argument("--start-t", type=int, default=None)
            sub.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("--checkpoint is required for this comparison")
    if args.output and Path(args.output).exists():
        parser.error("--output already exists; choose a new result path")
    if args.command == "eval":
        _evaluate(args, predictor_builder=predictor_builder, metadata=metadata)
        return
    predictor, _ = predictor_builder(args)
    payload = {
        **predict_community(predictor, args.q, args.k, args.t),
        **metadata,
        "slices_dir": str(Path(args.slices_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
