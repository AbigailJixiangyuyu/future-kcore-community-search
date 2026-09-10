#!/usr/bin/env python3
"""Select nodes by coreness-threshold BFS, then generate their community edges."""

import argparse
import json
import time
from pathlib import Path

from hybrid_community import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SLICES,
    EVALUATION_KS,
    _build_predictor,
)


def predict_community(predictor, q, k, t):
    """Run the full model-to-community pipeline without inspecting t+1 edges."""
    if k not in EVALUATION_KS:
        raise ValueError("k must be in 3..7")
    started = time.time()
    context = predictor.prepare_time(t)
    selected = set()
    if q in context.adjacency:
        selected = predictor.predict(q, k, t, context=context).community
    selected_at = time.time()
    result = predictor.generate_edges(t, selected, k, context=context)
    finished = time.time()
    return {
        "method": "coreness_generated_edges",
        "q": q,
        "k": k,
        "t": t,
        "target_t": t + 1,
        "history": "all_snapshots_through_t",
        "candidate_selection": "q_rooted_predicted_coreness_threshold_bfs",
        "edge_candidates": "cached_tppr_vertices_within_bfs_set",
        "deficit_policy": "immediate_effective_core_two_hop_product_sum_to_k",
        "node_processing_order": "affected_visited_first_then_effective_core_desc_id_asc",
        "branch_processing": "immediate_per_node",
        "effective_coreness": result.effective_coreness,
        "node_update_count": result.node_update_count,
        "reprocessed_node_count": result.reprocessed_node_count,
        "core_update_count": len(result.core_updates),
        "core_updates": [
            {"node": node, "old_core": old, "new_core": new}
            for node, old, new in result.core_updates
        ],
        "postprocessing": "none",
        "community": list(result.nodes),
        "community_size": len(result.nodes),
        "edges": [list(edge) for edge in result.edges],
        "generated_edge_count": len(result.edges),
        "bfs_selection_s": selected_at - started,
        "edge_generation_s": finished - selected_at,
        "elapsed_s": finished - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q", type=int, help="original node ID")
    parser.add_argument("k", type=int, choices=EVALUATION_KS)
    parser.add_argument("t", type=int, help="zero-based last historical snapshot")
    parser.add_argument("--slices-dir", default=str(DEFAULT_SLICES))
    parser.add_argument(
        "--checkpoint", required=True,
        help="explicit current-architecture coreness checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", help="optional JSON output (must not exist)")
    args = parser.parse_args(argv)
    if args.output and Path(args.output).exists():
        parser.error("--output already exists; choose a new result path")
    predictor, _ = _build_predictor(args)
    payload = predict_community(predictor, args.q, args.k, args.t)
    payload["slices_dir"] = str(Path(args.slices_dir).resolve())
    payload["checkpoint"] = str(Path(args.checkpoint).resolve())
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
