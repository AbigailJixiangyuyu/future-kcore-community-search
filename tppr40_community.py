#!/usr/bin/env python3
"""T-PPR capacity-40 ablation: model reads Top-20; edge generation reads Top-40."""

from historical_neighbor_community import main as comparison_main
from hybrid_community import _build_predictor
from methods.coreness_edge_generation import generate_predicted_edges


METADATA = {
    "method": "coreness_generated_edges_tppr40",
    "edge_candidates": "cached_tppr_top40_vertices_within_bfs_set",
    "t_ppr_internal_top_k": 40,
    "model_top_l": 20,
    "edge_top_l": 40,
    "history": "all_snapshots_through_t",
    "model_features": "top20_from_shared_internal40_state",
    "retrained": False,
}


class TPPR40Predictor:
    def __init__(self, predictor):
        if predictor._current_time != -1:
            raise ValueError("capacity must be configured before advancing state")
        if predictor.feature_config["top_l"] != 20:
            raise ValueError("this ablation requires a Top-20 model checkpoint")
        self.predictor = predictor
        predictor.t_ppr_index = predictor.t_ppr.streaming_index(
            top_l=20, internal_top_k=40,
            min_score=predictor.feature_config["min_probability"],
        )
        # Do not read or overwrite the default capacity-20 state-cache directory.
        if predictor.state_cache_dir is not None:
            predictor.state_cache_dir = (
                predictor.state_cache_dir / "tppr40_model20"
            )

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
        return generate_predicted_edges(
            selected, context.coreness_cache,
            lambda v: self.predictor.t_ppr_index.top_neighbors(v, top_l=40), k,
        )


def build_predictor(args):
    predictor, total_nodes = _build_predictor(args)
    return TPPR40Predictor(predictor), total_nodes


def main(argv=None):
    comparison_main(
        argv, predictor_builder=build_predictor,
        metadata=METADATA, description=__doc__,
    )


if __name__ == "__main__":
    main()
