"""Validate GRU/Transformer experiment artifacts and print paired metrics."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.hybrid_coreness import load_hybrid_coreness_model


KS = ("3", "4", "5", "6", "7")


def summarize(root, variants=("gru", "transformer"), config_key="history_encoder_type"):
    result = {
        "comparison": "{} minus {}; lower MAE is better".format(variants[1], variants[0]),
        "evaluation_k": [int(k) for k in KS],
        "datasets": {},
    }
    for dataset in ("email", "mooc"):
        directory = root / dataset
        checkpoints, paired = {}, {}
        for encoder in variants:
            model, checkpoint = load_hybrid_coreness_model(directory / (encoder + ".pt"))
            checkpoints[encoder] = checkpoint
            assert getattr(model, config_key) == encoder
            if config_key == "output_head_type":
                assert model.history_encoder_type == "gru"
            assert not hasattr(model, "lag_table")
            node = checkpoint["metrics"]["test"]
            per_k = {str(k): value for k, value in node["per_k"].items()}
            assert set(per_k) == set(KS)
            community = json.loads(
                (directory / (encoder + "_community_test.json")).read_text()
            )
            assert set(community["per_k"]) == set(KS)
            assert community["start_t"] == (127 if dataset == "email" else 50)
            paired[encoder] = {
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "history_parameter_count": sum(p.numel() for p in model.history_encoder.parameters()),
                "training": checkpoint["training_config"],
                "node_test": {
                    "accuracy": node["accuracy"],
                    "mae": node["mae"],
                    "macro_f1": sum(per_k[k]["f1"] for k in KS) / len(KS),
                    "per_k": per_k,
                },
                "community_test": {
                    "start_t": community["start_t"],
                    "samples": community["samples"],
                    "macro": {
                        metric: community["macro"][metric]
                        for metric in ("precision", "recall", "f1", "jaccard", "size_ratio")
                    },
                    "per_k": {
                        k: {
                            metric: community["per_k"][k][metric]
                            for metric in ("precision", "recall", "f1", "jaccard", "samples")
                        }
                        for k in KS
                    },
                },
            }
        a, b = (checkpoints[variant] for variant in variants)
        for key in ("feature_config", "split_config", "objective"):
            assert a[key] == b[key], key
        assert set(a["model_config"]) == set(b["model_config"])
        for key in a["model_config"]:
            if key != config_key:
                assert a["model_config"][key] == b["model_config"][key], key
        for key in a["training_config"]:
            if key not in ("best_epoch", "epochs_run"):
                assert a["training_config"][key] == b["training_config"][key], key
        a, b = (paired[variant] for variant in variants)
        assert a["community_test"]["samples"] == b["community_test"]["samples"]
        for k in KS:
            assert a["community_test"]["per_k"][k]["samples"] == b["community_test"]["per_k"][k]["samples"]
        paired["delta"] = {
            "node_test": {
                metric: b["node_test"][metric] - a["node_test"][metric]
                for metric in ("accuracy", "mae", "macro_f1")
            },
            "community_test": {
                metric: b["community_test"]["macro"][metric] - a["community_test"]["macro"][metric]
                for metric in a["community_test"]["macro"]
            },
            "node_per_k_f1": {
                k: b["node_test"]["per_k"][k]["f1"] - a["node_test"]["per_k"][k]["f1"]
                for k in KS
            },
            "community_per_k_f1": {
                k: b["community_test"]["per_k"][k]["f1"] - a["community_test"]["per_k"][k]["f1"]
                for k in KS
            },
        }
        result["datasets"][dataset] = paired
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir), indent=2, sort_keys=True))
