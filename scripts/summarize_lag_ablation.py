"""Validate paired Lag runs and print their metrics/deltas as JSON."""

import argparse
import json
from pathlib import Path

import torch


KS = ("3", "4", "5", "6", "7")


def read_json(path):
    return json.loads(path.read_text())


def node_metrics(metrics):
    assert set(metrics["per_k"]) == set(KS)
    return {
        "accuracy": metrics["accuracy"],
        "mae": metrics["mae"],
        "macro_f1": sum(metrics["per_k"][k]["f1"] for k in KS) / len(KS),
    }


def summarize(root):
    result = {
        "comparison": "no_lag minus with_lag; negative MAE delta is better",
        "evaluation_k": [int(k) for k in KS],
        "datasets": {},
    }
    for dataset in ("email", "mooc"):
        directory = root / dataset
        checkpoints = {
            variant: torch.load(directory / (variant + ".pt"), map_location="cpu")
            for variant in ("with_lag", "no_lag")
        }
        left, right = checkpoints["with_lag"], checkpoints["no_lag"]
        assert left["feature_config"] == right["feature_config"]
        assert left["objective"] == right["objective"]
        assert left["split_config"] == right["split_config"]
        assert left["model_config"]["use_lag"] is True
        assert right["model_config"]["use_lag"] is False
        for key in left["model_config"]:
            if key != "use_lag":
                assert left["model_config"][key] == right["model_config"][key], key
        for key in left["training_config"]:
            if key not in ("best_epoch", "epochs_run"):
                assert left["training_config"][key] == right["training_config"][key], key
        paired = {}
        for variant, checkpoint in checkpoints.items():
            paired[variant] = {
                "training": checkpoint["training_config"],
                "node_test": node_metrics(read_json(directory / (variant + ".json"))["test"]),
            }
        for scope in ("community", "community_test"):
            evaluations = {
                variant: read_json(directory / (variant + "_" + scope + ".json"))
                for variant in paired
            }
            a, b = evaluations["with_lag"], evaluations["no_lag"]
            assert a["start_t"] == b["start_t"]
            assert a["samples"] == b["samples"]
            assert set(a["per_k"]) == set(b["per_k"]) == set(KS)
            for k in KS:
                assert a["per_k"][k]["samples"] == b["per_k"][k]["samples"]
            for variant, evaluation in evaluations.items():
                paired[variant][scope] = {
                    "start_t": evaluation["start_t"],
                    "samples": evaluation["samples"],
                    "macro": {
                        metric: evaluation["macro"][metric]
                        for metric in ("precision", "recall", "f1", "jaccard", "size_ratio")
                    },
                    "per_k_f1": {k: evaluation["per_k"][k]["f1"] for k in KS},
                }
        deltas = {}
        for scope in ("node_test", "community", "community_test"):
            a, b = paired["with_lag"][scope], paired["no_lag"][scope]
            if scope != "node_test":
                a, b = a["macro"], b["macro"]
            deltas[scope] = {key: b[key] - a[key] for key in a}
        paired["delta"] = deltas
        result["datasets"][dataset] = paired
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir), indent=2, sort_keys=True))
