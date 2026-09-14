#!/usr/bin/env python3
"""One isolated run per loader/dataset; compare retention and exact predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.dataset_builder import build_snapshots
from datasets.snapshot_store import open_snapshot_store
from hybrid_community import HybridCommunityPredictor
from methods.hybrid_coreness import load_hybrid_coreness_model
from methods.t_ppr import TemporalPPR


def rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Linux /proc required")


def digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def worker(args):
    torch.set_num_threads(1)
    slices = ROOT / ("data/mooc/time_slices/step_43200_window_86400"
                     if args.dataset == "mooc" else
                     "data/email-Eu-core-temporal/time_slices/step_302400_window_604800")
    if args.mode == "migrate":
        started = time.perf_counter()
        store, *_ = open_snapshot_store(slices)
        result = {"migration_s": time.perf_counter() - started,
                  "directory": str(store.directory)}
    else:
        baseline_rss = rss_mb()
        started = time.perf_counter()
        loader = build_snapshots if args.mode == "eager" else open_snapshot_store
        snapshots, _, _, hmax = loader(slices)
        model, checkpoint = load_hybrid_coreness_model(
            Path(args.checkpoints) / (args.dataset + ".pt"), device="cpu"
        )

        def tppr_factory(*positional, **keywords):
            keywords["streaming_only"] = args.mode != "eager"
            return TemporalPPR(*positional, **keywords)

        with patch("hybrid_community.TemporalPPR", side_effect=tppr_factory):
            predictor = HybridCommunityPredictor(
                snapshots, model, checkpoint, hmax=hmax, device="cpu"
            )
        result = {"baseline_rss_mb": baseline_rss,
                  "loaded_rss_mb": rss_mb(),
                  "load_s": time.perf_counter() - started, "times": []}
        start_t = 52 if args.dataset == "mooc" else 133
        # Three consecutive slices in one run, not three repetitions.
        for t in range(start_t, start_t + 3):
            started = time.perf_counter()
            context = predictor.prepare_time(t)
            prepare_s = time.perf_counter() - started
            prepared_rss = rss_mb()
            hashes = {key: digest(value) for key, value in context.feature_table.items()}
            hashes["structure_values"] = digest(context.structure_table.values)
            hashes["tppr_scores"] = digest(predictor.t_ppr_index._state_scores)
            hashes["tppr_nodes"] = digest(predictor.t_ppr_index._state_nodes)
            hashes["tppr_times"] = digest(predictor.t_ppr_index._state_times)
            # Choose the same currently high-core query in both implementations.
            cores = snapshots[t]["core_dict"]
            q = min(cores, key=lambda node: (-cores[node], node))
            queries = []
            for k in range(3, 8):
                selected = predictor.predict(q, k, t, context=context)
                generated = predictor.generate_edges(t, selected.community, k, context=context)
                payload = {
                    "nodes": sorted(selected.community),
                    "predictions": sorted(context.coreness_cache.items()),
                    "edges": generated.edges,
                    "effective_coreness": sorted(generated.effective_coreness.items()),
                    "core_updates": generated.core_updates,
                    "edge_metrics": generated.edge_metrics(),
                    "reduction_metrics": generated.core_reduction_metrics(),
                }
                queries.append({"q": int(q), "k": k,
                                "node_count": len(selected.community),
                                "edge_count": len(generated.edges),
                                "hash": hashlib.sha256(
                                    json.dumps(payload, sort_keys=True).encode()
                                ).hexdigest()})
            result["times"].append({
                "t": t, "prepare_s": prepare_s, "prepared_rss_mb": prepared_rss,
                "after_query_rss_mb": rss_mb(), "hashes": hashes, "queries": queries,
                "resident_times": snapshots.resident_times()
                if hasattr(snapshots, "resident_times") else None,
            })
        result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--dataset", choices=["mooc", "email"])
    parser.add_argument("--mode", choices=["migrate", "eager", "bounded"])
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for dataset in ("mooc", "email"):
        results[dataset] = {}
        for mode in ("migrate", "eager", "bounded"):
            target = output / "{}-{}.json".format(dataset, mode)
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--output", str(target), "--checkpoints", args.checkpoints,
                       "--dataset", dataset, "--mode", mode]
            with (output / "{}-{}.log".format(dataset, mode)).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            results[dataset][mode] = json.loads(target.read_text())
        eager, bounded = results[dataset]["eager"], results[dataset]["bounded"]
        for a, b in zip(eager["times"], bounded["times"]):
            if a["hashes"] != b["hashes"] or a["queries"] != b["queries"]:
                raise AssertionError("{} t={} results changed".format(dataset, a["t"]))
        results[dataset]["exact_match"] = True
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("{}: exact inputs, TPPR, nodes, edges and core updates match".format(dataset),
              flush=True)


if __name__ == "__main__":
    main()
