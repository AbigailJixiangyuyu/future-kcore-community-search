#!/usr/bin/env python3
"""Archived StreamingTCS versus HCU community evaluation."""
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.dataset_builder import (
    DATASET_VALID_KS,
    DATASET_WINDOW,
    build_snapshots,
    time_slices_dir,
)
from datasets.community_eval_builder import (
    sample_qk_coreness_weighted,
    set_metrics,
)
from methods.hcu import predict as hcu_predict, reset_hcu_profile, get_hcu_profile
from archive.streaming_tcs.tcs_streaming import StreamingTCS
from archive.streaming_tcs import worker


def _print_table(header, valid_ks, per_k):
    print(f"  {header}")
    print(f"  {'k':>3} | {'F1':>7} | {'Prec':>7} | {'Recall':>7} | {'SizeR':>7} | {'PredR':>7}")
    print(f"  {'---':>3} | {'---':>7} | {'---':>7} | {'---':>7} | {'---':>7} | {'---':>7}")
    nan_row = {"f1": float("nan"), "precision": float("nan"),
               "recall": float("nan"), "size_ratio": float("nan"),
               "pred_ratio": float("nan")}
    mac = {"f1": [], "precision": [], "recall": [],
           "size_ratio": [], "pred_ratio": []}
    for k in sorted(valid_ks):
        row = per_k.get(k, nan_row)
        print(f"  {k:>3} | {row['f1']:>7.4f} | {row['precision']:>7.4f} | "
              f"{row['recall']:>7.4f} | {row['size_ratio']:>7.2f}x | "
              f"{row['pred_ratio']:>6.2f}%")
        for m in mac:
            v = row.get(m, float("nan"))
            if not np.isnan(v):
                mac[m].append(v)
    mac_str = " | ".join(
        f"{np.mean(mac[m]):>7.4f}" if m in ("f1", "precision", "recall")
        else f"{np.mean(mac[m]):>7.2f}x" if m == "size_ratio"
        else f"{np.mean(mac[m]):>6.2f}%"
        for m in ("f1", "precision", "recall", "size_ratio", "pred_ratio")
    )
    print(f"  {'MAC':>3} | {mac_str}")
    print()


def _print_profile_report(elapsed_total, phase_times, tcs_profile, hcu_time_s, hcu_calls,
                          worker_tcs_time, worker_tcs_calls, worker_hcu_time, worker_hcu_calls):
    print(f"\n{'='*60}")
    print("PROFILING REPORT")
    print(f"{'='*60}")
    print(f"  Total streaming eval wall time: {elapsed_total:.2f}s")
    print()
    print("  Phase breakdown:")
    for name, t_val in phase_times.items():
        pct = t_val / elapsed_total * 100 if elapsed_total > 0 else 0
        print(f"    {name:<35s} {t_val:>8.2f}s  ({pct:>5.1f}%)")

    print()
    print("  TCS internals (main process):")
    tcs_p = tcs_profile
    for name, val in tcs_p.items():
        if name.endswith("_calls"):
            print(f"    {name:<35s} {val}")
        else:
            pct = val / elapsed_total * 100 if elapsed_total > 0 else 0
            print(f"    {name:<35s} {val:>8.2f}s  ({pct:>5.1f}%)")

    print()
    print("  HCU (main process):")
    pct_h = hcu_time_s / elapsed_total * 100 if elapsed_total > 0 else 0
    print(f"    {'hcu_calls':<35s} {hcu_calls}")
    print(f"    {'hcu_total_s':<35s} {hcu_time_s:>8.2f}s  ({pct_h:>5.1f}%)")

    print()
    print("  Worker aggregate (sum across all forked workers):")
    pct_wt = worker_tcs_time / elapsed_total * 100 if elapsed_total > 0 else 0
    pct_wh = worker_hcu_time / elapsed_total * 100 if elapsed_total > 0 else 0
    print(f"    {'worker_tcs_predict_calls':<35s} {worker_tcs_calls}")
    print(f"    {'worker_tcs_predict_s':<35s} {worker_tcs_time:>8.2f}s  ({pct_wt:>5.1f}%)")
    print(f"    {'worker_hcu_predict_calls':<35s} {worker_hcu_calls}")
    print(f"    {'worker_hcu_predict_s':<35s} {worker_hcu_time:>8.2f}s  ({pct_wh:>5.1f}%)")
    print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--relax", type=float, default=4, help="relax factor for dynamic tau (None=fixed tau=0.15)")
    parser.add_argument("--step-seconds", type=int, default=None,
                        help="time-slice step; defaults to the dataset's configured window")
    parser.add_argument("--window-seconds", type=int, default=None,
                        help="time-slice window; defaults to the dataset's configured window")
    parser.add_argument("--dataset", choices=sorted(DATASET_VALID_KS), default=None,
                        help="run one dataset instead of the default dataset list")
    args = parser.parse_args()

    default_dataset_names = [
        "email-Eu-core-temporal",
        "mooc",
        # "sx-mathoverflow",
        # "DBLP1",
        "wiki-talk-temporal",
        # "sx-superuser",
    ]
    dataset_names = [args.dataset] if args.dataset else default_dataset_names

    for dn in dataset_names:
        valid_ks = DATASET_VALID_KS.get(dn, [3, 4, 5])
        default_window = DATASET_WINDOW.get(dn, 86400)
        step_seconds = args.step_seconds or default_window
        window_seconds = args.window_seconds or default_window
        slices_dir = time_slices_dir(dn, step_seconds, window_seconds)
        if not slices_dir.is_dir():
            raise FileNotFoundError(
                f"Time slices not found: {slices_dir}. Build them first with: "
                f"python -m datasets.build_time_slices {dn} "
                f"{step_seconds} {window_seconds}"
            )

        print(f"\n{'='*60}")
        print(f"[{dn}] Loading time slices (step={step_seconds}s, window={window_seconds}s) ...")
        t0 = time.time()
        snaps, total_nodes, kmax, _hmax = build_snapshots(slices_dir)
        total_snaps = len(snaps)
        split = int(total_snaps * 0.7)
        if split < 1 or split >= total_snaps - 1:
            raise ValueError(
                f"{dn} needs initialization history plus current and next "
                "snapshots for community evaluation"
            )
        print(f"  {total_snaps} snapshots, split={split} (70/30)")
        print(f"  Init: G_0..G_{split-1}, Stream: G_{split}..G_{total_snaps-1}")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Dataset kmax: {kmax}")
        print(f"  Loading took {time.time()-t0:.1f}s")

        t0 = time.time()
        init_snaps = snaps[:split]
        tcs = StreamingTCS(
            init_snaps,
            valid_ks,
            alpha=0.7,
            tau=0.15,
            total_nodes=total_nodes,
            relax=args.relax,
        )
        print(f"  StreamingTCS initialized in {time.time()-t0:.1f}s, current_t={tcs.current_t}")

        samples = sample_qk_coreness_weighted(
            snaps,
            split,
            valid_ks,
            dataset_name=dn,
            cache_dir=slices_dir / "sample_cache",
        )
        samples = [sample for sample in samples if sample["community"]]
        print(f"  {len(samples)} community samples (non-empty ground truth)")

        samples_by_t = defaultdict(list)
        for s in samples:
            samples_by_t[s["t"]].append(s)

        tcs_by_k = defaultdict(list)
        hcu_by_k = defaultdict(list)

        reset_hcu_profile()

        phase_times = defaultdict(float)
        worker_tcs_time = 0.0
        worker_tcs_calls = 0
        worker_hcu_time = 0.0
        worker_hcu_calls = 0

        print(f"\n  Streaming evaluation ...")
        t_eval_start = time.time()
        max_eval_t = total_snaps - 2
        for t in sorted(samples_by_t.keys()):
            if t > max_eval_t:
                continue

            while t > tcs.current_t:
                ingest_start = time.time()
                tcs.ingest(snaps[tcs.current_t + 1])
                phase_times["ingest"] += time.time() - ingest_start

            batch = samples_by_t[t]

            n_workers = min(24, max(1, len(batch) // 10))
            if n_workers > 1:
                t_fork = time.time()
                worker._G_STREAMING_TCS = tcs
                worker._G_SNAPS = snaps
                chunk_size = (len(batch) + n_workers - 1) // n_workers
                chunks = [(batch[i:i+chunk_size], total_nodes)
                          for i in range(0, len(batch), chunk_size)]
                phase_times["fork_setup"] += time.time() - t_fork

                t_dispatch = time.time()
                with ProcessPoolExecutor(max_workers=n_workers,
                                         mp_context=mp.get_context("fork")) as pool:
                    futs_tcs = [pool.submit(worker.eval_batch_streaming_tcs, c)
                                for c in chunks]
                    futs_hcu = [pool.submit(worker.eval_batch_streaming_hcu, c)
                                for c in chunks]
                    t_wait_tcs = time.time()
                    for fut in as_completed(futs_tcs):
                        rows, wt_tcs, wc_tcs = fut.result()
                        worker_tcs_time += wt_tcs
                        worker_tcs_calls += wc_tcs
                        for k, f1, precision, recall, size_ratio, pred_ratio, _ in rows:
                            tcs_by_k[k].append(
                                (f1, precision, recall, size_ratio, pred_ratio)
                            )
                    phase_times["wait_tcs_workers"] += time.time() - t_wait_tcs

                    t_wait_hcu = time.time()
                    for fut in as_completed(futs_hcu):
                        rows, wt_hcu, wc_hcu = fut.result()
                        worker_hcu_time += wt_hcu
                        worker_hcu_calls += wc_hcu
                        for k, f1, precision, recall, size_ratio, pred_ratio in rows:
                            hcu_by_k[k].append(
                                (f1, precision, recall, size_ratio, pred_ratio)
                            )
                    phase_times["wait_hcu_workers"] += time.time() - t_wait_hcu
                phase_times["parallel_dispatch_total"] += time.time() - t_dispatch
            else:
                t_single = time.time()
                for s in batch:
                    t_pred = time.time()
                    predicted_tcs, _ = tcs.predict(t, s["query"], s["k"])
                    phase_times["single_tcs_predict"] += time.time() - t_pred
                    metrics = set_metrics(predicted_tcs, s["community"])
                    size_ratio = len(predicted_tcs) / len(s["community"])
                    pred_ratio = len(predicted_tcs) / total_nodes * 100
                    tcs_by_k[s["k"]].append((
                        metrics["f1"], metrics["precision"], metrics["recall"],
                        size_ratio, pred_ratio,
                    ))

                    t_hcu = time.time()
                    predicted_hcu = hcu_predict(s, snaps)
                    phase_times["single_hcu_predict"] += time.time() - t_hcu
                    hcu_metrics = set_metrics(predicted_hcu, s["community"])
                    hcu_size_ratio = len(predicted_hcu) / len(s["community"])
                    hcu_pred_ratio = len(predicted_hcu) / total_nodes * 100
                    hcu_by_k[s["k"]].append((
                        hcu_metrics["f1"], hcu_metrics["precision"],
                        hcu_metrics["recall"], hcu_size_ratio, hcu_pred_ratio,
                    ))
                phase_times["single_thread_total"] += time.time() - t_single

        elapsed = time.time() - t_eval_start
        print(f"  Done in {elapsed:.1f}s")

        tcs_per_k = {}
        for k, vals in tcs_by_k.items():
            f1s, precisions, recalls, size_ratios, pred_ratios = zip(*vals)
            tcs_per_k[k] = {
                "f1": float(np.mean(f1s)),
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "size_ratio": float(np.mean(size_ratios)),
                "pred_ratio": float(np.mean(pred_ratios)),
            }

        hcu_per_k = {}
        for k, vals in hcu_by_k.items():
            f1s, precisions, recalls, size_ratios, pred_ratios = zip(*vals)
            hcu_per_k[k] = {
                "f1": float(np.mean(f1s)),
                "precision": float(np.mean(precisions)),
                "recall": float(np.mean(recalls)),
                "size_ratio": float(np.mean(size_ratios)),
                "pred_ratio": float(np.mean(pred_ratios)),
            }

        tau_mode = f"dynamic relax={args.relax}" if args.relax else "fixed tau=0.15"
        print(f"\n[{dn}] Community Results ({tau_mode}, 70/30 split, streaming)")
        _print_table(f"TCS (community, {tau_mode})", valid_ks, tcs_per_k)
        _print_table("HCU (community union)", valid_ks, hcu_per_k)

        hcu_prof = get_hcu_profile()
        # _print_profile_report(
        #     elapsed_total=elapsed,
        #     phase_times=dict(phase_times),
        #     tcs_profile=tcs.profile_summary(),
        #     hcu_time_s=hcu_prof["hcu_total_s"],
        #     hcu_calls=hcu_prof["hcu_calls"],
        #     worker_tcs_time=worker_tcs_time,
        #     worker_tcs_calls=worker_tcs_calls,
        #     worker_hcu_time=worker_hcu_time,
        #     worker_hcu_calls=worker_hcu_calls,
        # )

    print("[Done]")


if __name__ == "__main__":
    main()
