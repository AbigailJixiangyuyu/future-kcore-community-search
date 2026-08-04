import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.community_eval_builder import set_metrics
from methods.hcu import predict as predict_hcu

_G_SNAPS = None
_G_STREAMING_TCS = None

_worker_tcs_predict_time = 0.0
_worker_tcs_predict_calls = 0
_worker_hcu_predict_time = 0.0
_worker_hcu_predict_calls = 0


def eval_batch_streaming_tcs(args):
    global _worker_tcs_predict_time, _worker_tcs_predict_calls
    samples, total_nodes = args
    tcs = _G_STREAMING_TCS
    rows = []
    for s in samples:
        t, q, k = s["t"], s["query"], s["k"]
        t0 = time.time()
        pred, q_rejected = tcs.predict(t, q, k)
        _worker_tcs_predict_time += time.time() - t0
        _worker_tcs_predict_calls += 1
        m = set_metrics(pred, s["community"])
        sr = len(pred) / len(s["community"]) if len(s["community"]) > 0 else 0.0
        pr = len(pred) / total_nodes * 100
        rows.append((k, m["f1"], m["precision"], m["recall"], sr, pr, q_rejected))
    return rows, _worker_tcs_predict_time, _worker_tcs_predict_calls


def eval_batch_streaming_hcu(args):
    global _worker_hcu_predict_time, _worker_hcu_predict_calls
    samples, total_nodes = args
    snaps = _G_SNAPS
    rows = []
    for s in samples:
        t0 = time.time()
        pred = predict_hcu(s, snaps)
        _worker_hcu_predict_time += time.time() - t0
        _worker_hcu_predict_calls += 1
        m = set_metrics(pred, s["community"])
        sr = len(pred) / len(s["community"]) if len(s["community"]) > 0 else 0.0
        pr = len(pred) / total_nodes * 100
        rows.append((s["k"], m["f1"], m["precision"], m["recall"], sr, pr))
    return rows, _worker_hcu_predict_time, _worker_hcu_predict_calls
