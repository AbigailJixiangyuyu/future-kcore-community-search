import time as _time

_total_time = 0.0
_call_count = 0


def predict(s, snaps):
    global _total_time, _call_count
    _call_count += 1
    t0 = _time.time()
    t, q, k = s["t"], s["query"], s["k"]
    predicted = set()
    # t is the latest observed snapshot; predict the community at t + 1.
    for past_t in range(t + 1):
        k_info = snaps[past_t].get("k_core_comps", {}).get(k)
        if k_info is None:
            continue
        if q not in k_info["node_set"]:
            continue
        for comp in k_info["components"]:
            if q in comp:
                predicted.update(comp)
                break
    elapsed = _time.time() - t0
    _total_time += elapsed
    return frozenset(predicted)


def get_hcu_profile():
    return {"hcu_calls": _call_count, "hcu_total_s": _total_time}


def reset_hcu_profile():
    global _total_time, _call_count
    _total_time = 0.0
    _call_count = 0
