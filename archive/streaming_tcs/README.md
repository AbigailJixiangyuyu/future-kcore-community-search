# Archived StreamingTCS

StreamingTCS is no longer an active community prediction method in this
repository. Its implementation, Numba BFS kernels, multiprocessing worker, and
the original StreamingTCS-versus-HCU evaluation entry point are retained here
only to reproduce earlier experiments.

The active prediction paths are:

- `train_coreness.py` and `methods/hybrid_coreness.py` for next-snapshot node
  coreness prediction and community recovery.
- `zebra_community.py` for Zebra link prediction followed by k-core community
  extraction.

To reproduce the archived experiment from the repository root, install Numba
and run:

```bash
python -m archive.streaming_tcs.streaming_eval
python -m archive.streaming_tcs.streaming_eval --relax 5
```

The archived code still reads the active dataset builders and HCU baseline, so
results can change if those shared components or their input data change.
