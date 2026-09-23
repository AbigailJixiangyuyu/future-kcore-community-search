"""Score candidate links for snapshot t+1 using TFWaveFormer and history through t."""

import argparse
import json
from pathlib import Path

import torch

from methods.tfwaveformer import DEFAULT_ROOT, load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--edge", type=int, nargs=2, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tfwaveformer-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = load_predictor(args.slices_dir, args.checkpoint, args.device, args.tfwaveformer_root)
    predictor.prepare_time(args.t)
    scores = predictor.score_edges(args.edge, batch_size=args.batch_size)
    print(json.dumps(dict(
        metadata=predictor.metadata, target_t=args.t + 1,
        edges=[dict(u=u, v=v, score=float(score)) for (u, v), score in zip(args.edge, scores)],
    ), indent=2))


if __name__ == "__main__":
    main()
