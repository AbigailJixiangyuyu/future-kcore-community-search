"""Validate tied/independent linear output-head artifacts and print metrics."""

import argparse
import json
from pathlib import Path

from summarize_history_encoder_ablation import summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(
        args.output_dir, variants=("tied", "linear"), config_key="output_head_type"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
