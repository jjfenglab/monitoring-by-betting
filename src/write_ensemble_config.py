"""
Write a minimal config JSON for the ensemble method.

Produces the fields that aggregate_cell.py reads, with sensible defaults
for an ensemble that doesn't do its own concept learning.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write ensemble config JSON.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument("--concept-learner-type", default="ensemble",
                   help="label that tells ensemble variants apart in summaries and plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "concept_learner_type": args.concept_learner_type,
        "n_burn_in": args.n_burn_in,
        "scl_variant": "subset_scan",
        "noise_rate": 0.0,
        "classifier_type": "logistic",
        "search_all_sizes": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
