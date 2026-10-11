"""Aggregate concept prevalence before/after changepoint across seeds."""
from __future__ import annotations

import argparse
import glob
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-glob", type=str, required=True,
                        help="Glob pattern for result.pkl files")
    parser.add_argument("--changepoint", type=int, required=True,
                        help="True changepoint time (t_oracle)")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def extract_seed_from_path(path: str) -> int:
    """Extract seed number from path like .../seed_42/result.pkl or .../shift_1.0_seed_42/result.pkl"""
    for part in Path(path).parts:
        match = re.search(r"(?:^|_)seed_(\d+)$", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not extract seed from path: {path}")


def compute_prevalence(matrix: np.ndarray, changepoint: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute pre and post changepoint prevalence for each concept."""
    pre_prev = np.nanmean(matrix[:changepoint, :], axis=0)
    post_prev = np.nanmean(matrix[changepoint:, :], axis=0)
    return pre_prev, post_prev


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log)] if args.log else [logging.StreamHandler()],
    )

    pkl_files = sorted(glob.glob(args.result_glob))
    logging.info("Found %d result files matching %s", len(pkl_files), args.result_glob)

    rows = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        seed = extract_seed_from_path(pkl_path)
        if "final_sparse_concept_matrix" not in result:
            continue

        matrix = result["final_sparse_concept_matrix"]
        concepts = result["all_concepts"]
        alarm_fired = result["history"]["stopped_early"]

        pre_prev, post_prev = compute_prevalence(matrix, args.changepoint)

        for j, concept in enumerate(concepts):
            rows.append({
                "seed": seed,
                "concept": concept["concept"],
                "concept_id": concept["id"],
                "pre_prev": pre_prev[j],
                "post_prev": post_prev[j],
                "alarm_fired": alarm_fired,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
