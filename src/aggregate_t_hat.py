"""Aggregate t_hat estimates from debug.csv files and compute RMSE vs true changepoint.

Reads the t_hat in the last row of each shift run's debug.csv (the end of the run:
stream end or the --max-m-statistic stop), computes RMSE against N_PRE (true
changepoint), and reports bootstrap 95% CIs.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

SHIFT_SEED_RE = re.compile(r"shift_([0-9.]+)_seed_(\d+)")
CLT_RE = re.compile(r"/clt_([^/]+)/")
CPFEAT_RE = re.compile(r"/cpfeat_([^/]+)/")
CLF_RE = re.compile(r"/clf_([^/]+)/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv-globs", type=str, nargs="+", required=True,
                   help="Glob patterns for debug.csv files from shift runs")
    p.add_argument("--n-pre", type=int, required=True,
                   help="True changepoint (N_PRE)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--groupby-cols", type=str, nargs="+",
                   default=["concept_learner_type"],
                   help="Columns to group by for MSE computation")
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def extract_final_t_hat(path: Path) -> float | None:
    """Extract the final t_hat value from a debug.csv file."""
    df = pd.read_csv(path)
    if "t_hat" not in df.columns:
        return None
    final_t_hat = df["t_hat"].iloc[-1]
    if pd.isna(final_t_hat):
        return None
    return float(final_t_hat)


def parse_path_metadata(path: Path) -> dict:
    """Extract method configuration from the file path."""
    path_str = str(path)
    row = {}

    match = SHIFT_SEED_RE.search(path_str)
    if match:
        row["epsilon"] = float(match.group(1))
        row["seed"] = int(match.group(2))

    match = CLT_RE.search(path_str)
    if match:
        row["concept_learner_type"] = match.group(1)

    match = CPFEAT_RE.search(path_str)
    if match:
        row["cp_feature_source"] = match.group(1)

    match = CLF_RE.search(path_str)
    if match:
        row["classifier"] = match.group(1)

    return row


def bootstrap_rmse_ci(
    errors: np.ndarray,
    n_bootstrap: int,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute RMSE and bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    boot_rmses = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(errors), size=len(errors), replace=True)
        boot_rmses.append(np.sqrt(np.mean(errors[idx] ** 2)))

    alpha = 1 - confidence
    ci_lo = float(np.percentile(boot_rmses, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_rmses, 100 * (1 - alpha / 2)))
    return rmse, ci_lo, ci_hi


def main() -> None:
    args = parse_args()

    input_paths = [Path(p) for gl in args.input_csv_globs
                   for p in sorted(glob.glob(gl, recursive=True))]
    assert input_paths, f"no files match {args.input_csv_globs}"

    rows = []
    for path in input_paths:
        t_hat = extract_final_t_hat(path)
        if t_hat is None:
            continue
        row = parse_path_metadata(path)
        row["t_hat"] = t_hat
        row["error"] = t_hat - args.n_pre
        rows.append(row)

    if not rows:
        print("No valid t_hat values found")
        return

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs with t_hat values")

    results = []
    for group_vals, group_df in df.groupby(args.groupby_cols, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        errors = group_df["error"].values
        rmse, ci_lo, ci_hi = bootstrap_rmse_ci(
            errors, args.n_bootstrap, seed=args.seed
        )

        result = dict(zip(args.groupby_cols, group_vals))
        result["n_runs"] = len(group_df)
        result["rmse"] = rmse
        result["rmse_ci_lo"] = ci_lo
        result["rmse_ci_hi"] = ci_hi
        result["mean_error"] = float(np.mean(errors))
        result["std_error"] = float(np.std(errors))
        results.append(result)

    out_df = pd.DataFrame(results)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
