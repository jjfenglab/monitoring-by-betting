"""Augment oracle pool CSVs with synthetic binary noise concept columns.

Adds two kinds of noise columns for the oracle_random_pool arm:
- noise_ind_<i>:  independent Bernoulli(noise_p) per text
- noise_corr_<i>: noisy copy of --true-col (XOR Bernoulli(flip_rate); NaN where
  the true column is NaN, mirroring ground-truth missingness)

Values are drawn once per unique text over the UNION of all input pools, so
duplicate texts across pools agree by construction (required by OracleState's
duplicate-text assert). Outputs are the input CSVs plus the noise columns.

--zero-fill-cols: ground-truth concept columns whose NaN entries are imputed
as 0 BEFORE noise sampling (aspects CEBaB left unannotated read as "absent").
The oracle extractor leaves NaN unextracted, which breaks remove_then_detect's
fully-annotated-new-columns contract; zero-filling here keeps the
oracle_random_pool arm's pool complete.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(
        description="Add independent and correlated binary noise columns to oracle pool CSVs."
    )
    p.add_argument("--pool-csvs",     type=str, nargs="+", required=True)
    p.add_argument("--out-csvs",      type=str, nargs="+", required=True)
    p.add_argument("--text-column",   type=str, default="observation")
    p.add_argument("--n-independent", type=int, required=True)
    p.add_argument("--noise-p",       type=float, required=True)
    p.add_argument("--n-correlated",  type=int, required=True)
    p.add_argument("--true-col",      type=str, required=True)
    p.add_argument("--zero-fill-cols", type=str, nargs="*", default=[],
                   help="concept columns whose NaN entries are imputed as 0")
    p.add_argument("--flip-rate",     type=float, required=True)
    p.add_argument("--seed",          type=int, required=True)
    p.add_argument("--log",           type=str, required=True)
    return p.parse_args()


def collect_text_to_true(dfs, text_column, true_col):
    """Union of unique texts -> true-column value, asserting duplicates agree."""
    text_to_true = {}
    for df in dfs:
        for text, val in zip(df[text_column], df[true_col]):
            prev = text_to_true.get(text)
            assert prev is None or prev == val or (np.isnan(prev) and np.isnan(val)), (
                f"duplicate pool text with conflicting {true_col}: {text!r}"
            )
            text_to_true[text] = val
    return text_to_true


def sample_noise_values(text_to_true, n_independent, noise_p, n_correlated,
                        flip_rate, rng):
    """Per unique text: (n_independent + n_correlated) binary noise values."""
    texts = list(text_to_true)
    true_vals = np.array([text_to_true[t] for t in texts], dtype=float)
    n = len(texts)
    independent = rng.binomial(1, noise_p, size=(n, n_independent)).astype(float)
    flips = rng.binomial(1, flip_rate, size=(n, n_correlated)).astype(float)
    correlated = np.abs(true_vals[:, None] - flips)  # XOR; NaN truth stays NaN
    return dict(zip(texts, np.concatenate([independent, correlated], axis=1)))


def main():
    args = parse_args()
    logging.basicConfig(
        filename=args.log, level=logging.INFO, filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("args: %s", vars(args))
    assert len(args.pool_csvs) == len(args.out_csvs)

    rng = np.random.default_rng(args.seed)
    dfs = [pd.read_csv(f) for f in args.pool_csvs]
    noise_cols = ([f"noise_ind_{i}" for i in range(args.n_independent)]
                  + [f"noise_corr_{i}" for i in range(args.n_correlated)])
    for path, df in zip(args.pool_csvs, dfs):
        assert args.true_col in df.columns, f"{path} lacks {args.true_col}"
        collision = set(noise_cols) & set(df.columns)
        assert not collision, f"{path} already has noise columns: {collision}"
        missing = [c for c in args.zero_fill_cols if c not in df.columns]
        assert not missing, f"{path} lacks zero-fill columns {missing}"
        for col in args.zero_fill_cols:
            n_na = int(df[col].isna().sum())
            df[col] = df[col].fillna(0.0)
            logging.info("%s: zero-filled %d NaNs in %s", path, n_na, col)

    text_to_true = collect_text_to_true(dfs, args.text_column, args.true_col)
    text_to_noise = sample_noise_values(
        text_to_true, args.n_independent, args.noise_p,
        args.n_correlated, args.flip_rate, rng,
    )

    for df, out_csv in zip(dfs, args.out_csvs):
        noise = np.stack([text_to_noise[t] for t in df[args.text_column]])
        for j, col in enumerate(noise_cols):
            df[col] = noise[:, j]
        df.to_csv(out_csv, index=False)
        logging.info("wrote %s (+%d noise columns, %d rows)",
                     out_csv, len(noise_cols), len(df))


if __name__ == "__main__":
    main()
