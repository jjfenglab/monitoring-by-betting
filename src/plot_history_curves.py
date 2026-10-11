"""History-sample curves for one aspect: EDD and empirical ARL vs nominal ARL, one line per
`history_sample_size` of a single arm (default oracle_random_pool), the full-history run in black.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDD and calibration curves by history sample size.")
    p.add_argument("--csv", type=Path, required=True, help="all_summary.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-burn-in", type=int, required=True)
    p.add_argument("--n-pre", type=int, required=True, help="guide line EDD = ARL - (n_pre - n_burn_in)")
    p.add_argument("--threshold-max", type=float, default=None, help="drop rows above this nominal ARL")
    p.add_argument("--filter-col", default="oracle_variant", help="column selecting the arm")
    p.add_argument("--filter-val", default="oracle_random_pool")
    p.add_argument("--title-prefix", default="")
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    """Filter to one arm, label the unsampled run "full", draw the two panels."""
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    df = pd.read_csv(args.csv, dtype={"history_sample_size": str})
    d = df[df[args.filter_col].astype(str).eq(args.filter_val)].copy()
    assert len(d), f"no rows with {args.filter_col} == {args.filter_val}"
    if args.threshold_max is not None:
        d = d[d["nominal_arl"] <= args.threshold_max]
    d["N"] = d["history_sample_size"].fillna("full")
    order = sorted(d["N"].unique(), key=lambda n: float("inf") if n == "full" else int(n))
    palette = dict(zip(order, sns.color_palette("viridis", max(len(order) - 1, 1)) + ["black"]))
    logging.info("%d rows, sample sizes %s", len(d), order)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    lo, hi = d["nominal_arl"].min(), d["nominal_arl"].max()
    offset = args.n_pre - args.n_burn_in
    sns.lineplot(d, x="nominal_arl", y="edd", hue="N", hue_order=order, palette=palette, ax=axes[0])
    axes[0].plot([lo, hi], [lo - offset, hi - offset], ":", color="0.4", lw=1)
    axes[0].set(xlabel="Nominal ARL (1/α)", ylabel="EDD", title="Detection delay")
    axes[0].legend(title="History sample", fontsize=8, title_fontsize=8)
    sns.lineplot(d, x="nominal_arl", y="empirical_arl", hue="N", hue_order=order, palette=palette,
                 ax=axes[1], legend=False)
    axes[1].plot([lo, hi], [lo, hi], ":", color="0.4", lw=1)
    axes[1].set(xlabel="Nominal ARL (1/α)", ylabel="Empirical ARL", title="Calibration")
    fig.suptitle(f"{args.title_prefix}: {args.filter_val}, history sample size".lstrip(": "), fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    logging.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
