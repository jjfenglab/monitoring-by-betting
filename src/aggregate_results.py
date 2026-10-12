"""Generic CSV concatenator.

Concats one or more cell-summary CSVs into one master CSV. With
`--groupby-cols`, also group-means numeric columns within each group.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate per-cell summary CSVs.")
    p.add_argument("--input-csv", type=Path, nargs="+", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--groupby-cols", type=str, nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.concat([pd.read_csv(f) for f in args.input_csv], ignore_index=True)
    if args.groupby_cols:
        df = df.groupby(args.groupby_cols).mean(numeric_only=True).reset_index()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df)
    print(f"wrote {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
