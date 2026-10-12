"""Slice embeddings.npz down to the rows of one mapping.csv split_label.

Dataset-agnostic: used by exp_ai_scribe to pull out the "p0_sae" slice that
src/assemble_ai_scribe.py carves off the p0 pool, so the SAE trains on a set
of documents disjoint from every stream's "p0" draws (see the
src/assemble_ai_scribe.py module docstring).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--mapping-path", type=Path, required=True)
    p.add_argument("--split-label", required=True, help='e.g. "p0" or "p1"')
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("args: %s", vars(args))

    all_emb = np.load(args.embeddings_path)["emb"]
    mapping_df = pd.read_csv(args.mapping_path)
    rows = mapping_df[mapping_df["split_label"] == args.split_label]
    assert len(rows) > 0, f"no rows with split_label={args.split_label!r} in {args.mapping_path}"
    sliced = all_emb[rows["row_index"].values]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, emb=sliced)
    logging.info("sliced %d/%d rows (split_label=%s) -> %s",
                 len(sliced), len(all_emb), args.split_label, args.out)


if __name__ == "__main__":
    main()
