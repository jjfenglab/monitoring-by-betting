"""Assemble CEBaB sae/p0/p1 sets from the embedded population via variant selection.

Consumes population.csv + population_embeddings.npz (built once by
assemble_cebab_population.py). The planted shift is MENTION ADDITION of one
aspect and lives in the variant-selection policy, not in document-level tilted
sampling: the distinct original_ids are split disjointly into sae/p0/p1 sides,
then each original is represented by ONE variant — a uniform choice on the sae
and p0 sides, a sigma(beta0 + beta * mentioned)-weighted choice (beta > 0
prefers aspect-mentioning variants) on the p1 side. The three sides therefore share
no originals (no SAE train/test leak, no cross-pool near-duplicates), and p0
vs p1 differ only in which rewrite represents each review. Sentiment is not
stratified — mention rates are flat across star ratings — but star and
valence balance are logged as diagnostics. Embeddings are sliced from the
population npz, never recomputed.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pool_builder import (
    attribute_prevalences,
    choose_variant_per_group,
    setup_run_logging,
    split_groups,
    write_pool_artifacts,
    write_true_concepts,
)

SHIFT_COLUMNS = [
    "food_mentioned", "food_negative",
    "ambiance_mentioned", "ambiance_negative",
    "service_mentioned", "service_negative",
    "noise_mentioned", "noise_negative",
]
GLOSSES = {
    "food_mentioned":
        "the review comments on the food (either positively or negatively)",
    "food_negative":
        "the review comments negatively on the food",
    "ambiance_mentioned":
        "the review comments on the ambiance or decor "
        "(either positively or negatively)",
    "ambiance_negative":
        "the review comments negatively on the ambiance or decor",
    "service_mentioned":
        "the review comments on the service (either positively or negatively)",
    "service_negative":
        "the review comments negatively on the service",
    "noise_mentioned":
        "the review comments on the noise level "
        "(either positively or negatively)",
    "noise_negative":
        "the review comments negatively on the noise level",
    "review_negative": "the review is negative overall",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble CEBaB sae/p0/p1 sets by variant selection.")
    p.add_argument("--population-csv", type=Path, required=True)
    p.add_argument("--population-embeddings-npz", type=Path, required=True)
    p.add_argument("--shift-aspect", type=str, default="ambiance_mentioned",
                   choices=SHIFT_COLUMNS)
    p.add_argument("--beta-pre", type=float, default=0.0)
    p.add_argument("--beta-post", type=float, default=4.0,
                   help="Variant-choice tilt on the p1 side; positive prefers "
                        "aspect-mentioning variants (mention addition), "
                        "negative prefers aspect-silent ones (removal).")
    p.add_argument("--beta0-post", type=float, default=-4.0)
    p.add_argument("--beta0-pre", type=float, default=0)
    p.add_argument("--side-fracs", type=float, nargs=3, default=[0.2, 0.55, 0.25],
                   metavar=("SAE", "P0", "P1"),
                   help="Disjoint original_id split fractions.")
    p.add_argument("--n-sae", type=int, default=None,
                   help="Optional uniform subsample size (default: full side).")
    p.add_argument("--n-p0", type=int, default=None)
    p.add_argument("--n-p1", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sae-pool-path", type=Path, required=True)
    p.add_argument("--sae-embeddings-path", type=Path, required=True)
    p.add_argument("--p0-pool-path", type=Path, required=True)
    p.add_argument("--p1-pool-path", type=Path, required=True)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--mapping-path", type=Path, required=True)
    p.add_argument("--true-concepts-path", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def subsample(pool: pd.DataFrame, n: int | None, rng: np.random.Generator
              ) -> pd.DataFrame:
    """Uniform without-replacement subsample; full pool when n is None."""
    if n is None:
        return pool
    assert n <= len(pool), f"requested {n} rows from a side of {len(pool)}"
    return pool.iloc[rng.choice(len(pool), size=n, replace=False)].reset_index(
        drop=True)


def log_side(name: str, pool: pd.DataFrame, shift_col: str) -> None:
    """Log diagnostics for the shift column."""
    col_values = pool[shift_col]
    logging.info(
        "%s: n=%d | %s mean=%.3f | mean stars=%.3f",
        name, len(pool), shift_col, col_values.mean(), pool["stars"].mean(),
    )


def main() -> None:
    args = parse_args()
    setup_run_logging(args.log)
    logging.info("args: %s", vars(args))
    rng = np.random.default_rng(args.seed)

    pop = pd.read_csv(args.population_csv)
    pop["row"] = np.arange(len(pop))
    emb = np.load(args.population_embeddings_npz)["emb"]
    assert len(emb) == len(pop), (len(emb), len(pop))

    attr_col = args.shift_aspect
    df = pop[pop[attr_col].notna() & pop["stars"].notna()].copy()
    df["review_negative"] = (df["stars"] <= 2).astype(float)
    logging.info("usable variants: %d of %d (labeled on %s, star majority); "
                 "%d distinct originals",
                 len(df), len(pop), attr_col, df["original_id"].nunique())

    sae_ids, p0_ids, p1_ids = split_groups(
        df["original_id"].values, args.side_fracs, rng)
    sae_pool = choose_variant_per_group(
        df[df["original_id"].isin(sae_ids)], "original_id", rng,
        attr_col=attr_col, beta0=args.beta0_pre, beta=args.beta_pre)
    p0_pool = choose_variant_per_group(
        df[df["original_id"].isin(p0_ids)], "original_id", rng,
        attr_col=attr_col, beta0=args.beta0_pre, beta=args.beta_pre)
    p1_pool = choose_variant_per_group(
        df[df["original_id"].isin(p1_ids)], "original_id", rng,
        attr_col=attr_col, beta0=args.beta0_post, beta=args.beta_post)

    sae_pool = subsample(sae_pool, args.n_sae, rng)
    p0_pool = subsample(p0_pool, args.n_p0, rng)
    p1_pool = subsample(p1_pool, args.n_p1, rng)
    for name, pool in [("sae", sae_pool), ("p0", p0_pool), ("p1", p1_pool)]:
        log_side(name, pool, args.shift_aspect)

    args.sae_pool_path.parent.mkdir(parents=True, exist_ok=True)
    sae_pool.to_csv(args.sae_pool_path, index=False)
    np.savez(args.sae_embeddings_path, emb=emb[sae_pool["row"].values])
    logging.info("saved SAE set (%d rows) → %s, %s",
                 len(sae_pool), args.sae_pool_path, args.sae_embeddings_path)

    write_pool_artifacts(
        p0_pool, p1_pool, emb,
        args.p0_pool_path, args.p1_pool_path,
        args.embeddings_path, args.mapping_path,
    )

    betas_post = {col: 0.0 for col in SHIFT_COLUMNS}
    betas_post[attr_col] = args.beta_post
    betas_post["review_negative"] = 0.0
    betas_pre = {col: 0.0 for col in SHIFT_COLUMNS}
    betas_pre[attr_col] = args.beta_pre
    betas_pre["review_negative"] = 0.0
    provenance = {
        "population_csv": str(args.population_csv),
        "shift": "mention removal via p1-side variant selection",
        "shift_aspect": args.shift_aspect,
        "side_fracs": {"sae": args.side_fracs[0], "p0": args.side_fracs[1],
                       "p1": args.side_fracs[2]},
        "seed": args.seed,
        "n_sae": len(sae_pool), "n_p0": len(p0_pool), "n_p1": len(p1_pool),
        "beta0_pre": args.beta0_pre,
        "beta0_post": args.beta0_post,
        "betas_post": {k: float(v) for k, v in betas_post.items()},
        "betas_pre": {k: float(v) for k, v in betas_pre.items()},
    }
    write_true_concepts(
        args.true_concepts_path, betas_pre, betas_post, GLOSSES,
        args.beta0_pre, args.beta0_post, p0_pool, p1_pool, provenance,
    )


if __name__ == "__main__":
    main()
