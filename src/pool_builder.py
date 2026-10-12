"""Dataset-agnostic machinery for building p0/p1 text pools and their artifacts.

A dataset module (assemble_cebab, assemble_yelp, ...) supplies a DataFrame with
an `observation` text column plus binary attribute columns, and calls
run_pool_build with its betas, glosses, and provenance; this module samples the
uniform p0 pool and sigma-weighted p1 pool as independent draws over that
population, writes the pool CSVs, embeds them into embeddings.npz (key "emb",
p0 rows then p1 rows) + mapping.csv, and writes true_concepts.json for the
recovery judge.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data import embed

# Smallest |p1 - p0| a beta-carrying attribute must reach for the build to be real;
# the shipped cebab/yelp/cub configs land at 0.21 to 0.59.
MIN_TILT_GAP = 0.05


def _validate_betas(df: pd.DataFrame, betas: Mapping[str, float]) -> None:
    """Crash unless every beta key is a column of df, listing the offenders."""
    missing = [a for a in betas if a not in df.columns]
    if missing:
        raise ValueError(
            f"beta attributes missing from df columns: {missing}; "
            f"df columns: {list(df.columns)}"
        )


def sample_pools(
    df: pd.DataFrame,
    betas: Mapping[str, float],
    n_p0: int,
    n_p1: int,
    beta0: float,
    rng: np.random.Generator,
    strata_cols: list[str] | None = None,
    disjoint_sides: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw a uniform p0 pool and a sigma-weighted p1 pool from one population.

    p0 estimates the population (the null distribution); p1 reweights it by
    sigma(beta0 + beta . a(x)).

    Without `disjoint_sides` the two are independent without-replacement draws
    over the SAME rows, so the pools may share rows. Carving p1 out first and
    drawing p0 from the remainder instead would deplete p0 of shifted documents
    whenever the pools use up much of the population, silently exaggerating the
    p0/p1 contrast.

    With `disjoint_sides`, a uniform random n_p0 rows become the p0 pool
    outright (p0 is a uniform draw, so the side and the pool coincide) and p1
    is tilt-drawn from the remaining rows. Splitting at random rather than by
    the tilt leaves both sides unbiased for the population, so this avoids that
    depletion AND yields pools sharing no rows; the cost is tilt strength,
    since p1's positives are only those left on its side, which is why p0
    takes no more than it needs. This is the shape assemble_cebab uses via
    split_groups.

    With `strata_cols`, p1 is drawn stratified: stratum sizes follow the
    population's proportions and the sigma tilt applies only WITHIN strata, so
    every strata_cols marginal matches p0 by construction while the beta
    attributes still shift. This is how a latent shift is kept from leaking
    into correlated observable covariates.
    """
    _validate_betas(df, betas)
    if not disjoint_sides:
        p0_frame = p1_frame = df
    else:
        assert len(df) >= n_p0 + n_p1, (
            f"population of {len(df)} rows too small for disjoint pools {n_p0}/{n_p1}")
        order = rng.permutation(len(df))
        p0_frame = df.iloc[order[:n_p0]].reset_index(drop=True)
        p1_frame = df.iloc[order[n_p0:]].reset_index(drop=True)
    attr = p1_frame[list(betas)].values
    logit = beta0 + attr @ np.array(list(betas.values()))
    w = 1.0 / (1.0 + np.exp(-logit))
    if strata_cols is None:
        p1_idx = rng.choice(len(p1_frame), size=n_p1, replace=False, p=w / w.sum())
    else:
        p1_idx = _stratified_choice(p1_frame, w, strata_cols, n_p1, rng)
    p0_idx = rng.choice(len(p0_frame), size=n_p0, replace=False)
    p0_pool = p0_frame.iloc[p0_idx].reset_index(drop=True)
    p1_pool = p1_frame.iloc[p1_idx].reset_index(drop=True)
    return p0_pool, p1_pool


def _stratified_choice(
    df: pd.DataFrame,
    w: np.ndarray,
    strata_cols: list[str],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Within-stratum sigma-tilted draw with stratum sizes apportioned to the population."""
    groups = df.groupby(strata_cols, sort=True).indices
    total = len(df)
    quotas = {k: n * len(ix) / total for k, ix in groups.items()}
    alloc = {k: min(int(q), len(groups[k])) for k, q in quotas.items()}
    remaining = n - sum(alloc.values())
    # Largest-remainder apportionment, capped at stratum capacity.
    for k in sorted(groups, key=lambda k: quotas[k] - int(quotas[k]), reverse=True):
        if remaining == 0:
            break
        if alloc[k] < len(groups[k]):
            alloc[k] += 1
            remaining -= 1
    assert remaining == 0, f"n={n} exceeds stratified capacity across {len(groups)} strata"
    picks = [
        rng.choice(ix, size=alloc[k], replace=False, p=w[ix] / w[ix].sum())
        for k, ix in groups.items()
        if alloc[k] > 0
    ]
    return np.concatenate(picks)


def split_groups(
    group_ids: np.ndarray,
    fracs: list[float],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Partition the unique group ids into disjoint sides with given fractions."""
    assert abs(sum(fracs) - 1.0) < 1e-9, f"fractions must sum to 1, got {fracs}"
    ids = rng.permutation(np.unique(group_ids))
    cuts = np.cumsum([int(round(f * len(ids))) for f in fracs[:-1]])
    return np.split(ids, cuts)


def choose_variant_per_group(
    df: pd.DataFrame,
    group_col: str,
    rng: np.random.Generator,
    attr_col: str | None = None,
    beta0: float = 0.0,
    beta: float = 0.0,
) -> pd.DataFrame:
    """Pick ONE row per group: uniform, or sigma(beta0 + beta*attr)-weighted.

    With attr_col set, rows where the attribute holds are up-weighted (beta >
    0) or down-weighted (beta < 0) within each group — e.g. preferring the
    aspect-mentioning variant of each review.
    """
    work = df.reset_index(drop=True)
    if attr_col is None:
        w = np.ones(len(work))
    else:
        w = 1.0 / (1.0 + np.exp(-(beta0 + beta * work[attr_col].values)))
    picks = [
        rng.choice(idx, p=w[idx] / w[idx].sum())
        for idx in work.groupby(group_col, sort=True).indices.values()
    ]
    return work.iloc[sorted(picks)].reset_index(drop=True)


def attribute_prevalences(
    pool: pd.DataFrame, attr_cols: list[str], joint_cols: list[str] | None = None
) -> dict:
    """Per-attribute prevalences, plus a joint block naming its operands."""
    prevs = {c: float(pool[c].mean()) for c in attr_cols}
    if joint_cols:
        joint = pool[list(joint_cols)].values == 1
        prevs["joint"] = {
            "attributes": list(joint_cols),
            "and": float(joint.all(axis=1).mean()),
            "or": float(joint.any(axis=1).mean()),
        }
    return prevs


def add_pool_args(
    p: argparse.ArgumentParser, n_p0: int | None = None, n_p1: int | None = None,
) -> None:
    """Register the CLI arguments every dataset module shares.

    A pool size left as None has no fallback: the flag becomes required, so a
    caller whose sizes are derived elsewhere (the sconscript) fails loudly
    rather than silently building pools of some stale default size.
    """
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-p0", type=int, default=n_p0, required=n_p0 is None,
                   help="Null pool size.")
    p.add_argument("--n-p1", type=int, default=n_p1, required=n_p1 is None,
                   help="Shifted pool size.")
    p.add_argument("--beta0", type=float, default=-4.0,
                   help="Intercept of the p1 sampling logit.")
    p.add_argument("--embed-model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--local-files-only", action="store_true",
                   help="Resolve the model from the local HF cache only; no Hub call.")
    p.add_argument("--p0-pool-path", type=Path, required=True)
    p.add_argument("--p1-pool-path", type=Path, required=True)
    p.add_argument("--embeddings-path", type=Path, required=True)
    p.add_argument("--mapping-path", type=Path, required=True)
    p.add_argument("--true-concepts-path", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)


def setup_run_logging(log_path: Path) -> None:
    """Route logging to the run's log file (fresh file per run)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
    )


def run_pool_build(
    df: pd.DataFrame,
    betas: Mapping[str, float],
    glosses: dict[str, str],
    provenance: dict,
    args: argparse.Namespace,
    strata_cols: list[str] | None = None,
    is_image: bool = False,
    disjoint_sides: bool = False,
) -> None:
    """Shared driver: sample the pools from df, embed, write all artifacts.

    With is_image=True the `observation` column holds image paths and
    args.embed_model must be an image-capable SentenceTransformer.
    """
    attr_cols = list(betas)
    logging.info("full-pool prevalences: %s", attribute_prevalences(df, attr_cols))

    rng = np.random.default_rng(args.seed)
    p0_pool, p1_pool = sample_pools(
        df, betas, args.n_p0, args.n_p1, args.beta0, rng,
        strata_cols=strata_cols, disjoint_sides=disjoint_sides)
    logging.info("p0 pool prevalences: %s", attribute_prevalences(p0_pool, attr_cols))
    logging.info("p1 pool prevalences: %s", attribute_prevalences(p1_pool, attr_cols))
    # A beta on an attribute that is near-constant in the population yields p1 ~= p0:
    # every downstream stage still runs and reports "no detection", which reads as a
    # negative result rather than a misconfiguration.
    tilt_gaps = {c: abs(p1_pool[c].mean() - p0_pool[c].mean())
                 for c, b in betas.items() if b != 0}
    assert tilt_gaps and min(tilt_gaps.values()) > MIN_TILT_GAP, \
        f"tilt moved no prevalence (need > {MIN_TILT_GAP}): {tilt_gaps}"

    embed_and_write_pools(
        p0_pool, p1_pool, args.embed_model,
        args.p0_pool_path, args.p1_pool_path,
        args.embeddings_path, args.mapping_path,
        is_image=is_image,
        local_files_only=args.local_files_only,
    )
    betas_pre = {c: 0.0 for c in betas}
    write_true_concepts(
        args.true_concepts_path, betas_pre, betas, glosses,
        0.0, args.beta0, p0_pool, p1_pool, provenance,
    )


def embed_and_write_pools(
    p0_pool: pd.DataFrame,
    p1_pool: pd.DataFrame,
    embed_model: str,
    p0_pool_path: Path,
    p1_pool_path: Path,
    embeddings_path: Path,
    mapping_path: Path,
    is_image: bool = False,
    local_files_only: bool = False,
) -> None:
    """Write pool CSVs, embed observations, and save embeddings.npz + mapping.csv.

    With is_image=True the `observation` column holds image paths, embed_model
    must be an image-capable SentenceTransformer (e.g. clip-ViT-B-32), and
    mapping.csv's `text` column carries the paths.
    """
    from sentence_transformers import SentenceTransformer

    p0_pool_path.parent.mkdir(parents=True, exist_ok=True)
    p0_pool.to_csv(p0_pool_path, index=False)
    p1_pool.to_csv(p1_pool_path, index=False)

    model = SentenceTransformer(embed_model, local_files_only=local_files_only)
    model.eval()
    logging.info("embedding with %s", embed_model)

    p0_obs = p0_pool["observation"].tolist()
    p1_obs = p1_pool["observation"].tolist()
    if is_image:
        from PIL import Image

        # sentence-transformers ships CLIP checkpoints under the same
        # SentenceTransformer class, and their encode() accepts PIL images
        p0_emb = embed([Image.open(p).convert("RGB") for p in p0_obs], model)
        p1_emb = embed([Image.open(p).convert("RGB") for p in p1_obs], model)
    else:
        p0_emb = embed(p0_obs, model)
        p1_emb = embed(p1_obs, model)
    all_emb = np.vstack([p0_emb, p1_emb])

    n_p0, n_p1 = len(p0_emb), len(p1_emb)
    mapping_df = pd.DataFrame({
        "row_index": np.arange(n_p0 + n_p1),
        "split_label": ["p0"] * n_p0 + ["p1"] * n_p1,
        "text": p0_obs + p1_obs,
        "source_idx": list(range(n_p0)) + list(range(n_p1)),
        "true_group": [0] * n_p0 + [1] * n_p1,
    })

    np.savez(embeddings_path, emb=all_emb)
    mapping_df.to_csv(mapping_path, index=False)

    logging.info(
        "saved embeddings(%d rows, %d dims) → %s",
        len(all_emb), all_emb.shape[1], embeddings_path,
    )
    logging.info("saved mapping (p0=%d, p1=%d) → %s", n_p0, n_p1, mapping_path)


def write_pool_artifacts(
    p0_pool: pd.DataFrame,
    p1_pool: pd.DataFrame,
    population_emb: np.ndarray,
    p0_pool_path: Path,
    p1_pool_path: Path,
    embeddings_path: Path,
    mapping_path: Path,
    row_col: str = "row",
) -> None:
    """Write pool CSVs + embeddings.npz + mapping.csv from precomputed embeddings.

    Same artifact contract as embed_and_write_pools, but each pool row carries
    `row_col` (its row in the population embedding matrix) and embeddings are
    sliced instead of recomputed — deterministic and torch-free.
    """
    p0_pool_path.parent.mkdir(parents=True, exist_ok=True)
    p0_pool.to_csv(p0_pool_path, index=False)
    p1_pool.to_csv(p1_pool_path, index=False)

    all_emb = np.vstack([
        population_emb[p0_pool[row_col].values],
        population_emb[p1_pool[row_col].values],
    ])
    n_p0, n_p1 = len(p0_pool), len(p1_pool)
    mapping_df = pd.DataFrame({
        "row_index": np.arange(n_p0 + n_p1),
        "split_label": ["p0"] * n_p0 + ["p1"] * n_p1,
        "text": p0_pool["observation"].tolist() + p1_pool["observation"].tolist(),
        "source_idx": list(range(n_p0)) + list(range(n_p1)),
        "true_group": [0] * n_p0 + [1] * n_p1,
    })

    np.savez(embeddings_path, emb=all_emb)
    mapping_df.to_csv(mapping_path, index=False)
    logging.info("saved embeddings(%d rows, %d dims) → %s",
                 len(all_emb), all_emb.shape[1], embeddings_path)
    logging.info("saved mapping (p0=%d, p1=%d) → %s", n_p0, n_p1, mapping_path)


def write_true_concepts(
    path: Path,
    betas_pre: Mapping[str, float],
    betas_post: Mapping[str, float],
    glosses: dict[str, str],
    beta0_pre: float,
    beta0_post: float,
    p0_pool: pd.DataFrame,
    p1_pool: pd.DataFrame,
    provenance: dict,
) -> None:
    """Write ground-truth attributes with beta weights, glosses, and prevalences."""
    attr_cols = list(betas_post)
    shifted = [c for c, b in betas_post.items() if b != 0]
    payload = {
        "beta0_pre": beta0_pre,
        "beta0_post": beta0_post,
        "concepts": [
            {
                "name": c,
                "beta_pre": float(betas_pre[c]),
                "beta_post": float(betas_post[c]),
                "gloss": glosses[c],
            }
            for c in betas_post
        ],
        "provenance": provenance,
        "p0_pool_prevalences": attribute_prevalences(p0_pool, attr_cols, shifted),
        "p1_pool_prevalences": attribute_prevalences(p1_pool, attr_cols, shifted),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info("saved true concepts → %s", path)
