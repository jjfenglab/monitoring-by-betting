"""Build attribute-weighted CUB p0/p1 image pools over a fixed species population.

Semi-synthetic design shared with the text datasets: a single population of
images (e.g. every seabird photo) supplies both pools; p0 is a uniform draw and
p1 is drawn with probability proportional to sigma(beta0 + beta . a(x)) over
chosen binary CUB attributes, so the changepoint is a shift in attribute
prevalence rather than a switch of species. The population is first split at
random into disjoint p0 and p1 sides, so no image appears both pre- and
post-change: p0 takes n_p0 uniform rows and p1 is tilt-drawn from the rest.
Splitting at random rather than by the tilt keeps both sides unbiased for the
population, which carving p1 out of the whole population would not.

The tilted attributes are the ground-truth concepts. Extra visible attributes
can be declared at beta = 0 as distractors for the oracle arm and as negative
controls for the recovery judge, mirroring assemble_cebab's unshifted aspects.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cub_data import attribute_gloss, class_names, load_class_images
from pool_builder import add_pool_args, run_pool_build, setup_run_logging


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the attribute-weighted CUB pool build."""
    p = argparse.ArgumentParser(
        description="Build and embed attribute-weighted CUB p0/p1 pools.")
    p.add_argument("--cub-dir", type=Path, default=Path("exp_cub/data"),
                   help="Directory holding CUB_200_2011/ and attributes.txt; keep it "
                        "relative to the project root so artifact paths stay portable.")
    p.add_argument("--population-classes", type=int, nargs="+", required=True,
                   help="1-based CUB class ids forming the population both pools draw from.")
    p.add_argument("--tilt-attributes", type=str, nargs="+", required=True,
                   help="CUB attribute names carrying the shift, e.g. has_bill_color::yellow.")
    p.add_argument("--distractor-attributes", type=str, nargs="*", default=[],
                   help="Attributes recorded at beta = 0: oracle-arm distractors and "
                        "recovery-judge negative controls.")
    p.add_argument("--beta", type=float, default=6.0,
                   help="Logit weight applied to each tilted attribute.")
    p.add_argument("--consensus", action="store_true",
                   help="Replace each tilted attribute's image label by (image label AND the "
                        "species majority label), so per-image annotator noise is not part "
                        "of the planted shift. CUB image attributes disagree with their "
                        "species majority on a quarter of seabird images.")
    # No pool-size fallbacks: the sconscript derives --n-p0/--n-p1 from the stream
    # lengths, and a run without them should fail rather than build stale sizes.
    add_pool_args(p)
    p.set_defaults(embed_model="clip-ViT-B-32")
    return p.parse_args()


def main() -> None:
    """Build the attribute-weighted CUB pools: population load, sigma tilt, CLIP embeddings."""
    args = parse_args()
    setup_run_logging(args.log)
    logging.info("args: %s", vars(args))

    df, attr_cols = load_class_images(args.cub_dir, args.population_classes)
    named = args.tilt_attributes + args.distractor_attributes
    unknown = [a for a in named if a not in attr_cols]
    assert not unknown, f"unknown CUB attributes: {unknown}"
    assert not set(args.tilt_attributes) & set(args.distractor_attributes), \
        "an attribute cannot be both tilted and a distractor"
    logging.info("population: %d images over %d classes",
                 len(df), df.class_id.nunique())
    consensus = {}
    if args.consensus:
        for a in args.tilt_attributes:
            class_rate = df.groupby("class_id")[a].transform("mean")
            before = float(df[a].mean())
            df[a] = ((df[a] == 1) & (class_rate >= 0.5)).astype(int)
            n_pos_species = int(df.groupby("class_id")[a].max().sum())
            consensus[a] = {"rule": "image label AND species majority (class rate >= 0.5)",
                            "population_prevalence_before": before,
                            "population_prevalence_after": float(df[a].mean()),
                            "n_positive_species": n_pos_species}
            logging.info("consensus %s: prevalence %.3f -> %.3f over %d positive species",
                         a, before, df[a].mean(), n_pos_species)

    betas = {a: args.beta for a in args.tilt_attributes}
    betas.update({a: 0.0 for a in args.distractor_attributes})
    glosses = {a: attribute_gloss(a) for a in betas}

    provenance = {
        "dataset": "CUB_200_2011",
        "design": "attribute-weighted sampling over a fixed species population",
        "classes": {str(cid): name for cid, name
                    in class_names(args.cub_dir, args.population_classes).items()},
        "population_classes": args.population_classes,
        "population_size": len(df),
        "disjoint_sides": True,
        "tilt_attributes": args.tilt_attributes,
        "distractor_attributes": args.distractor_attributes,
        "ground_truth": "the tilted attributes; beta = the sampling logit weight",
        "consensus": consensus,
    }
    # p0 takes exactly n_p0 uniform rows and p1 is tilt-drawn from the rest, so no
    # image appears both pre- and post-change and p1 keeps every row p0 does not need.
    run_pool_build(df, betas, glosses, provenance, args, is_image=True,
                   disjoint_sides=True)


if __name__ == "__main__":
    main()
