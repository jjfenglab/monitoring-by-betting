"""Write a CTMConfig JSON file from command-line arguments.

Used by sconscript to generate per-cell config files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_prior_monitoring.conformal_martingale import SCORER_NAMES
from llm_prior_monitoring.ctm_config import CTMConfig
from llm_prior_monitoring.utils import parse_float_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a CTMConfig JSON file.")
    p.add_argument("--scorer", type=str, required=True, choices=SCORER_NAMES)
    p.add_argument("--direction-path", type=str, default=None,
                   help="directions.npz from write_text_direction.py or "
                        "write_pool_direction.py; guided only.")
    p.add_argument("--variant", type=str, default="",
                   help="Arm label for the direction file (text-oracle/...); guided only.")
    p.add_argument("--standardize", action="store_true",
                   help="z-score coordinates before the cosine; for anisotropic embeddings.")
    p.add_argument("--n-reference", type=int, required=True,
                   help="Controls used to fit the scorer; the rest calibrate.")
    p.add_argument("--jump-rates", type=str, required=True,
                   help="Comma-separated; one value = classic Simple Jumper.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-burn-in", type=int, required=True,
                   help="Steps scored into the bag before the martingale starts betting.")
    p.add_argument("--no-sr", dest="use_sr", action="store_false",
                   help="Report the plain wealth E_t instead of the SR statistic.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert (args.variant != "") == (args.scorer == "guided"), \
        "--variant is required for guided and must be empty otherwise"
    config = CTMConfig(
        scorer=args.scorer,
        n_reference=args.n_reference,
        jump_rates=tuple(parse_float_list(args.jump_rates)),
        alpha=args.alpha,
        use_sr=args.use_sr,
        direction_path=args.direction_path,
        standardize=args.standardize,
        variant=args.variant,
        n_burn_in=args.n_burn_in,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config.to_json(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
