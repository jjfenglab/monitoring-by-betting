"""Config dataclass for the conformal test martingale baseline.

Serialised per (scorer, jumper) cell as ctm_config.json, the CTM counterpart of
scl_config.json: run_ctm_monitor.py consumes it, aggregate_ctm_cell.py reads it back for
the descriptive columns so every result row is self-describing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CTMConfig:
    """Everything that defines a CTM run except the stream and the RNG seed."""

    scorer: str  # key into conformal_martingale.SCORER_REGISTRY
    n_reference: int  # controls fitting the scorer; the rest calibrate
    jump_rates: tuple[float, ...] = (0.01,)  # one value = classic Simple Jumper (Vovk's J)
    alpha: float = 0.05  # online threshold is 1 / alpha
    use_sr: bool = True  # False reports the plain wealth E_t instead
    direction_path: str | None = None  # directions.npz; required iff scorer == "guided"
    standardize: bool = False  # z-score coordinates before the cosine; anisotropic embeddings
    variant: str = ""  # arm label for the direction file; guided only
    n_burn_in: int = 0  # steps scored into the bag before the martingale starts betting

    def to_json(self, path: str | Path) -> None:
        """Write config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "CTMConfig":
        """Load config from JSON file. JSON has no tuples, so coerce jump_rates back."""
        with open(path) as f:
            d = json.load(f)
        d["jump_rates"] = tuple(d["jump_rates"])
        return cls(**d)
