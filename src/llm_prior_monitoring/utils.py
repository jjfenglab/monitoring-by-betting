"""Dependency-light helpers shared across detectors, metrics, and demos.

No heavy module-level imports (sklearn, sentence-transformers): any script can
reuse the formatting and bootstrap helpers without dragging in the detector or
embedding stack. matplotlib is imported lazily inside `get_pyplot`, only when a
caller actually plots.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import json 


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated CLI string into a list of floats."""
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def write_rows(rows: list[dict], out_path: Path) -> None:
    """Write a list of dict rows to CSV with a sorted-union of keys as columns.

    Pure CSV-out: callers that also want plots invoke `plot.plot_summary_rows`
    separately (now split into its own SCons target).
    """
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary -> {out_path}")

# DEPRECATED: 
def format_or_na(value, spec: str = ".2f") -> str:
    """Format a number for console tables, returning 'N/A' for None or NaN."""
    if value is None:
        return "N/A"
    try:
        if np.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{float(value):{spec}}"

# DEPRECATED: 
def _fmt(value) -> str:
    return format_or_na(value, ".1f")

# DEPRECATED: 
def nanmean_or_nan(values: Iterable[float]) -> float:
    """Mean of the finite entries, or NaN if there are none."""
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean, ignoring NaNs."""
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))
    means = vals[idx].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def average_finite(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean of the finite entries along `axis`; NaN (without a warning) where none are."""
    finite = np.isfinite(values)
    with np.errstate(invalid="ignore"):
        return np.where(finite, values, 0.0).sum(axis=axis) / finite.sum(axis=axis)


def bootstrap_curve_bands(
    per_seed: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Pointwise and sup-t simultaneous bootstrap bands for a mean-over-seeds curve.

    `per_seed` is [n_seed, n_point]; whole rows are resampled so each seed's curve stays
    intact and the correlation across points carries into the band. NaN entries drop out
    of each mean. Both bands are est ± q·se with z = |boot - est| / se: q is the per-point
    quantile of z (pointwise) or the quantile of max_point z (simultaneous; Montiel Olea &
    Plagborg-Møller 2019), so simultaneous ⊇ pointwise. Points outside `valid_mask`, with
    se = 0, or with a non-finite resample get NaN bands and are left out of the max.
    """
    n_seed, n_point = per_seed.shape
    bands = {key: np.full(n_point, np.nan) for key in ("pw_lo", "pw_hi", "sim_lo", "sim_hi")}
    bands["crit"] = float("nan")
    if n_seed == 0:
        return bands

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seed, size=(n_bootstrap, n_seed))
    boot = average_finite(per_seed[idx], axis=1)  # [n_bootstrap, n_point]
    est = average_finite(per_seed, axis=0)
    se = boot.std(axis=0)
    usable = valid_mask & np.isfinite(est) & np.isfinite(boot).all(axis=0) & (se > 0)
    if not usable.any():
        return bands

    z = np.abs(boot[:, usable] - est[usable]) / se[usable]
    q_pointwise = np.quantile(z, confidence, axis=0)
    crit = float(np.quantile(z.max(axis=1), confidence))
    bands["pw_lo"][usable] = est[usable] - q_pointwise * se[usable]
    bands["pw_hi"][usable] = est[usable] + q_pointwise * se[usable]
    bands["sim_lo"][usable] = est[usable] - crit * se[usable]
    bands["sim_hi"][usable] = est[usable] + crit * se[usable]
    bands["crit"] = crit
    return bands


# DEPRECATED: 
def extract_json_array(content: str, expected_len: int) -> list:
    """Extract and parse a JSON array from LLM output.

    Tolerates markdown fences and trailing commentary by slicing from the
    first '[' to the last ']'. Raises ValueError on parse failure or length
    mismatch. Adapted from exp_simple/generate_subtle_shift.py.
    """
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Could not find JSON array delimiters in LLM output: {content[:300]}"
        )
    json_str = content[start : end + 1]
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted JSON could not be parsed: {json_str[:300]}") from e
    assert isinstance(result, list) and len(result) == expected_len, (
        f"Expected JSON list of length {expected_len}, got "
        f"{type(result).__name__} of length "
        f"{len(result) if isinstance(result, list) else 'N/A'}"
    )
    return result
