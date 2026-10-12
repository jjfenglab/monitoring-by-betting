"""Synthetic Gaussian-shift stream generators used by simulation experiments.

Controlled feature shifts with known ground truth (true changepoint index and
the indices of the shifted features). Used by `make_data_stream.py`'s synthetic
mode (`make_synthetic_stream`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    """Stream-construction parameters for `make_data_stream.make_synthetic_stream`."""

    dim: int = 20
    n_shift_features: int = 3
    shift_kind: str = "sparse"
    gradual_window: int = 0
    n_pre: int = 100
    n_post: int = 250
    n_controls: int = 15


def true_shift_features(config: SyntheticConfig) -> np.ndarray:
    """Ground-truth indices of the shifted features (all dims if `dense`)."""
    if config.shift_kind == "dense":
        return np.arange(config.dim, dtype=int)
    return np.arange(config.n_shift_features, dtype=int)


def sample_p0(rng: np.random.Generator, n: int, config: SyntheticConfig) -> np.ndarray:
    """Draw `n` i.i.d. standard-normal observations (the null distribution)."""
    return rng.normal(0.0, 1.0, size=(n, config.dim))


def sample_p1(
    rng: np.random.Generator,
    n: int,
    config: SyntheticConfig,
    *,
    delta: float,
) -> np.ndarray:
    """Draw `n` samples from p1: standard normal with `true_shift_features` shifted by `delta`."""
    samples = sample_p0(rng, n, config)
    features = true_shift_features(config)
    samples[:, features] += delta
    return samples
