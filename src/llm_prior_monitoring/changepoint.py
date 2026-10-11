"""ChangepointLearner family: estimate the single most-likely changepoint
in past observations, used to feed `t_hat` into e-process / e-detector updates.

"""

from __future__ import annotations

import abc
import logging
import numpy as np
import ruptures as rpt
import ot


class ChangepointLearner(abc.ABC):
    """Estimate the single most-likely changepoint position in past observations.

    `estimate(past)` returns the 0-indexed split `s` such that `past[:s]` is the
    pre-change segment and `past[s:]` is the post-change segment, or `None` when
    the past is too short / inconclusive. Used to produce the `t_hat` argument
    fed to e-process / e-detector `update(...)` calls (e.g.
    `PermutationEProcess` reads `rho = t_hat / (t + 1)`).
    """

    @abc.abstractmethod
    def estimate(self, past: np.ndarray) -> int | None:
        """Return the estimated 0-indexed changepoint split, or None."""

    @abc.abstractmethod
    def estimate_with_statistic(self, past: np.ndarray) -> tuple[int | None, float]:
        """Return (split, statistic_value) for the best changepoint.

        The statistic value allows comparison across feature subsets for
        joint feature selection and changepoint detection (Enikeeva & Harchaoui 2019).
        Returns (None, -inf) when the past is too short.
        """

    @property
    @abc.abstractmethod
    def min_history(self) -> int:
        """Minimum `len(past)` at/above which `estimate` returns a non-None
        split. The orchestrator asserts `n_burn_in >= min_history` so that
        `t_hat` is guaranteed available once burn-in completes."""

    def whiten(self, x: np.ndarray, eps=0.1) -> np.ndarray:
        x_mean = x.mean(axis=0, keepdims=True)
        centered_x = x - x_mean
        values, counts = np.unique(x, return_counts=True)
        top_2indices = np.argsort(counts)[::-1][:2]
        top_2values = values[top_2indices]
        if (0 in top_2values) and (1 in top_2values):
            # likely bernoulli annotations
            variance_p = np.minimum(x_mean * (1.0 - x_mean) + eps, 0.25)
        else:
            variance_p = np.var(x, axis=0, keepdims=True) + eps
        assert np.all(variance_p > 0)
        
        return centered_x/np.sqrt(variance_p)

class MeanShiftChangepointLearner(ChangepointLearner):
    """Variance-weighted squared-mean-shift scan.

    For each candidate split `s` on a `candidate_step` grid, score
        T(s) = (s * (L - s) / L) * ||mean(past[:s]) - mean(past[s:])||^2
    and return the argmax. The factor `s(L-s)/L` is the precision (inverse
    variance) of the mean-difference estimator under i.i.d. observations, so
    T(s) is (up to a constant) the Gaussian-mean-shift log-likelihood-ratio
    statistic — a single-split scan of the binary-segmentation family.

    Returns `None` when the past is too short to leave both blocks at least
    `min_block` long.
    """

    def __init__(self, min_block: int = 5, candidate_step: int = 1):
        assert min_block >= 1, f"min_block must be positive, got {min_block}"
        assert candidate_step >= 1, (
            f"candidate_step must be positive, got {candidate_step}"
        )
        self.min_block = min_block
        self.candidate_step = candidate_step

    @property
    def min_history(self) -> int:
        return 2 * self.min_block

    def estimate(self, past: np.ndarray) -> int | None:
        split, _ = self.estimate_with_statistic(past)
        return split

    def estimate_with_statistic(self, past: np.ndarray) -> tuple[int | None, float]:
        L = len(past)
        past = self.whiten(past)
        if L < 2 * self.min_block:
            return None, -np.inf
        best_s: int | None = None
        best_stat = -np.inf
        for s in range(self.min_block, L - self.min_block + 1, self.candidate_step):
            diff = past[:s].mean(axis=0) - past[s:].mean(axis=0)
            stat = (s * (L - s) / L) * float(diff @ diff)
            if stat > best_stat:
                best_stat = stat
                best_s = s
        return best_s, best_stat


class _GramCost(rpt.base.BaseCost):
    """ruptures BaseCost backed by a precomputed (cached) Gram matrix.

    Dynp calls fit() once (no-op here), then error(start, end) for each
    candidate segment during the DP.  The Gram is supplied externally via
    set_gram() so Dynp never rebuilds it from raw data.
    """

    model = "cached_gram"
    min_size = 2  # class-level placeholder; overridden per instance below

    def __init__(self, min_block: int):
        assert min_block >= 2, f"min_block must be >= 2, got {min_block}"
        self.min_size = min_block
        self._gram: np.ndarray | None = None

    def set_gram(self, gram: np.ndarray) -> None:
        self._gram = gram

    def fit(self, signal: np.ndarray):
        # ruptures.Dynp.predict() reads self.cost.signal.shape[0] to determine
        # n_samples; we store the reference so that lookup succeeds.
        self.signal = signal
        return self

    def error(self, start: int, end: int) -> float:
        # c(seg) = Σᵢ k(xᵢ,xᵢ) − (1/n_seg) Σᵢ,ⱼ k(xᵢ,xⱼ)
        # Within-segment RKHS variance (Arlot et al. 2012, eq. 2).
        sub = self._gram[start:end, start:end]
        return float(sub.diagonal().sum() - sub.sum() / (end - start))


class KernelCPDLearner(ChangepointLearner):
    """KernelCPD changepoint estimator using cosine or RBF kernel.

    Finds the split τ minimising c(past[:τ]) + c(past[τ:]), where
        c(seg) = Σᵢ k(xᵢ,xᵢ) − (1/n_seg) Σᵢ,ⱼ k(xᵢ,xⱼ)
    is the within-segment RKHS variance.  Equivalently, this maximises
    (n₁n₂/n)·MMD²(past[:τ], past[τ:]) — detecting any distributional
    change representable in the RKHS (Arlot et al. 2012; Garreau & Arlot 2018).

    The Gram matrix is built incrementally: each new observation adds one
    row/column (O(nd)) rather than rebuilding from scratch (O(n²d)).
    Dynamic programming is handled by ruptures.Dynp via a custom BaseCost.

    Args:
        kernel: "cosine" or "rbf".
        min_block: minimum observations required on each side of any split.
        candidate_step: stride between candidate split positions (maps to
            ruptures' `jump`); larger values trade resolution for speed.
        gamma: RBF bandwidth in exp(−gamma·‖x−y‖²).  None = median heuristic,
            recomputed from cached pairwise distances on every estimate() call.
    """

    def __init__(
        self,
        kernel: str = "cosine",
        min_block: int = 5,
        candidate_step: int = 5,
        gamma: float | None = None,
        static_features: bool = True,
    ):
        assert kernel in ("cosine", "rbf"), f"kernel must be 'cosine' or 'rbf', got {kernel!r}"
        assert min_block >= 2, f"min_block must be >= 2, got {min_block}"
        assert candidate_step >= 1, f"candidate_step must be >= 1, got {candidate_step}"
        self.kernel = kernel
        self.min_block = min_block
        self.candidate_step = candidate_step
        self._user_gamma = gamma
        # When features are not static (concept space grows/refits each step), the
        # incremental Gram cache is invalid; reset it every estimate() call.
        self.static_features = static_features

        self._gram: np.ndarray | None = None
        self._dist_sq: np.ndarray | None = None  # RBF only
        self._n_seen: int = 0

        self._gram_cost = _GramCost(min_block)
        self._dynp = rpt.Dynp(
            custom_cost=self._gram_cost,
            min_size=min_block,
            jump=candidate_step,
        )

    @property
    def min_history(self) -> int:
        return 2 * self.min_block

    def _extend_cosine_gram(self, past: np.ndarray) -> None:
        """Grow the cosine Gram matrix by observations past[_n_seen:].

        For cosine kernel, k(x,y) = x·y / (‖x‖‖y‖) and k(x,x) = 1.
        The Gram is built incrementally and reused directly in estimate().
        """
        for i in range(self._n_seen, len(past)):
            x_norm = past[i] / (np.linalg.norm(past[i]) + 1e-12)
            if self._gram is None:
                self._gram = np.array([[1.0]])
            else:
                prev_norms = past[:i] / (np.linalg.norm(past[:i], axis=1, keepdims=True) + 1e-12)
                k_row = prev_norms @ x_norm          # shape (i,)
                self._gram = np.block([
                    [self._gram, k_row[:, None]],
                    [k_row[None, :], np.array([[1.0]])],
                ])

    def _extend_dist_sq(self, past: np.ndarray) -> None:
        """Grow the pairwise squared-Euclidean distance matrix by observations past[_n_seen:].

        For RBF, the Gram depends on gamma which is recomputed on every estimate()
        call (median heuristic).  Caching dist_sq avoids recomputing all O(n²)
        pairwise distances; only the new row/column is computed each step (O(nd)).
        """
        for i in range(self._n_seen, len(past)):
            diffs_sq = np.sum((past[:i] - past[i]) ** 2, axis=1)  # shape (i,)
            if self._dist_sq is None:
                self._dist_sq = np.array([[0.0]])
            else:
                self._dist_sq = np.block([
                    [self._dist_sq, diffs_sq[:, None]],
                    [diffs_sq[None, :], np.array([[0.0]])],
                ])

    def estimate(self, past: np.ndarray) -> int | None:
        split, _ = self.estimate_with_statistic(past)
        return split

    def _build_gram(self, past: np.ndarray) -> np.ndarray:
        """Build/extend the Gram matrix and return it."""
        n = len(past)
        if not self.static_features:
            self._gram = None
            self._dist_sq = None
            self._n_seen = 0

        if self.kernel == "cosine":
            self._extend_cosine_gram(past)
            gram = self._gram
        else:
            self._extend_dist_sq(past)
            gamma = self._user_gamma
            if gamma is None:
                idx = np.triu_indices(n, k=1)
                gamma = 1.0 / np.median(self._dist_sq[idx])
            gram = np.exp(-gamma * self._dist_sq)

        self._n_seen = n
        return gram

    def _mmd_squared(self, gram: np.ndarray, s: int) -> float:
        """Compute (n1*n2/n) * MMD² for split s using the Gram matrix.

        MMD²(X, Y) = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        The (n1*n2/n) weighting corrects for boundary variance.
        """
        n = gram.shape[0]
        n1, n2 = s, n - s
        k_xx = gram[:s, :s].sum() / (n1 * n1)
        k_yy = gram[s:, s:].sum() / (n2 * n2)
        k_xy = gram[:s, s:].sum() / (n1 * n2)
        mmd_sq = k_xx + k_yy - 2 * k_xy
        return (n1 * n2 / n) * mmd_sq

    def estimate_with_statistic(self, past: np.ndarray) -> tuple[int | None, float]:
        n = len(past)
        if n < 2 * self.min_block:
            return None, -np.inf

        gram = self._build_gram(past)

        best_s: int | None = None
        best_stat = -np.inf
        for s in range(self.min_block, n - self.min_block + 1, self.candidate_step):
            stat = self._mmd_squared(gram, s)
            if stat > best_stat:
                best_stat = stat
                best_s = s
        return best_s, best_stat


def _sw_pp(proj_pre: np.ndarray, proj_post: np.ndarray, p: int) -> float:
    """Mean W_p^p over all projection directions.

    Args:
        proj_pre:  (n1, n_dirs) projected pre-segment observations.
        proj_post: (n2, n_dirs) projected post-segment observations.
        p: Wasserstein order (1 or 2).

    Returns:
        Scalar: average over directions of the 1D p-Wasserstein distance^p.
    """
    n1, n2 = len(proj_pre), len(proj_post)
    # Common quantile grid with n1+n2-1 interior points to interpolate both CDFs.
    t = np.linspace(0, 1, n1 + n2 + 1)[1:-1]
    # Sort each direction column; then evaluate quantiles vectorised over directions.
    qa = np.quantile(np.sort(proj_pre, axis=0), t, axis=0)   # (n1+n2-1, n_dirs)
    qb = np.quantile(np.sort(proj_post, axis=0), t, axis=0)  # (n1+n2-1, n_dirs)
    return float(np.mean(np.abs(qa - qb) ** p))


class SlicedWassersteinCPDLearner(ChangepointLearner):
    """Sliced-Wasserstein changepoint estimator.

    Scans candidate splits τ with the statistic
        T(τ) = (n₁n₂/n) · SW_p^p(past[:τ], past[τ:])
    where SW_p^p = (1/n_dirs) Σ_θ W_p^p(θᵀpast[:τ], θᵀpast[τ:]).

    The (n₁n₂/n) factor corrects for boundary variance — small segments have
    noisy SW estimates — so the scan is not biased toward edge or mid splits.
    This mirrors E-Divisive's Q statistic (Matteson & James 2014).

    Two direction modes:
    - "stored": directions fixed after first call (random or user-supplied);
      projections cached incrementally, one new row per step.
    - "random": fresh random directions each call via ot.sliced_wasserstein_distance;
      no projection cache.

    Args:
        n_dirs: number of projection directions.
        directions: (n_dirs, d) array of user-supplied directions.  If given,
            used as the initial stored directions (overrides random generation).
        p: Wasserstein order (1 or 2).
        direction_mode: "stored" or "random".
        min_block: minimum observations required on each side of any split.
        candidate_step: stride between candidate split positions.
        seed: random seed for direction generation and random mode.
    """

    def __init__(
        self,
        n_dirs: int = 20,
        directions: np.ndarray | None = None,
        p: int = 2,
        direction_mode: str = "stored",
        min_block: int = 5,
        candidate_step: int = 5,
        seed: int = 42,
        static_features: bool = True,
    ):
        assert direction_mode in ("stored", "random"), (
            f"direction_mode must be 'stored' or 'random', got {direction_mode!r}"
        )
        assert p in (1, 2), f"p must be 1 or 2, got {p}"
        assert min_block >= 2, f"min_block must be >= 2, got {min_block}"
        assert candidate_step >= 1, f"candidate_step must be >= 1, got {candidate_step}"
        self.n_dirs = n_dirs
        self.p = p
        self.direction_mode = direction_mode
        self.min_block = min_block
        self.candidate_step = candidate_step
        # When features are not static (concept space grows/refits each step), the
        # stored directions + projection cache are invalid; reset them each estimate().
        self.static_features = static_features
        self._rng = np.random.default_rng(seed)

        self._stored_dirs: np.ndarray | None = None
        if directions is not None:
            self._stored_dirs = self._normalise(directions)
        self._proj_cache: np.ndarray | None = None  # (n_seen, n_dirs)
        self._n_seen: int = 0

    @property
    def min_history(self) -> int:
        return 2 * self.min_block

    @staticmethod
    def _normalise(dirs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        assert np.all(norms > 0), "directions must be non-zero"
        return dirs / norms

    def _extend_proj_cache(self, past: np.ndarray) -> None:
        """Project observations past[_n_seen:] onto stored directions and cache."""
        for i in range(self._n_seen, len(past)):
            proj_row = past[i] @ self._stored_dirs.T   # shape (n_dirs,)
            if self._proj_cache is None:
                self._proj_cache = proj_row[None, :]
            else:
                self._proj_cache = np.vstack([self._proj_cache, proj_row])

    def estimate(
        self,
        past: np.ndarray,
        directions: np.ndarray | None = None,
    ) -> int | None:
        """Estimate the single most-likely changepoint split.

        Args:
            past: (n, d) array of past observations (does not include current x_t).
            directions: optional (n_dirs, d) override; updates stored directions
                and invalidates the projection cache.
        """
        split, _ = self.estimate_with_statistic(past, directions)
        return split

    def estimate_with_statistic(
        self,
        past: np.ndarray,
        directions: np.ndarray | None = None,
    ) -> tuple[int | None, float]:
        """Estimate changepoint split and return (split, statistic_value).

        Args:
            past: (n, d) array of past observations (does not include current x_t).
            directions: optional (n_dirs, d) override; updates stored directions
                and invalidates the projection cache.
        """
        n = len(past)
        if n < 2 * self.min_block:
            return None, -np.inf

        if self.direction_mode == "random":
            return self._estimate_random_with_statistic(past)

        # --- stored mode ---
        if not self.static_features:
            self._stored_dirs = None
            self._proj_cache = None
            self._n_seen = 0

        if directions is not None:
            self._stored_dirs = self._normalise(directions)
            self._proj_cache = None
            self._n_seen = 0
        elif self._stored_dirs is None:
            d = past.shape[1]
            raw = self._rng.standard_normal((self.n_dirs, d))
            self._stored_dirs = self._normalise(raw)

        self._extend_proj_cache(past)
        self._n_seen = n

        best_s: int | None = None
        best_T = -np.inf
        for s in range(self.min_block, n - self.min_block + 1, self.candidate_step):
            n1, n2 = s, n - s
            sw_pp = _sw_pp(self._proj_cache[:s], self._proj_cache[s:], self.p)
            T = (n1 * n2 / n) * sw_pp
            if T > best_T:
                best_T = T
                best_s = s
        return best_s, best_T

    def _estimate_random_with_statistic(self, past: np.ndarray) -> tuple[int | None, float]:
        """Random-direction mode: delegate per-split SW to POT."""
        n = len(past)
        best_s: int | None = None
        best_T = -np.inf
        for s in range(self.min_block, n - self.min_block + 1, self.candidate_step):
            n1, n2 = s, n - s
            sw = ot.sliced_wasserstein_distance(
                past[:s], past[s:],
                n_projections=self.n_dirs,
                seed=int(self._rng.integers(1_000_000)),
                p=self.p,
            )
            T = (n1 * n2 / n) * sw ** self.p
            if T > best_T:
                best_T = T
                best_s = s
        return best_s, best_T


class SubsetScanChangepointLearner(ChangepointLearner):
    """Joint feature selection and changepoint detection via subset scanning.

    Implements the sparse changepoint detection approach of Enikeeva & Harchaoui (2019):
    search over all K-subsets of features to find the subset that maximizes the
    changepoint statistic. This breaks the circular dependency between feature
    selection and changepoint estimation.

    For a fixed subset size K, the scan statistic is:
        L_scan(s, m) = statistic(past[:, m], s)
    where m is a K-subset of feature indices and s is the candidate split.
    The optimal (m*, s*) jointly maximizes this over all subsets and splits.

    Args:
        base_learner: ChangepointLearner to use for computing statistics on projected data.
        subset_size: number of features K to select (must be <= total features).
        search_all_sizes: if False, use fixed subset size K; if True, search over
            sizes 1 to K and penalize by subset size (statistic - k).
    """

    def __init__(
        self,
        base_learner: ChangepointLearner,
        subset_size: int,
        search_all_sizes: bool = False,
    ):
        # assert subset_size >= 1, f"subset_size must be >= 1, got {subset_size}"
        self.base_learner = base_learner
        self.subset_size = subset_size
        self.search_all_sizes = search_all_sizes
        self._last_best_subset: np.ndarray | None = None

    @property
    def min_history(self) -> int:
        return self.base_learner.min_history

    def estimate(self, past: np.ndarray) -> int | None:
        split, _, _ = self.estimate_with_subset(past)
        return split

    def estimate_with_statistic(self, past: np.ndarray) -> tuple[int | None, float]:
        split, stat, _ = self.estimate_with_subset(past)
        return split, stat

    def estimate_with_subset(
        self, past: np.ndarray
    ) -> tuple[int | None, float, np.ndarray | None]:
        """Joint changepoint and feature subset estimation.

        Args:
            past: (n, d) array of observations.

        Returns:
            (best_split, best_statistic, best_subset_indices) where best_subset_indices
            is a 1D array of K feature column indices. Returns (None, -inf, None) when
            past is too short.
        """
        n, d = past.shape
        if n < self.base_learner.min_history:
            self._last_best_subset = None
            return None, -np.inf, None
        
        k = min(self.subset_size, d)
        if k == d and not self.search_all_sizes:
            split, stat = self.base_learner.estimate_with_statistic(past)
            self._last_best_subset = np.arange(d)
            return split, stat, self._last_best_subset

        past = self.whiten(past)
        best_split: int | None = None
        best_stat = -np.inf
        best_subset: np.ndarray | None = None
        best_k_prime = 0
        single_feature_max_stats = np.full(d, -np.inf) if self.search_all_sizes else None
        logging.info(f"searching for changepoint {self.search_all_sizes}")
        for s in range(self.base_learner.min_block, n - self.base_learner.min_block + 1):
            mean_before = past[:s].mean(axis=0)
            mean_after = past[s:].mean(axis=0)
            # logging.info(f"split={s}")
            # logging.info(f"mean_before={mean_before}")
            # logging.info(f"mean_after={mean_after}")
            # logging.info(f"diff={mean_after - mean_before}")
            sq_diff = (mean_before - mean_after) ** 2
            weight = s * (n - s) / n

            if self.search_all_sizes:
                single_feature_stats_at_s = weight * sq_diff - 1
                single_feature_max_stats = np.maximum(single_feature_max_stats, single_feature_stats_at_s)
                sorted_idx = np.argsort(-sq_diff)
                cumsum_sq = np.cumsum(sq_diff[sorted_idx])
                for k_prime in range(1, k + 1):
                    stat = weight * cumsum_sq[k_prime - 1] - k_prime
                    logging.info(f"CP split={s}, size={k_prime}, stat={stat}")
                    if stat > best_stat:
                        best_stat = stat
                        best_split = s
                        best_subset = sorted_idx[:k_prime].copy()
                        best_k_prime = k_prime
            else:
                top_k_idx = np.argpartition(-sq_diff, k - 1)[:k]
                stat = weight * sq_diff[top_k_idx].sum()
                if stat > best_stat:
                    best_stat = stat
                    best_split = s
                    best_subset = top_k_idx.copy()
                    best_k_prime = k
        logging.info(f"best_subset T={past.shape[0]}, t_hat={best_split}, {best_subset} {best_stat} {best_k_prime}")
        logging.info(f"single_feature_max_stats {single_feature_max_stats}")

        if self.search_all_sizes and best_k_prime < k and best_subset is not None:
            single_feature_max_stats[best_subset] = -np.inf
            num_to_fill = k - best_k_prime
            fill_idx = np.argpartition(-single_feature_max_stats, num_to_fill - 1)[:num_to_fill]
            best_subset = np.concatenate([best_subset, fill_idx])
            logging.info(f"filled subset to size {k}: {best_subset}")

        self._last_best_subset = best_subset
        return best_split, best_stat, best_subset

    @property
    def last_best_subset(self) -> np.ndarray | None:
        """The feature subset indices from the most recent estimate call."""
        return self._last_best_subset
