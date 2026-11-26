"""Online kernel two-sample drift statistics (baselines): sliding-window MMD^2 and LSDD.

Both compare a fixed reference (a stream's controls) with the last W stream points, following
the conventions of alibi-detect 0.13.0's MMDDriftOnline and LSDDDriftOnline (https://github.com/SeldonIO/alibi-detect).
They are re-implemented from the papers:

- MMD^2: Gretton et al., JMLR 2012.
- LSDD: Bu, Alippi & Zhao, IEEE TNNLS 2018, eqs. 7, 11 and 12; Sugiyama et al., Neural
  Computation 2013.

Each detector emits only its statistic, and alarms are decided downstream.

Differences from alibi-detect:
  - No reference bootstrap or ERT thresholds. Thresholds are calibrated on null streams
    (null_calibration.py).
  - The window starts empty and fills with stream points, and the statistic is NaN for
    t < n_burn_in. alibi pre-fills the window with W reference points and scores from t = 1.
  - Every control is used. MMD's reference term uses all of them (alibi holds out 2W - 1).
    LSDD splits them into n_centers kernel centres and a reference (alibi uses 2W random
    centres and drops another 2W - 1 points from the reference).
  - LSDD's lambda is chosen per stream at the end of burn-in, from random splits of its
    reference controls plus its burn-in points, using alibi's grid and relative-difference rule.
    alibi chooses it once, from bootstrap splits of the reference.
  - NumPy and one bandwidth; no preprocess_fn, predict/is_drift or state saving.
"""

from __future__ import annotations

import numpy as np

# alibi-detect's candidate regularisation constants, largest first.
LAMBDA_GRID = 4.0 ** -np.arange(10)
STD_EPS = 1e-12


def compute_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[Nx, Ny] squared Euclidean distances via ||x||^2 + ||y||^2 - 2 x y^T, clamped at 0."""
    sq = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T
    return np.maximum(sq, 0.0)


def compute_median_bandwidth(points: np.ndarray) -> float:
    """2 sigma^2 = median squared distance over distinct pairs of points (median heuristic).

    Takes the lower middle value when the number of pairs is even, as alibi's sigma_median
    does.
    """
    pair_sq = compute_squared_distances(points, points)[np.triu_indices(len(points), k=1)]
    two_sigma2 = float(np.sort(pair_sq)[(len(pair_sq) - 1) // 2])
    assert two_sigma2 > 0, "median squared distance is 0: too many duplicated points"
    return two_sigma2


def compute_rbf_kernel(x: np.ndarray, y: np.ndarray, two_sigma2: float) -> np.ndarray:
    """[Nx, Ny] Gaussian kernel exp(-||x - y||^2 / (2 sigma^2))."""
    return np.exp(-compute_squared_distances(x, y) / two_sigma2)


def compute_offdiag_average(kernel_matrix: np.ndarray) -> float:
    """Average of a square kernel matrix over its off-diagonal entries."""
    n = len(kernel_matrix)
    return float((kernel_matrix.sum() - np.trace(kernel_matrix)) / (n * (n - 1)))


class OnlineMMDDetector:
    """Sliding-window unbiased MMD^2 between a stream's controls and its last W points.

        MMD^2_t = controls_kernel_mean_sq_norm + window_kernel_mean_sq_norm
                  - 2 * cross_kernel_mean_inner_product

    These are the unbiased estimates of ||mu_controls||^2, ||mu_window||^2 and
    <mu_controls, mu_window>, where mu is the kernel mean embedding in the Gaussian RKHS. The
    bandwidth (median heuristic on the controls) and the controls term are fixed at
    construction. Each update costs one kernel column against the controls plus the W x W
    window block.
    """

    def __init__(self, controls: np.ndarray, window_size: int, n_burn_in: int):
        assert 2 <= window_size <= n_burn_in, (
            f"need 2 <= window_size <= n_burn_in, got {window_size=} {n_burn_in=}")
        controls = np.asarray(controls, dtype=float)
        assert len(controls) >= 2, "the controls term needs at least 2 controls"
        self.controls = controls
        self.window_size = window_size
        self.n_burn_in = n_burn_in
        self.two_sigma2 = compute_median_bandwidth(controls)
        self.controls_kernel_mean_sq_norm = compute_offdiag_average(
            compute_rbf_kernel(controls, controls, self.two_sigma2))
        self.window = np.empty((0, controls.shape[1]))
        # cross product terms
        self.k_controls_window = np.empty((len(controls), 0))
        self.history_statistic: list[float] = []

    @property
    def t(self) -> int:
        """Number of stream observations consumed."""
        return len(self.history_statistic)

    def update(self, x_t: np.ndarray) -> float:
        """Push x_t into the window and return MMD^2_t (NaN during burn-in)."""
        self._push_to_window(np.asarray(x_t, dtype=float))
        statistic = float("nan") if self.t < self.n_burn_in else self._compute_mmd2()
        self.history_statistic.append(statistic)
        return statistic

    def _push_to_window(self, x_t: np.ndarray) -> None:
        """Append x_t and its kernel column; drop the oldest point beyond W."""
        k_column = compute_rbf_kernel(self.controls, x_t[None, :], self.two_sigma2)
        self.window = np.vstack([self.window, x_t[None, :]])[-self.window_size:]
        self.k_controls_window = np.hstack([self.k_controls_window, k_column])[:, -self.window_size:]

    def _compute_mmd2(self) -> float:
        """Unbiased MMD^2 over the full window."""
        assert len(self.window) == self.window_size, "window not full"
        window_kernel_mean_sq_norm = compute_offdiag_average(
            compute_rbf_kernel(self.window, self.window, self.two_sigma2))
        cross_kernel_mean_inner_product = self.k_controls_window.mean()
        return float(self.controls_kernel_mean_sq_norm + window_kernel_mean_sq_norm
                     - 2 * cross_kernel_mean_inner_product)


class OnlineLSDDDetector:
    """Sliding-window least-squares density difference between controls and the last W points.

    Controls and stream points are standardized by the controls' per-dimension mean and
    (unbiased) std. The bandwidth comes from the median heuristic on all standardized controls.
    The first n_centers controls are the Gaussian kernel centres c_l, and the rest form the
    reference. With mu the kernel mean embedding:

        h_t    = reference_kernel_mean_at_centres - window_kernel_mean_at_centres     (eq. 7)
        LSDD_t = h_t^T M h_t,   M = 2A - A H A,   A = (H + lambda I)^{-1}              (eq. 11)

    where H_ll' = exp(-||c_l - c_l'||^2 / (4 sigma^2)); the constant (pi sigma^2)^{d/2} is
    dropped, as in alibi. lambda is chosen once, at the end of burn-in (_select_lambda).
    """

    def __init__(self, controls: np.ndarray, window_size: int, n_burn_in: int,
                 n_centers: int, lambda_rd_max: float, n_lambda_splits: int, seed: int):
        assert 2 <= window_size <= n_burn_in, (
            f"need 2 <= window_size <= n_burn_in, got {window_size=} {n_burn_in=}")
        controls = np.asarray(controls, dtype=float)
        assert 1 <= n_centers < len(controls), (
            f"need at least one centre and one reference control, got {n_centers=} "
            f"of {len(controls)} controls")
        self.window_size = window_size
        self.n_burn_in = n_burn_in
        self.n_centers = n_centers
        self.lambda_rd_max = lambda_rd_max
        self.n_lambda_splits = n_lambda_splits
        self.rng = np.random.default_rng(seed)

        self.controls_mean = controls.mean(0)
        self.controls_std = controls.std(0, ddof=1)
        z_controls = self._standardize(controls)
        self.two_sigma2 = compute_median_bandwidth(z_controls)
        self.centres = z_controls[:n_centers]
        assert len(np.unique(self.centres, axis=0)) == n_centers, "duplicate kernel centres"
        # sigma -> sqrt(2) sigma doubles 2 sigma^2.
        self.H = compute_rbf_kernel(self.centres, self.centres, 2 * self.two_sigma2)
        self.k_reference_centres = compute_rbf_kernel(
            z_controls[n_centers:], self.centres, self.two_sigma2)
        self.reference_kernel_mean_at_centres = self.k_reference_centres.mean(0)

        self.k_burn_in_centres: list[np.ndarray] = []
        self.k_window_centres = np.empty((0, n_centers))
        self.lam: float | None = None
        self.mean_rd: np.ndarray | None = None
        self.M: np.ndarray | None = None
        self.history_statistic: list[float] = []

    @property
    def t(self) -> int:
        """Number of stream observations consumed."""
        return len(self.history_statistic)

    def update(self, x_t: np.ndarray) -> float:
        """Push x_t into the window and return LSDD_t (NaN during burn-in).

        lambda is selected on the step that completes burn-in, before any scored step.
        """
        k_row = compute_rbf_kernel(self._standardize(np.asarray(x_t, dtype=float))[None, :],
                                   self.centres, self.two_sigma2)
        self.k_window_centres = np.vstack([self.k_window_centres, k_row])[-self.window_size:]
        if self.t < self.n_burn_in:
            self.k_burn_in_centres.append(k_row[0])
        if self.t == self.n_burn_in - 1:
            self._select_lambda()
        statistic = float("nan") if self.t < self.n_burn_in else self._compute_lsdd()
        self.history_statistic.append(statistic)
        return statistic

    def selection_info(self) -> dict:
        """lambda-selection record for result.pkl."""
        return {
            "lambda": self.lam,
            "lambda_grid": LAMBDA_GRID.tolist(),
            "mean_rd": None if self.mean_rd is None else self.mean_rd.tolist(),
            "lambda_rd_max": self.lambda_rd_max,
            "n_lambda_splits": self.n_lambda_splits,
            "n_centers": self.n_centers,
            "two_sigma2": self.two_sigma2,
        }

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Per-dimension standardization by the controls' mean and std."""
        return (x - self.controls_mean) / (self.controls_std + STD_EPS)

    def _select_lambda(self) -> None:
        """alibi's lambda rule on this stream's pre-scoring null data.

        The pool is the reference controls plus the burn-in points (assumed pre-change). Each
        of n_lambda_splits random splits draws a reference of the reference's size and a window
        of window_size points, giving one h. The relative difference
        RD(lambda) = 1 - theta^T H theta / theta^T h, with theta = (H + lambda I)^{-1} h
        (Bu et al. eq. 12), is averaged over splits. lambda is the largest grid value whose
        mean RD is below lambda_rd_max.
        """
        pool = np.vstack([self.k_reference_centres, np.asarray(self.k_burn_in_centres)])
        n_reference = len(self.k_reference_centres)
        assert len(pool) >= n_reference + self.window_size, "lambda pool too small"
        h_splits = []
        for _ in range(self.n_lambda_splits):
            perm = self.rng.permutation(len(pool))
            h_splits.append(pool[perm[:n_reference]].mean(0)
                            - pool[perm[n_reference:n_reference + self.window_size]].mean(0))
        h_splits = np.asarray(h_splits)
        eye = np.eye(self.n_centers)
        mean_rd = []
        for lam in LAMBDA_GRID:
            theta = np.linalg.solve(self.H + lam * eye, h_splits.T).T
            theta_h = (theta * h_splits).sum(1)
            theta_H_theta = np.einsum("bj,jk,bk->b", theta, self.H, theta)
            mean_rd.append(float(np.mean(1 - theta_H_theta / theta_h)))
        self.mean_rd = np.asarray(mean_rd)
        passing = np.flatnonzero(self.mean_rd < self.lambda_rd_max)
        assert len(passing), (
            f"no lambda in the grid has mean RD < {self.lambda_rd_max}: {self.mean_rd}")
        self.lam = float(LAMBDA_GRID[passing].max())
        A = np.linalg.inv(self.H + self.lam * eye)
        self.M = 2 * A - A @ self.H @ A

    def _compute_lsdd(self) -> float:
        """h^T M h over the full window."""
        assert len(self.k_window_centres) == self.window_size, "window not full"
        window_kernel_mean_at_centres = self.k_window_centres.mean(0)
        h = self.reference_kernel_mean_at_centres - window_kernel_mean_at_centres
        return float(h @ self.M @ h)
