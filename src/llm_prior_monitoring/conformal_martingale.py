"""Conformal test martingale: scorer -> conformal p-value -> jumper -> Shiryaev-Roberts.

Standalone baseline for the LLM-guided monitor: consumes raw embeddings, needs no labels,
concepts, or changepoint learner. Layers 1-3 follow Vovk et al., "Retrain or not retrain",
PMLR 152, 2021 (references/Vovk2021-Retrain.pdf), sections 2-3; layer 4 follows Vovk,
"Testing Randomness Online", Statistical Science 36(4), 2021 (references/Vovk2021.pdf), s. 4.

Notation follows this repo, not Vovk: E_t is the betting wealth (his S_n), L_t the
per-step factor E_t/E_{t-1}, M_t the Shiryaev-Roberts statistic (his R_n).

Validity: under exchangeability the conformal p-values are iid U(0,1) (Retrain Prop 1), E is
a positive martingale, and M_t = (M_{t-1} + 1) * L_t satisfies E[tau_1] >= c for any
threshold c > 1 (Vovk2021 Prop 4.2) -- so the threshold is the nominal ARL and needs no
calibration. This holds ONLY if the scorer is fit on data disjoint from the calibration
set; see ConformityScorer.

Burn-in: with n_burn_in > 0 the first n_burn_in observations are scored and DO enter the
p-value bag, but no bet is placed and M_t is recorded as NaN -- the martingale's clock
starts at the first post-burn-in sample. metrics.py subtracts n_burn_in so run lengths are
in that same process time, which is the clock Ville's bound (and hence nominal_arl) refers
to. Mirrors SharedLEDetector's warm-up so the two families are directly comparable.

Sign convention: scores are NONCONFORMITY (higher = stranger) fed into Retrain eq. (1)'s
strictly-less count, so strange observations produce p near 1. Retrain s. 2 notes the Simple
Jumper "will work equally well with -alpha_i in place of alpha_i", and the invariance is
exact here -- flipping the score sign maps p -> 1-p, which leaves every L_t and M_t
bit-identical (see test_sign_convention_is_immaterial_to_the_jumper).
"""

from __future__ import annotations

import abc
import bisect

import numpy as np
from sklearn.metrics import pairwise_distances

EPSILONS = np.array([-1.0, 0.0, 1.0])  # betting directions: typical / abstain / strange
MIN_COORD_STD = 1e-8  # below this a coordinate is dead, not merely narrow


# === Layer 1: conformity scorers (x_t -> alpha_t) ===


class ConformityScorer(abc.ABC):
    """Map an observation to a nonconformity score; higher = stranger.

    Fed to ConformalPValue, which counts strictly-less (Retrain eq. 1), so stranger => p
    near 1. Retrain does the same in its s. 3 covariate-shift example, where a 1-NN Euclidean
    distance is plugged straight into that formula.

    MUST be fit on a reference set disjoint from the calibration set whose scores seed the
    p-value bag. Fitting on the calibration set biases its scores low, biases stream
    p-values high, and inflates the false alarm rate -- measured at 100% separation for the
    centroid at d=384, n_ref=15.
    """

    def __init__(self, metric: str = "cosine"):
        assert metric in ("cosine", "euclidean")
        self.metric = metric

    @abc.abstractmethod
    def fit(self, reference: np.ndarray) -> "ConformityScorer": ...

    @abc.abstractmethod
    def score_batch(self, X: np.ndarray) -> np.ndarray: ...

    def score(self, x: np.ndarray) -> float:
        return float(self.score_batch(x[None, :])[0])


class CentroidScore(ConformityScorer):
    """Distance to the centroid of the reference set."""

    def fit(self, reference: np.ndarray) -> "CentroidScore":
        assert reference.ndim == 2 and len(reference) >= 1
        # keepdims -> (1, d), the 2-D second argument pairwise_distances expects
        self.centroid = reference.mean(axis=0, keepdims=True)
        return self

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        # pairwise_distances((n, d), (1, d)) -> (n, 1); ravel -> one distance per row.
        # metric="cosine" is 1 - cosine similarity; "euclidean" is the L2 norm.
        return pairwise_distances(X, self.centroid, metric=self.metric).ravel()


class NearestNeighborScore(ConformityScorer):
    """Distance to the nearest reference point."""

    def fit(self, reference: np.ndarray) -> "NearestNeighborScore":
        assert reference.ndim == 2 and len(reference) >= 1
        self.reference = reference
        return self

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        # (n, n_ref) distance matrix, then the row-wise minimum. Brute force is right at
        # d=384: kd/ball-trees degrade to brute force in high dimension, and cost per step
        # is O(n_ref * d) with n_ref fixed -- constant in stream length.
        return pairwise_distances(X, self.reference, metric=self.metric).min(axis=1)


class GuidedScore(ConformityScorer):
    """Cosine between (x - control centroid) and one or more anticipated drift directions.

    A direction is any unit vector in embedding space along which the shift is expected to
    move observations -- e.g. embed(OC description) - embed(IC description), or an empirical
    pool-mean difference. With k > 1
    directions the score is the row-wise max, so each observation is judged against whichever
    hypothesis it best matches; the conformal layer absorbs the null distribution of that max,
    so no multiplicity correction is needed.

    SIGNED, unlike the geometric scorers' distances: +1 fully along the drift, -1 fully against
    it, 0 orthogonal. That still reads "higher = stranger" (the direction points at the OC
    description), so it needs no negation -- it is the analogue of Retrain's signed conformity
    measure y - yhat (its eq. 3), which the paper keeps signed rather than taking |.| (eq. 6).

    Unlike the geometric scorers this is SCALE-FREE: a point 10 sigma out along the direction
    scores the same as one 0.1 sigma out, because the deviation is divided by its own norm. All
    magnitude information is discarded, and a shift orthogonal to the direction is invisible no
    matter how large. That is the known weakness of this family.

    Only `reference_center` (and, under standardize, the coordinate scale) is estimated, and
    only from `reference` -- never the calibration half. A text-derived direction touches no
    data at all; keeping an empirical direction disjoint from the monitored stream is the
    caller's responsibility.
    """

    def __init__(self, directions: np.ndarray, standardize: bool = False):
        super().__init__(metric="cosine")
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        # A cosine already divides by ||d_i||, so direction scale cannot change the score at any
        # k. Normalising once here just lets score_batch be a single matmul against a unit
        # deviation instead of dividing by ||d_i|| every call.
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.standardize = standardize

    def fit(self, reference: np.ndarray) -> "GuidedScore":
        assert reference.ndim == 2 and reference.shape[1] == self.directions.shape[1], (
            f"direction dim {self.directions.shape[1]} != embedding dim {reference.shape[1]}: "
            "the direction was built with a different embedding model"
        )
        self.reference_center = reference.mean(axis=0)
        if not self.standardize:
            self._scale = None
            self._directions = self.directions
            return self

        coord_std = reference.std(axis=0)
        # Neutralise dead coordinates instead of dividing by them. `== 0` is not enough:
        # all-MiniLM-L6-v2 coord 223 has std 2.5e-33 -- nonzero in float64, so dividing would
        # amplify float noise there by ~1e33 and swamp every real coordinate in the cosine.
        self._scale = np.where(coord_std < MIN_COORD_STD, 1.0, coord_std)
        # The direction must be carried into the SAME z-scored space as the deviations will be;
        # an angle between vectors in different coordinate systems is meaningless.
        # norm(..., axis=1, keepdims=True) = L2 length of each row, shaped (k, 1) so it
        # broadcasts down the rows -- i.e. divide every row by its own length.
        scaled = self.directions / self._scale
        self._directions = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
        return self

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        dev = X - self.reference_center
        if self._scale is not None:
            dev = dev / self._scale  # raw units -> IC standard deviations, matching _directions
        dev = dev / np.maximum(np.linalg.norm(dev, axis=1, keepdims=True), 1e-12)
        return (dev @ self._directions.T).max(axis=1)


SCORER_REGISTRY = {
    "cosine_centroid": lambda: CentroidScore(metric="cosine"),
    "cosine_nn": lambda: NearestNeighborScore(metric="cosine"),
    "euclidean_centroid": lambda: CentroidScore(metric="euclidean"),
    "euclidean_nn": lambda: NearestNeighborScore(metric="euclidean"),
}

SCORER_NAMES = sorted(SCORER_REGISTRY) + ["guided"]


def build_scorer(name: str, directions=None, standardize: bool = False) -> ConformityScorer:
    """Instantiate by name. guided requires directions; the others must not get any.

    The registry values are zero-arg lambdas, which a directional scorer cannot satisfy, so
    dispatch sits beside the registry rather than inside it.
    """
    if name == "guided":
        assert directions is not None, "scorer 'guided' requires --direction-path"
        return GuidedScore(directions, standardize=standardize)
    assert directions is None, f"scorer '{name}' does not take a direction"
    return SCORER_REGISTRY[name]()


# === Layer 2: conformal p-value, growing bag (alpha_t -> p_t) ===


class ConformalPValue:
    """Smoothed online conformal p-value against a growing bag (Vovk2021-Retrain eq. 1).

        p_t = ( #{bag < alpha_t} + U * #{bag == alpha_t} ) / |bag|

    Scores are NONCONFORMITY (higher = stranger), so a strange observation lands near the top
    of the bag and scores p near 1 -- Retrain's own convention, which uses `<` while calling a
    1-NN distance a "conformity score" (its Section 3, covariate-shift example).

    At stream step n (0-indexed) the bag holds BOTH parts:
        bag_n = {calib scores}  U  {a_0 .. a_n},   |bag_n| = n_calib + n + 1
    The calibration half is fixed forever; the stream half grows by one per step and
    includes the current point. The randomised tie term is required, not cosmetic: without
    it the only exchangeability martingales are a.s. constants (Remark 3.3).

    Both halves are held SORTED so each step costs two O(log n) bisections instead of
    rescanning the whole bag -- the rewrite CTM_NN.R made with findInterval, needed once
    streams reach a few hundred steps. For a sorted array a and value v:
        bisect_left(a, v) = #{a < v},   bisect_right(a, v) = #{a <= v}
    so left is the strictly-less count directly, and #{a == v} = right - left.

    NOTE: stream_sorted is sorted, NOT chronological. It is a counting structure, not a
    history -- reading it as one silently gives scrambled data. The arrival-order score and
    p-value histories live on ConformalTestMartingaleMonitor, which is what debug.csv and
    result["history"] are written from.
    """

    def __init__(self, calib_scores: np.ndarray):
        assert calib_scores.ndim == 1 and len(calib_scores) >= 1
        self.calib_sorted = sorted(float(a) for a in calib_scores)
        self.stream_sorted: list[float] = []

    def update(self, alpha_t: float) -> float:
        bisect.insort(self.stream_sorted, alpha_t)  # bag now includes alpha_t itself
        n_less = n_equal = 0
        for arr in (self.calib_sorted, self.stream_sorted):
            lo = bisect.bisect_left(arr, alpha_t)
            hi = bisect.bisect_right(arr, alpha_t)
            n_less += lo
            n_equal += hi - lo  # >= 1 from alpha_t itself
        n_bag = len(self.calib_sorted) + len(self.stream_sorted)
        return (n_less + np.random.uniform() * n_equal) / n_bag


# === Layer 3: betting (p_t -> L_t) ===


class JumperMartingale:
    """Simple Jumper, Vovk et al., "Retrain or not retrain: Conformal test martingales
    for change-point detection", PMLR 152, 2021, Algorithm 1 and eq. 5.

    Bookkeeping follows EDetector: the wealth path is `history_E`, the per-step factors
    `history_increment`, and `t` / `E` are properties over the former. `update` also
    RETURNS L_t (unlike EProcess.update) so it chains straight into the SR layer.

    Sub-capitals C[k, eps] over jump rates J_k and eps in {-1, 0, +1}, init 1/3.
        mix:  C[k,e] <- (1 - J_k) C[k,e] + (J_k / 3) sum_e C[k,e]
        bet:  C[k,e] <- C[k,e] * (1 + e * (p - 0.5))
        E     = mean_k ( sum_e C[k,e] )
    One jump rate collapses the mean to a plain sum -- exactly the classic Simple Jumper.
    Several give the composite. The K chains are INDEPENDENT and averaged only at the end:
    an average of martingales is a martingale, whereas coupling them through a shared total
    (as WATCH's composite_jumper_martingale does) is not.

    No renormalisation of C: the update is positively homogeneous, so rescaling leaves
    every L_t identical, and the minimum total wealth under H0 is 6.8e-03 at T=1000 and
    5.1e-171 even at T=100_000 -- never near float64 underflow.
    """

    def __init__(self, jump_rates):
        jump_rates = np.asarray(jump_rates, dtype=float).ravel()
        assert len(jump_rates) >= 1
        assert np.all((jump_rates >= 0.0) & (jump_rates <= 1.0))
        self.jump_rates = jump_rates
        self.C = np.full((len(jump_rates), 3), 1.0 / 3)
        self.history_E: list[float] = []
        self.history_increment: list[float] = []

    @property
    def t(self):
        return len(self.history_E)

    @property
    def E(self):
        return self.history_E[-1] if self.history_E else 1.0

    def _compute_increment(self, p_value_t: float) -> float:
        """Per-step multiplicative factor L_t > 0. Mixes, bets, returns the wealth ratio."""
        assert 0.0 <= p_value_t <= 1.0
        E_prev = self.E
        J = self.jump_rates[:, None]
        self.C = (1 - J) * self.C + (J / 3) * self.C.sum(axis=1, keepdims=True)
        self.C = self.C * (1 + EPSILONS * (p_value_t - 0.5))
        return float(self.C.sum(axis=1).mean()) / E_prev

    def update(self, p_value_t: float) -> float:
        L = self._compute_increment(p_value_t)
        assert L > 0, f"L must be positive, got {L}"
        self.history_E.append(self.E * L)
        self.history_increment.append(L)
        return L


# === Layer 4: Shiryaev-Roberts alarm rule (L_t -> M_t) ===


class ConformalSRDetector:
    """Shiryaev-Roberts over a conformal martingale (Vovk eqs. 13-15; his R_n = our M_t).

        M_t = (M_{t-1} + 1) * L_t,   M_0 = 0

    Equivalently Vovk2021-Retrain eq. (8), psi_n = sum_{i=0}^{n-1} S_n / S_i with psi_0 = 0:
    factoring out S_n/S_{n-1} = L_n gives exactly the recursion above. Appendix B of that
    paper is why we recurse rather than evaluate the sum directly -- the naive form underflows
    once S_n decays past 5e-324, visibly breaking around n = 2e5 in its Figure 13.

    Matches SharedLEDetector's value, but as an O(1)-memory scalar recursion: that
    class keeps one e-process per post-burn-in step and sums them (O(t) work per step).

    Threshold semantics: M_t - t is a martingale with M_0 = 0, so E[M_t] = t under H0 and
    E[tau_1] >= threshold for any threshold > 1 (Prop 4.2). The threshold IS the nominal
    ARL; no calibration streams are needed.

    use_sr=False reports the plain wealth E_t instead: exactly valid via Ville,
    P(sup_t E_t >= 1/alpha) <= alpha, but slow to recover after a long quiet prefix.
    """

    def __init__(self, alpha: float = 0.05, use_sr: bool = True,
                 max_m_statistic: float | None = None):
        assert 0 < alpha < 1
        assert max_m_statistic is None or max_m_statistic > 1, (
            f"max_m_statistic must exceed the null level M=1, got {max_m_statistic}"
        )
        self.alpha = alpha
        self.use_sr = use_sr
        self.threshold = 1.0 / alpha
        self.max_m_statistic = max_m_statistic
        self.history_MStatistic: list[float] = []
        self.history_increment: list[float] = []
        # Wealth kept in LOG space, and separately from history_MStatistic: under use_sr the
        # latter is psi_n, not S_n, so this is the only record of the wealth path that
        # estimate_changepoint can read. Log space is Vovk's Appendix B underflow fix -- the
        # Simple Jumper's median capital at T=1e6 is 1e-1720, far below float64's 5e-324, and
        # in linear space the argmin below would degenerate into an arbitrary tie-break.
        self.history_log_E: list[float] = []

    @property
    def t(self):
        return len(self.history_MStatistic)

    @property
    def MStatistic(self):
        # A burn-in step records NaN; treat it as "no wealth yet" so the first real update
        # starts from psi_0 = 0 instead of propagating NaN through the rest of the run.
        if self.history_MStatistic and not np.isnan(self.history_MStatistic[-1]):
            return self.history_MStatistic[-1]
        return 0.0 if self.use_sr else 1.0  # psi_0 := 0 (eq. 8); S_0 := 1 for plain Ville

    @property
    def log_E(self):
        if self.history_log_E and not np.isnan(self.history_log_E[-1]):
            return self.history_log_E[-1]
        return 0.0

    def update(self, L_t: float) -> tuple[float, bool]:
        assert L_t > 0, f"increment must be positive, got {L_t}"
        # Read the previous values through the properties BEFORE appending.
        M = (self.MStatistic + 1.0) * L_t if self.use_sr else self.MStatistic * L_t
        self.history_log_E.append(self.log_E + float(np.log(L_t)))
        self.history_MStatistic.append(M)
        self.history_increment.append(L_t)
        return M, M >= self.threshold

    def skip_step(self) -> None:
        """Record a step the detector did not bet on, keeping every history aligned.

        Callers own the reason (burn-in); this class deliberately holds no n_burn_in, so
        warm-up policy stays with the concrete monitor that needs it. All three histories
        must advance together: `t` counts history_MStatistic, debug.csv indexes with
        arange(t), and metrics.py reads raw-index crossings off the same trajectory.
        """
        self.history_MStatistic.append(float("nan"))
        self.history_increment.append(float("nan"))
        self.history_log_E.append(float("nan"))

    def is_rejected(self) -> bool:
        return self.MStatistic >= self.threshold

    def force_stop(self) -> bool:
        """Return True iff the run may halt: M_t >= max_m_statistic (cost cap)."""
        return (
            self.max_m_statistic is not None
            and self.t > 0
            and self.MStatistic >= self.max_m_statistic
        )

    def estimate_changepoint(self) -> int:
        """argmax_j E_t / E_j == argmin_j log E_j over j = 0 .. t-1 (log E_0 = 0)."""
        assert self.history_log_E, "no observations yet"
        # nanargmin, not argmin: burn-in steps log NaN and argmin would return the
        # first of them as the changepoint.
        return int(np.nanargmin(np.concatenate([[0.0], self.history_log_E[:-1]])))


# === Layer 5: embedding-facing monitor ===


class ConformalTestMartingaleMonitor:
    """Drive scorer -> p-value -> jumper -> SR one observation at a time.

    Re-exposes the EDetector read API (history_MStatistic, history_increment, MStatistic,
    threshold, t, is_rejected, force_stop, estimate_changepoint) so metrics.py and
    plot_e_statistic.py work unchanged. It is NOT an EDetector: it consumes raw x_t,
    not (p_t, p_null, t_hat).

    reference and calibration must be disjoint -- see ConformityScorer.
    """

    def __init__(self, reference, calibration, scorer, jump_rates,
                 alpha: float = 0.05, use_sr: bool = True,
                 max_m_statistic: float | None = None, n_burn_in: int = 0):
        assert len(reference) >= 1 and len(calibration) >= 1
        assert reference.shape[1] == calibration.shape[1]
        assert n_burn_in >= 0, f"n_burn_in must be >= 0, got {n_burn_in}"
        self.n_burn_in = n_burn_in
        self.scorer = scorer.fit(reference)
        self.pvalue = ConformalPValue(self.scorer.score_batch(calibration))
        self.jumper = JumperMartingale(jump_rates)
        self.detector = ConformalSRDetector(
            alpha=alpha, use_sr=use_sr, max_m_statistic=max_m_statistic)
        self.history_score: list[float] = []
        self.history_pvalue: list[float] = []

    def update(self, x_t: np.ndarray) -> tuple[float, bool]:
        alpha_t = self.scorer.score(x_t)
        # Above the burn-in branch on purpose: burn-in samples still join the bag, so the
        # p-value denominator grows from t=0 even though no bet is placed yet.
        p_value_t = self.pvalue.update(alpha_t)
        self.history_score.append(alpha_t)
        self.history_pvalue.append(p_value_t)
        if self.t < self.n_burn_in:
            self.detector.skip_step()
            return float("nan"), False
        return self.detector.update(self.jumper.update(p_value_t))

    # --- EDetector read API passthrough ---
    @property
    def history_MStatistic(self) -> list[float]:
        return self.detector.history_MStatistic

    @property
    def MStatistic(self) -> float:
        return self.detector.MStatistic

    @property
    def history_increment(self) -> list[float]:
        return self.detector.history_increment

    @property
    def threshold(self) -> float:
        return self.detector.threshold

    @property
    def t(self) -> int:
        return self.detector.t

    def is_rejected(self) -> bool:
        return self.detector.is_rejected()

    def force_stop(self) -> bool:
        return self.detector.force_stop()

    def estimate_changepoint(self) -> int:
        return self.detector.estimate_changepoint()
