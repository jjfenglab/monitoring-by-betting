"""Logistic regression whose penalty follows the width of the design matrix.

The SCL hands one classifier to every arm, but the fit width spans two orders of
magnitude: the identity arm fits on the raw embedding (384 columns) while concept
arms fit on a handful of concept columns. A single penalty cannot serve both, so
`DimensionAdaptiveLogistic` picks one per fit from `X.shape[1]`.

sklearn >= 1.8 spelling (this repo pins 1.9.0): `penalty` is deprecated and
`l1_ratio` alone sets the penalty form -- `l1_ratio=0` is ridge, and `C=np.inf`
means no penalty. Do not reintroduce `penalty=`; it raises a FutureWarning and is
removed in 1.10.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


class DirectionFixedLogistic(ClassifierMixin, BaseEstimator):
    """Logistic regression with direction fixed to (post_mean - pre_mean).

    Projects X onto the direction vector, then fits slope + intercept on the
    1D projection. Reports coef_ as the scaled direction for compatibility
    with code that inspects classifier coefficients.
    """

    def __init__(self, solver: str = "lbfgs", max_iter: int = 2000):
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        pre_mean = X[y == 0].mean(axis=0)
        post_mean = X[y == 1].mean(axis=0)
        self.direction_ = post_mean - pre_mean
        print("DIRECTION", self.direction_)

        projections = X @ self.direction_
        self._inner = LogisticRegression(
            C=np.inf, solver=self.solver, max_iter=self.max_iter
        )
        self._inner.fit(projections.reshape(-1, 1), y)

        self.classes_ = self._inner.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._inner.predict((X @ self.direction_).reshape(-1, 1))

    def predict_proba(self, X):
        return self._inner.predict_proba((X @ self.direction_).reshape(-1, 1))

    @property
    def coef_(self):
        return self._inner.coef_[0, 0] * self.direction_.reshape(1, -1)

    @property
    def intercept_(self):
        return self._inner.intercept_


class DimensionAdaptiveLogistic(ClassifierMixin, BaseEstimator):
    """Unpenalized when the design matrix is narrow and tall, ridge otherwise.

    The fit is unpenalized (C=inf) only when BOTH hold:

        n_features <= max_unpenalized_dim     -- few enough inputs to justify it
        n_samples  >= 2 * n_features          -- enough rows to identify them

    anything else is ridge (l1_ratio=0). The choice is remade at every `fit`, so an
    arm whose concept set grows past the cutoff starts being penalized from that step
    on rather than inheriting a decision made when it had two concepts.

    Deciding inside `fit` (rather than at construction) also keeps every fit path in
    agreement: `SequentialConceptLearner` fits this on the full matrix, on the
    restricted removal matrix, and on `clone`d copies inside greedy removal's
    cross-validation. Greedy permutes columns in place, so a clone always sees the
    same width as its parent and picks the same penalty.
    """

    def __init__(
        self,
        max_unpenalized_dim: int = 10,
        C: float = 1.0,
        solver: str = "saga",
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        l1_ratio: float = 0.0,
        cv: bool = False,
        cv_folds: int = 5,
        cv_n_cs = np.power(10, np.arange(5)),
        random_state: int | None = None,
    ):
        # Stored verbatim and unvalidated: sklearn's get_params/clone contract.
        self.max_unpenalized_dim = max_unpenalized_dim
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.cv = cv
        self.cv_folds = cv_folds
        self.cv_n_cs = cv_n_cs
        self.random_state = random_state
        self.l1_ratio = l1_ratio

    def _build_estimator(self, penalized: bool):
        """The inner sklearn model for this fit's penalty decision."""
        if not penalized:
            return LogisticRegression(
                C=np.inf, solver=self.solver, max_iter=self.max_iter,
                class_weight=self.class_weight,
            )
        if self.cv:
            print("self.l1_ratio", self.l1_ratio)
            return LogisticRegressionCV(
                l1_ratios=[self.l1_ratio,], Cs=self.cv_n_cs, cv=self.cv_folds,
                scoring="neg_log_loss", solver=self.solver, max_iter=self.max_iter,
                class_weight=self.class_weight, refit=True,
                random_state=self.random_state, use_legacy_attributes=False,
                n_jobs=-1,
            )
        return LogisticRegression(
            l1_ratio=self.l1_ratio, C=self.C, solver=self.solver, max_iter=self.max_iter,
            class_weight=self.class_weight,
        )

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        # Dropping the penalty needs both a narrow matrix and enough rows to identify
        # it: logistic regression is guaranteed to separate whenever n <= p + 1, so
        # the row condition guards the removal-stage and greedy-CV fits, which run on
        # far fewer rows (as few as 2 * min_block) than the full fit.
        self.penalized_ = not (
            n_features <= self.max_unpenalized_dim and n_samples >= 2 * n_features
        )
        self.estimator_ = self._build_estimator(self.penalized_).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    # Properties, not copies: before fit these raise AttributeError, so the SCL's
    # `hasattr(classifier["clf"], "coef_")` guard reads False on an unfitted model.
    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def intercept_(self):
        return self.estimator_.intercept_
