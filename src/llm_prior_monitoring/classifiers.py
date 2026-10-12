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
from sklearn.linear_model import LogisticRegression


class DimensionAdaptiveLogistic(ClassifierMixin, BaseEstimator):
    """Unpenalized when the design matrix is narrow and tall, ridge otherwise.

    The fit is unpenalized (C=inf) only when BOTH hold:

        n_features <= max_unpenalized_dim     -- few enough inputs to justify it
        n_samples  >= 2 * n_features          -- enough rows to identify them

    anything else is ridge (l1_ratio=0). The choice is remade at every `fit`, so an
    arm whose concept set grows past the cutoff starts being penalized from that step
    on rather than inheriting a decision made when it had two concepts.

    `SequentialConceptLearner` fits this once per update, on the selected concept
    matrix, so the penalty always follows the current number of concepts.
    """

    def __init__(
        self,
        max_unpenalized_dim: int = 10,
        C: float = 1.0,
        solver: str = "saga",
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        l1_ratio: float = 0.0,
    ):
        # Stored verbatim and unvalidated: sklearn's get_params/clone contract.
        self.max_unpenalized_dim = max_unpenalized_dim
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.l1_ratio = l1_ratio

    def _build_estimator(self, penalized: bool):
        """The inner sklearn model for this fit's penalty decision."""
        if not penalized:
            return LogisticRegression(
                C=np.inf, solver=self.solver, max_iter=self.max_iter,
                class_weight=self.class_weight,
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
        # the row condition keeps fits on few rows (e.g. early in the stream)
        # penalized.
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
