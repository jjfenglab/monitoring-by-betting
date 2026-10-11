from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Dict, List
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

# Lazy: imported inside methods only (real LLM mode)
if TYPE_CHECKING:
    from HACHI.src.keyphrase import Keyphrase
    from HACHI.src.ensemble_trainer.concept_generator import StandardConceptGenerator

from .feature_extractors import SparseFeatureExtractor, SparseExtractionMatrix
from .concept_lifecycle import ConceptLifecycleRow


class SequentialConceptLearner(abc.ABC):
    """Sequential concept-based classifier that builds concepts over time.

    1. Concept proposer: Extract keyphrases from new obs, propose new concepts
    2. Annotate (observation, concept) pairs
    3. Train a classifier on concept features
    4. Output predicted probability for the e-process bet

    Abstract base holding the shared machinery. Concrete variants define the per-step
    ordering via `update()` and the x_t annotation policy via `annotate_observations()`.
    SubsetScanSCL is the main variant, using joint concept selection and changepoint detection.
    """

    def __init__(
        self,
        keyphrase_extractor: Keyphrase,
        concept_generator: StandardConceptGenerator,
        concept_selector,
        sparse_feature_extractor: SparseFeatureExtractor,
        classifier: BaseEstimator,
        fixed_calibs: np.ndarray,
        calib_df: pd.DataFrame,
        n_burn_in: int = 30,
        n_init_concepts: int = 10,
        n_new_concepts_per_update: int = 2,
        real_query_budget_ceiling: int | None = None,
        text_column: str = "text",
        seed: int = 0,
        freeze_time: int | None = None,
        removal_rule: str = "none",
        rf_importance: str = "permutation",
        greedy_group_size: int = 1,
        greedy_max_group_candidates: int = 50,
        removal_n_jobs: int = 1,
    ):
        """
        Args:
            keyphrase_extractor: HACHI Keyphrase extractor
            concept_generator: HACHI StandardConceptGenerator
            sparse_feature_extractor: extracts concept values from text
            classifier: sklearn-compatible classifier (e.g. LogisticRegression)
            text_column: name of text column in data_df
            fixed_calibs: calib-pool embeddings
            calib_df: calib-pool DataFrame with text_column and y_true columns
        """
        self.keyphrase_extractor = keyphrase_extractor
        self.concept_generator = concept_generator

        self.concept_selector = concept_selector
        self.sparse_feature_extractor = sparse_feature_extractor
        self.classifier = classifier
        self.n_burn_in = n_burn_in
        self.n_init_concepts = n_init_concepts
        self.n_new_concepts_per_update = n_new_concepts_per_update
        # Fixed-size regime: once len(history) > freeze_time, each step removes the
        # n_new weakest concepts (by removal_rule) after testing (Option Y). The step
        # AT freeze_time proposes the initial batch and removes nothing, so the pool
        # rests at n_init. None = off.
        assert removal_rule in ("none", "importance", "greedy", "random"), (
            f"Unknown removal_rule: {removal_rule!r}"
        )
        assert rf_importance in ("permutation", "impurity"), (
            f"Unknown rf_importance: {rf_importance!r}"
        )
        assert greedy_group_size in (1, n_new_concepts_per_update), (
            f"greedy_group_size must be 1 or n_new_concepts_per_update, got {greedy_group_size}"
        )
        self.freeze_time = freeze_time
        self.removal_rule = removal_rule
        self.rf_importance = rf_importance
        self.greedy_group_size = greedy_group_size
        self.greedy_max_group_candidates = greedy_max_group_candidates
        self.removal_n_jobs = removal_n_jobs
        # Real mode only: cap on expected per-step extraction queries (None = uncapped, mock).
        self.real_query_budget_ceiling = real_query_budget_ceiling
        self.text_column = text_column
        self.seed = seed


        self.all_concepts: List[Dict] = []
        self.sparse_concept_matrix: SparseExtractionMatrix = SparseExtractionMatrix((0, 0))

        # Calibration dataset
        self.fixed_calibs = fixed_calibs
        assert text_column in calib_df.columns, f"calib_df must have column '{text_column}'"
        assert "y_true" in calib_df.columns, "calib_df must have column 'y_true'"
        self.calib_df = calib_df.copy()
        self.n_calib = calib_df.shape[0]
        self.last_fit_converged: bool = True  # set each refit; read for debug diagnostics
        # Per-step fit-size counters for the timing audit; read by _record_debug.
        # Fit-based keys stay NaN during burn-in (no fit); keyphrase_row_idx is set every step.
        self.last_fit_debug: dict = {}

        # Every concept dict carries a stable "id" (from _next_concept_id) that rides
        # along when all_concepts is filtered on removal.
        self._next_concept_id: int = 0
        self.last_added_concepts: List[Dict] = []     # this step's proposals (debug)
        self.last_removed_concepts: List[Dict] = []   # this step's removals (debug)

        # Lifecycle diagnostics. Attached by the monitor after construction; None
        # means the SCL records nothing (identity, or the artifact is disabled).
        # `_next_candidate_id` numbers scored removal candidates across the run.
        self.episode_recorder = None
        self.removal_recorder = None
        self._next_candidate_id: int = 0
        # concept_id -> candidate_id of the candidate that removed it, this step.
        self.last_removal_candidate_id: dict[int, int] = {}

        # Extract keyphrases for calibration pool at init
        self.calib_df = self._ensure_keyphrases(self.calib_df)
        assert np.all(self.calib_df["llm_output"] != "")

    @property
    def n_concepts(self):
        return len(self.all_concepts)

    @abc.abstractmethod
    def update(
        self,
        structured_history: np.ndarray,
        unstructured_history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run one per-step SCL update (variant-specific ordering) and refit the classifier.

        Args:
            structured_history: embeddings of all past observations, shape (t, d).
                     Owned by AdaptiveMonitor; not stored inside SCL.
            unstructured_history_df: history rows (len == history.shape[0]) with columns:
                - text_column, llm_output, y (estimated changepoint labels),
                  y_true (true labels, mock mode only)

        Returns:
            unstructured_history_df with keyphrases added for the last row (new object).
        """

    @abc.abstractmethod
    def annotate_observations(self, obs_row: dict) -> np.ndarray:
        """Annotate a fresh observation, extracting keyphrases if needed.

        Mutates obs_row["llm_output"] in place if not already populated.

        Args:
            obs_row: Dict with text_column key (and optionally llm_output, y_true).

        Returns:
            Concept features for the observation, shape (1, n_concepts).
        """

    def _ensure_keyphrases(self, df: pd.DataFrame, rows: np.ndarray | None = None) -> pd.DataFrame:
        """Extract keyphrases for rows, populating llm_output column.

        Args:
            df: DataFrame with text_column.
            rows: Row indices to extract (default: all rows).

        Returns:
            DataFrame with llm_output column populated for specified rows.
        """
        pending = rows if rows is not None else np.arange(len(df))
        result = self.keyphrase_extractor.extract(df, pending)
        if "index" in result.columns:
            result = result.drop(columns=["index"])
        return result.reset_index(drop=True)

    def _prepare_step_frame(
        self,
        structured_history: np.ndarray,
        unstructured_history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keyphrase the pending past rows and reset the per-step debug counters."""
        curr_n_obs = structured_history.shape[0]
        assert curr_n_obs == len(unstructured_history_df), (
            f"history rows ({curr_n_obs}) must match data_df rows ({len(unstructured_history_df)})"
        )
        # update() is first called at t = n_burn_in with no row yet keyphrased, so the
        # whole warm-up batch is pending; every later step adds exactly one row. Without
        # the backfill the initial proposal fits on a corpus of one document.
        pending = (np.arange(curr_n_obs) if curr_n_obs == self.n_burn_in
                   else np.array([curr_n_obs - 1]))
        data_df = self._ensure_keyphrases(unstructured_history_df, pending)

        # Timing-audit counters: keyphrase touches the last past row (curr_n_obs - 1)
        # every step except the first, which backfills the whole warm-up batch;
        # fit-based counts stay NaN until burn-in completes below.
        self.last_fit_debug = {
            "keyphrase_row_idx": curr_n_obs - 1,
            "n_keyphrase_rows": len(pending),
            "n_obs_scl": curr_n_obs,
            "imputer_n_train": float("nan"),
            "clf_n_train": float("nan"),
            "imputer_best_config": None,
        }
        return data_df

    def _propose_refine_concepts(self, structured_history: np.ndarray, data_df: pd.DataFrame,
                                  rows: np.ndarray | None = None) -> list:
        """Propose new concepts and/or refine existing ones.

        Generators return a list of concept dicts. Each dict may have:
        - "updates_idx": index into all_concepts to update (refinement)
        - No "updates_idx": a new concept to add

        Returns only the new concepts (caller adds to all_concepts).
        Refinements are applied in-place to all_concepts.
        """
        curr_n_obs = structured_history.shape[0]

        self.sparse_concept_matrix.resize((curr_n_obs, self.n_concepts))
        if rows is not None:
            data_df = data_df.iloc[rows].reset_index(drop=True)

        if curr_n_obs == self.n_burn_in:
            proposed = self.concept_generator.generate_initial_concepts(
                init_seed=self.seed,
                data_df=data_df,
                concept_selector=self.concept_selector,
            )
        else:
            concept_features = self.sparse_concept_matrix.to_dense()
            if rows is not None:
                concept_features = concept_features[rows]
            current_extractions = self._extractions_to_dict(concept_features)
            proposed = self.concept_generator.generate_candidate_concepts(
                init_data_df=data_df,
                concept_selector=self.concept_selector,
                current_concepts=self.all_concepts,
                current_extractions=current_extractions,
            )

        updates = [c for c in proposed if "updates_idx" in c]
        new_concepts = [c for c in proposed if "updates_idx" not in c]

        for update in updates:
            idx = update["updates_idx"]
            old_concept = self.all_concepts[idx]
            old_name = old_concept["concept"]
            new_name = update["concept"]
            if old_name != new_name:
                logging.info(f"REFINED concept {idx}: '{old_name}' -> '{new_name}'")
            old_concept["concept"] = new_name
            if "keyphrases" in update:
                old_concept["keyphrases"] = update["keyphrases"]

        if curr_n_obs > self.n_burn_in and new_concepts and self.n_new_concepts_per_update > 0:
            if len(new_concepts) > self.n_new_concepts_per_update:
                keep = np.random.choice(
                    len(new_concepts), size=self.n_new_concepts_per_update, replace=False)
                logging.info(
                    "subsampling %d of %d proposed concepts", self.n_new_concepts_per_update,
                    len(new_concepts))
                new_concepts = [new_concepts[i] for i in sorted(keep)]

        for c in new_concepts:
            c["id"] = self._next_concept_id
            c["proposed_t"] = curr_n_obs
            self._next_concept_id += 1
            logging.info(f"NEW concept {c['id']}: {c['concept']}")
            if self.episode_recorder is not None:
                self.episode_recorder.open_episode(
                    concept_id=c["id"], concept=c["concept"],
                    proposed_t=curr_n_obs, removal_rule=self.removal_rule,
                )

        for concept in self.all_concepts:
            logging.info(f"all_concepts: {concept}")

        return new_concepts

    def _fit_classifier_on(self, features: np.ndarray, y: np.ndarray) -> None:
        """Fit `self.classifier` on the given design matrix; record convergence.

        Convergence warnings are captured (record=True), never printed; the flag
        reaches debug.csv (classifier_converged / restricted_fit_converged)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.classifier.fit(features, y)
            if hasattr(self.classifier["clf"], "coef_"):
                coef_df = pd.DataFrame({
                    "concept": [c_dict["concept"] for c_dict in self.all_concepts],
                    "coef": self.classifier["clf"].coef_.flatten(),
                })
                logging.info(f"coef concepts: {coef_df}")
        self.last_fit_converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )

    @abc.abstractmethod
    def get_concept_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Get concept features from embeddings or annotations.

        Args:
            embeddings: observation embeddings (n_samples, embedding_dim)

        Returns:
            concept features (n_samples, n_concepts)
        """

    @abc.abstractmethod
    def predict_proba(self, embeddings: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
        """Predict class probabilities.

        Args:
            embeddings: observation embeddings (n_samples, embedding_dim)
            observed: optional (n_samples, n_concepts) annotations

        Returns:
            predicted probabilities (n_samples, n_classes)
        """

    @abc.abstractmethod
    def predict_proba_null(self) -> float:
        """p_null for the bet L = 1 + lam * (p_t - p_null), scored on calibration rows."""

    def linear_coef(self) -> np.ndarray | None:
        """The fitted linear coefficients (per concept), or None for non-linear models.

        Logistic classifiers are wrapped in a Pipeline([scaler, LR]); reach the final
        estimator for `coef_`. Random forests have no `coef_`, so this returns None.
        """
        final = self.classifier[-1] if isinstance(self.classifier, Pipeline) else self.classifier
        return getattr(final, "coef_", None)

    def _extractions_to_dict(self, feature_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert feature matrix to extractions dict format expected by concept generators."""
        if feature_matrix is None:
            return {}
        extractions = {}
        for i, concept in enumerate(self.all_concepts):
            extractions[concept["concept"]] = feature_matrix[:, i].reshape((-1, 1))
        return extractions

    def _extract_masked_cells(self, data_df: pd.DataFrame, mask) -> SparseExtractionMatrix:
        """Execute an annotation mask: extract the masked (row, concept) cells from
        `data_df` and return the sparse delta. Shared by every annotation channel;
        callers merge the delta into their matrix (or read it directly for x_t)."""
        return self.sparse_feature_extractor.extract_sync(
            data_df=data_df,
            all_concepts=self.all_concepts,
            obs_concept_to_extract=mask,
            text_column=self.text_column,
            error_if_missing=False,
        )

    def append_observed_concepts(self, observed_row: np.ndarray) -> None:
        """Append a matrix row carrying x_t's frozen annotations (carry-forward).

        Called by the monitor after the e-detector decision, so x_t's scoring-time
        annotations are reused when it becomes history (not re-annotated on old columns).
        """
        n_rows, n_cols = self.sparse_concept_matrix.shape
        self.sparse_concept_matrix.resize((n_rows + 1, n_cols))
        if observed_row is not None:
            observed_row = np.asarray(observed_row).reshape(-1)
            assert observed_row.shape[0] == n_cols, (
                f"observed_row has {observed_row.shape[0]} concepts but matrix has {n_cols}"
            )
            self.sparse_concept_matrix[n_rows, :] = observed_row

    def _reindex_concepts(self, keep_idx: np.ndarray, removed_t: int) -> None:
        """Filter every concept-aligned structure to `keep_idx` positions, in lockstep.

        Covers the whole history: `sparse_concept_matrix` holds every past observation's
        annotations (rows = obs, cols = concepts), so dropping columns here reduces all
        history rows to the survivors — not just the current x_t.
        """
        keep_idx = np.asarray(keep_idx, dtype=int)
        # The single place concepts leave all_concepts, so the single place a
        # residency episode can close.
        if self.episode_recorder is not None:
            survivors = {int(j) for j in keep_idx}
            for j, concept in enumerate(self.all_concepts):
                if j not in survivors:
                    self.episode_recorder.close_episode(concept["id"], removed_t)
        self.all_concepts = [self.all_concepts[j] for j in keep_idx]
        self.sparse_concept_matrix.select_columns(keep_idx)

        # Mock keeps a parallel informative_flags list (position-aligned); reindex it too.
        state = getattr(self.concept_selector, "state", None)
        if state is not None and hasattr(state, "informative_flags"):
            state.informative_flags = [state.informative_flags[j] for j in keep_idx]
        # All concept-aligned structures must now agree on width.
        C = len(self.all_concepts)
        assert self.sparse_concept_matrix.shape[1] == C, (
            f"history matrix cols {self.sparse_concept_matrix.shape[1]} != concepts {C}"
        )

    # def build_lifecycle_rows(
    #     self, t: int, current_observed: np.ndarray, remove_set: set[int],
    # ) -> tuple[ConceptLifecycleRow, ...]:
    #     """Snapshot the active concept set as long-form rows, one per concept.

    #     Called by the monitor after x_t is annotated and after the step's classifier fit.
    #     `remove_set` are the local indices that were removed this step (subset selection).

    #     NOT pure: each row also increments its concept's episode active-step count
    #     (episode_recorder.note_active), so n_steps_active counts the rows built here.
    #     """
    #     concepts = self.all_concepts
    #     n_concepts = len(concepts)
    #     current_observed = np.asarray(current_observed).reshape(-1)
    #     assert current_observed.size == n_concepts

    #     coefficients = self.linear_coef()
    #     if coefficients is not None:
    #         coefficients = np.ravel(coefficients)
    #         assert coefficients.size == n_concepts

    #     history_observed = self.sparse_concept_matrix.get_filled_entries()
    #     calibration_observed = self.calib_concept_matrix.get_filled_entries()
    #     assert history_observed.shape[1] == n_concepts
    #     assert calibration_observed.shape[1] == n_concepts
    #     history_counts = history_observed.sum(axis=0)
    #     calibration_counts = calibration_observed.sum(axis=0)
    #     history_n = history_observed.shape[0]
    #     calibration_n = calibration_observed.shape[0]

    #     new_ids = {c["id"] for c in self.last_added_concepts}
    #     rows = []
    #     for j, concept in enumerate(concepts):
    #         annotation = current_observed[j]
    #         if self.episode_recorder is not None:
    #             self.episode_recorder.note_active(concept["id"])
    #         rows.append(ConceptLifecycleRow(
    #             t=t,
    #             scl_variant=self.scl_variant,
    #             removal_rule=self.removal_rule,
    #             fit_index=j,
    #             concept_id=concept["id"],
    #             proposed_t=concept["proposed_t"],
    #             new_this_step=concept["id"] in new_ids,
    #             classifier_coefficient=(
    #                 None if coefficients is None else float(coefficients[j])
    #             ),
    #             removed_this_step=j in remove_set,
    #             removal_candidate_id=self.last_removal_candidate_id.get(concept["id"]),
    #             current_observed=not np.isnan(annotation),
    #             current_annotation=(
    #                 None if np.isnan(annotation) else float(annotation)
    #             ),
    #             history_n=history_n,
    #             history_n_observed=int(history_counts[j]),
    #             history_observed_rate=(
    #                 float(history_counts[j] / history_n) if history_n else np.nan
    #             ),
    #             calibration_n=calibration_n,
    #             calibration_n_observed=int(calibration_counts[j]),
    #             calibration_observed_rate=(
    #                 float(calibration_counts[j] / calibration_n)
    #                 if calibration_n else np.nan
    #             ),
    #         ))
    #     return tuple(rows)


class SubsetScanSCL(SequentialConceptLearner):
    """Subset-scan concept selection: joint feature selection and changepoint detection.

    Uses SubsetScanChangepointLearner to jointly select the K-best concepts AND the
    changepoint by maximizing the changepoint statistic over all K-subsets.

    Flow each step:
    1. Propose new concepts
    2. Fully annotate all (history_obs, new_concept) and (calib_obs, new_concept) pairs
    3. Run SubsetScanChangepointLearner on annotated features -> (t_hat, best_subset)
    4. Filter concepts to the selected subset
    5. Fit classifier on selected concepts

    All observations are fully annotated - no imputation or budget-based sampling needed.

    Uses two feature extractors:
    - sparse_feature_extractor (batched): annotates multiple concepts per observation
      for history rows (efficient for bulk annotation)
    - bettor_feature_extractor (single-concept): annotates one concept per observation
      for calibration and x_t (consistent conditions for the e-process bet)
    """

    scl_variant: str = "subset_scan"

    def __init__(
        self,
        *,
        min_block: int,
        bettor_feature_extractor: SparseFeatureExtractor | None = None,
        concept_projector=None,
        subset_size: int | None = None,
        candidate_step: int = 1,
        search_all_sizes: bool = False,
        history_sample_size: int | None = None,
        **kwargs,
    ):
        """
        Args:
            min_block: CP learner's min_block for changepoint detection
            bettor_feature_extractor: extractor for calibration and x_t (single-concept
                mode for consistent betting conditions). If None, uses sparse_feature_extractor.
            concept_projector: For identity/sae modes, projects embeddings to concept space.
                If None, uses annotation matrix directly (real/mock/oracle modes).
            subset_size: Number of concepts K to select. Defaults to n_init_concepts.
            candidate_step: Grid step for changepoint candidate splits.
            search_all_sizes: if True, search subset sizes 1..K with penalty (stat - k).
            history_sample_size: if set, each update proposes, annotates, scans and fits on
                a fresh uniform sample of this many history rows (all rows while
                t + 1 <= N), which makes annotation cost linear in t. None = full history.
            **kwargs: shared SequentialConceptLearner params
        """
        kwargs["removal_rule"] = "none"
        super().__init__(**kwargs)

        assert min_block >= 2, "min_block must be >= 2"
        self.min_block = min_block
        self.concept_projector = concept_projector
        assert history_sample_size is None or concept_projector is None, (
            "history_sample_size only applies to annotation-based learners"
        )
        assert history_sample_size is None or history_sample_size >= 2 * min_block, (
            f"history_sample_size {history_sample_size} leaves no room for min_block {min_block}"
        )
        self.history_sample_size = history_sample_size
        # Own generator so the None path leaves the global stream (proposer, extractor) untouched.
        self._sample_rng = np.random.default_rng(self.seed)
        # Cumulative (row, concept) cells requested from the annotators; cache-independent.
        self.n_annotation_cells_history = 0
        self.n_annotation_cells_calib = 0
        self.n_annotation_cells_xt = 0

        self.bettor_feature_extractor = (
            bettor_feature_extractor if bettor_feature_extractor is not None
            else self.sparse_feature_extractor
        )

        from .changepoint import MeanShiftChangepointLearner, SubsetScanChangepointLearner
        k = subset_size if subset_size is not None else self.n_init_concepts
        base_learner = MeanShiftChangepointLearner(
            min_block=self.min_block, candidate_step=candidate_step
        )
        self.subset_scan_learner = SubsetScanChangepointLearner(
            base_learner, subset_size=k, search_all_sizes=search_all_sizes
        )
        self.last_selected_subset: np.ndarray | None = None
        self.last_scan_statistic: float = float("nan")

    def get_concept_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Get concept features from annotations or embeddings.

        For real/mock/oracle: returns fully-annotated matrix directly.
        For identity/sae: uses concept_projector to map embeddings to concepts.
        """
        if self.concept_projector is not None:
            # Identity/SAE: project embeddings to concept space
            return self.concept_projector.project(embeddings)
        # Real/mock/oracle: use annotation matrix
        return self.sparse_concept_matrix.to_dense()

    def update(
        self,
        structured_history: np.ndarray,
        unstructured_history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run subset-scan concept selection and changepoint detection."""
        curr_n_obs = structured_history.shape[0]
        data_df = self._prepare_step_frame(structured_history, unstructured_history_df)
        assert curr_n_obs >= self.n_burn_in, "monitor only calls update() post-burn-in"

        # Track matrix dimensions before resize
        n_prev_obs = self.sparse_concept_matrix.shape[0]
        n_prev_concepts = self.n_concepts

        # 0. TODO: REMOVE? Under subsampling, this step's rows; the retained concepts are completed on
        # them first so the proposer, scan and classifier all see one complete submatrix.
        # 0. For keyphrase learning, only use rows where the keyphrase is non-empty
        rows = np.where(data_df.llm_output != "")[0]
        # rows = self._sample_rows(curr_n_obs)
        if rows is not None:
            self.sparse_concept_matrix.update_from_sparse(
                self._get_annotations(data_df, n_prev_concepts=n_prev_concepts,
                                      n_prev_obs=curr_n_obs, rows=rows)
            )

        # 1. Propose new concepts
        new_concepts = self._propose_refine_concepts(structured_history, data_df, rows)
        self.last_added_concepts = new_concepts
        self.all_concepts.extend(new_concepts)

        # 2. Annotate new cells
        self.sparse_concept_matrix.resize((curr_n_obs, self.n_concepts))
        self.sparse_concept_matrix.update_from_sparse(
            self._get_annotations(data_df, n_prev_concepts=n_prev_concepts,
                                  n_prev_obs=n_prev_obs, rows=rows)
        )

        # Get concept features from annotations (or embeddings for identity/sae)
        concept_features = self.get_concept_features(structured_history)
        if rows is not None:
            concept_features = concept_features[rows]
            assert not np.isnan(concept_features).any(), "sampled rows must be fully annotated"

        # 3. Run subset scan to jointly select best K concepts AND t_hat
        # 3A. Estimate best subset and t_hat (split is in sample units; t_hat in time units)
        split, stat, best_subset = self.subset_scan_learner.estimate_with_subset(concept_features)
        t_hat = split if rows is None or split is None else int(rows[split])
        logging.info(f"ESTIMATED T-HAT {t_hat}")
        logging.info(concept_features[:split].mean(axis=0))
        logging.info(concept_features[split:].mean(axis=0))
        logging.info(f"ORACLE T-HAT")
        logging.info(concept_features[:100].mean(axis=0))
        logging.info(concept_features[100:].mean(axis=0))
        for c_idx, c in enumerate(self.all_concepts):
            logging.info(f"BEST {c_idx}, {c_idx in best_subset}, {c}")
        assert t_hat is not None, "subset scan must return valid t_hat post-burn-in"
        self.last_selected_subset = best_subset
        self.last_scan_statistic = stat

        # 3B. Apply subset selection - keep only the selected concepts
        C = self.n_concepts
        if best_subset is not None and len(best_subset) < C:
            remove_set = set(range(C)) - set(best_subset.tolist())
            self.last_removed_concepts = [self.all_concepts[i] for i in sorted(remove_set)]
            keep_idx = np.array(sorted(best_subset), dtype=int)
            self._reindex_concepts(keep_idx, removed_t=curr_n_obs)
        else:
            self.last_removed_concepts = []

        # 4. Fit pre- vs post- classifier on the selected concept features
        data_df["y"] = (np.arange(len(data_df)) >= t_hat).astype(int)
        selected_features = self.get_concept_features(structured_history)
        fit_rows = slice(None) if rows is None else rows
        self._fit_classifier_on(selected_features[fit_rows], data_df["y"].values[fit_rows])
        self.last_fit_debug["clf_n_train"] = selected_features[fit_rows].shape[0]

        # Store debug info
        self.last_fit_debug["n_annotation_cells_history"] = self.n_annotation_cells_history
        self.last_fit_debug["scan_statistic"] = stat
        self.last_fit_debug["selected_subset"] = (
            ";".join(map(str, best_subset)) if best_subset is not None else ""
        )

        return data_df

    def _sample_rows(self, curr_n_obs: int) -> np.ndarray | None:
        """This step's history rows under subsampling: a sorted uniform draw without
        replacement of `history_sample_size` rows out of 0..t, or None for full history."""
        n = self.history_sample_size
        if n is None or curr_n_obs <= n:
            return None
        return np.sort(self._sample_rng.choice(curr_n_obs, size=n, replace=False))

    def _get_annotations(
        self,
        data_df: pd.DataFrame,
        n_prev_concepts: int = 0,
        n_prev_obs: int = 0,
        use_bettor_extractor: bool = False,
        rows: np.ndarray | None = None,
    ) -> SparseExtractionMatrix:
        """Extract annotations for cells with row >= n_prev_obs or col >= n_prev_concepts.

        With `rows`, restrict to those rows only.
        Returns the sparse extraction matrix (caller merges into their matrix).
        """
        n_obs, C = len(data_df), len(self.all_concepts)
        dense = np.zeros((n_obs, C), dtype=np.int8)
        dense[n_prev_obs:, :] = 1
        dense[:, n_prev_concepts:] = 1
        if rows is not None:
            keep = np.zeros(n_obs, dtype=bool)
            keep[rows] = True
            dense[~keep, :] = 0

        if use_bettor_extractor:
            self.n_annotation_cells_calib += int(dense.sum())
        else:
            self.n_annotation_cells_history += int(dense.sum())
        extract_fn = self._extract_bettor_cells if use_bettor_extractor else self._extract_masked_cells
        return extract_fn(data_df, sp.lil_array(dense))

    def annotate_observations(self, obs_row: dict) -> np.ndarray:
        """Fully annotate all concepts on a fresh observation (x_t).

        Extracts keyphrases if llm_output is not already populated (mutates obs_row).
        Uses the bettor_feature_extractor (single-concept mode) for consistent conditions
        in the e-process bet.
        """
        # Extract keyphrases if not already present
        if "llm_output" not in obs_row or obs_row["llm_output"] is None:
            frame = pd.DataFrame([{self.text_column: obs_row[self.text_column]}])
            frame = self._ensure_keyphrases(frame)
            obs_row["llm_output"] = frame["llm_output"].iloc[0]

        C = len(self.all_concepts)
        frame = pd.DataFrame([{
            self.text_column: obs_row[self.text_column],
            "llm_output": obs_row["llm_output"],
            "y_true": obs_row.get("y_true"),
        }])
        dense = np.ones((1, C), dtype=np.int8)  # annotate ALL concepts
        self.n_annotation_cells_xt += C
        annot_new = self._extract_bettor_cells(frame, sp.lil_array(dense)).to_dense()
        logging.info(f"NEW OBSERVATION #concepts {C}: {annot_new}")
        self.append_observed_concepts(annot_new[0])
        return annot_new

    def _extract_bettor_cells(self, data_df: pd.DataFrame, mask) -> SparseExtractionMatrix:
        """Extract using the bettor extractor (for calibration and x_t)."""
        return self.bettor_feature_extractor.extract_sync(
            data_df=data_df,
            all_concepts=self.all_concepts,
            obs_concept_to_extract=mask,
            text_column=self.text_column,
            error_if_missing=False,
        )

    def predict_proba(self, embeddings: np.ndarray, observed: np.ndarray | None = None) -> float | None:
        """Predict class probabilities using concept features.

        For identity/sae: uses concept_projector to map embeddings.
        For real/mock/oracle: uses the provided annotations.
        """
        if self.concept_projector is not None:
            features = self.concept_projector.project(embeddings)
        elif observed is not None:
            features = observed
        else:
            features = self.sparse_concept_matrix.to_dense()[-embeddings.shape[0]:]
        if np.any(np.isnan(features)):
            return None
        return float(self.classifier.predict_proba(features)[0,1])

    def predict_proba_null(self) -> float:
        """p_null for the bet L = 1 + lam * (p_t - p_null), scored on calibration rows."""
        if self.concept_projector is not None:
            features = self.concept_projector.project(self.fixed_calibs)
        else:
            logging.info("NULL OBS")
            calib_matrix = SparseExtractionMatrix((self.n_calib, self.n_concepts))
            calib_matrix.update_from_sparse(
                self._get_annotations(self.calib_df, use_bettor_extractor=True)
            )
            features = calib_matrix.to_dense()
            num_nan = np.isnan(features).sum(axis=1)
            features = features[num_nan == 0,:]
        return float(np.mean(self.classifier.predict_proba(features)[:, 1]))