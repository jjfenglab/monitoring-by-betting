"""AdaptiveMonitoring orchestrator: composes the four learnable parts.

One instance per stream. Each step drives:
  SequentialConceptLearner -> EDetector

AdaptiveMonitoring takes a pre-built SequentialConceptLearner.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .concept_lifecycle import CsvRecorder
from .e_detector import EDetector
from .sequential_concept_learner import SequentialConceptLearner

import litellm
litellm.drop_params = True

log = logging.getLogger(__name__)


def _nan_frac(arr: np.ndarray) -> float:
    """Fraction of NaN (imputed-not-annotated) entries; NaN for an empty array."""
    return float(np.mean(np.isnan(arr))) if arr.size else float("nan")


class AdaptiveMonitoring:
    """Per-step monitoring loop composing the four learnable parts.

    Owns the rolling `structured_history` (embeddings) and `unstructured_history_df` (text + labels),
    both fed to SCL each step. The e_detector owns the running M-statistic
    and per-step trajectory.

    Composition order each step (PAST structured_history -> t_hat -> scl.update ->
    p_t/p_null -> e_detector -> append structured_obs_t to structured_history) enforces predictable
    scoring: every fit at step t sees only data through t-1.
    """

    def __init__(
        self,
        sequential_concept_learner: SequentialConceptLearner,
        e_detector: EDetector,
        cp_feature_source: str = "embedding",
        n_burn_in: int = 1,
        debug_csv: Path | None = None,
        matrix_snapshot_dir: Path | None = None,
        concept_lifecycle_recorder: CsvRecorder | None = None,
    ):
        assert cp_feature_source in ("embedding", "concept"), (
            f"cp_feature_source must be 'embedding' or 'concept', got {cp_feature_source!r}"
        )
        assert e_detector.n_burn_in == n_burn_in, (
            f"LLMGuidedMonitoring.n_burn_in ({n_burn_in}) must match "
            f"e_detector.n_burn_in ({e_detector.n_burn_in})"
        )
        self.scl = sequential_concept_learner
        assert self.scl.scl_variant == "subset_scan"

        self.e_detector = e_detector
        self.n_burn_in = n_burn_in
        self.cp_feature_source = cp_feature_source
        self.structured_history: np.ndarray | None = None
        self.unstructured_history_df: pd.DataFrame | None = None
        self.t_hat_history: list[int | None] = []
        self.debug_csv = debug_csv
        self.matrix_snapshot_dir = matrix_snapshot_dir
        self.snapshot_files: list[str] = []
        # concept_lifecycle.csv writer; the SCL owns the episode/removal writers
        # because it is where concepts are proposed and removed.
        self.concept_lifecycle_recorder = concept_lifecycle_recorder
        self.debug_records: list[dict] = []
        self._alarmed = False

    @property
    def t(self):
        return len(self.unstructured_history_df) if self.unstructured_history_df is not None else 0

    def update(
        self,
        structured_unstructured_obs_t: np.ndarray,
        unstructured_unstructured_obs_t: dict,
    ) -> None:
        """Process one observation"""
        active = self.structured_history is not None and len(self.structured_history) >= self.n_burn_in
        if active:
            self._step_active(structured_unstructured_obs_t, unstructured_unstructured_obs_t)
        else:
            self._step_burn_in(structured_unstructured_obs_t, unstructured_unstructured_obs_t)

    def _step_burn_in(self, structured_obs_t: np.ndarray, unstructured_obs_t: dict) -> None:
        """Warm-up step"""
        n_before = len(self.scl.all_concepts)
        self.e_detector.update()
        self._record_debug(
            None, float("nan"), None, n_before, len(self.scl.all_concepts),
            [], float("nan"), true_group=unstructured_obs_t.get("true_group"),
            source_idx=unstructured_obs_t.get("source_idx")
        )
        self._append(structured_obs_t, unstructured_obs_t)
        self.t_hat_history.append(None)

    def _step_active(
        self,
        structured_obs_t: np.ndarray,
        unstructured_obs_t: dict,
    ) -> None:
        """
        Post-burn-in step: structured_history and t_hat are guaranteed present (asserted).
        Estimate t_hat from PAST structured_history, label unstructured_history_df, refit SCL, score structured_obs_t and
        the null, and step the e_detector.
        """
        assert self.structured_history is not None

        # 1. Propose concepts, select concepts. Fit score function.
        n_before = len(self.scl.all_concepts)
        self.unstructured_history_df = self.scl.update(self.structured_history, self.unstructured_history_df)
        n_after = len(self.scl.all_concepts)
        
        # Read t_hat back from the y labels the SCL wrote
        y = self.unstructured_history_df["y"].values
        t_hat = int(np.argmax(y == 1)) if (y == 1).any() else len(y) // 2
        assert n_after > 0, "post-burn-in SCL must have fitted concepts"

        # Snapshot the exact feature set the classifier was fit on for this step's p_t:
        # pre-removal under detect_then_remove (removal happens in 5c below), post-removal
        # under remove_then_detect (removal already ran inside scl.update).
        active_concepts = [c["concept"] for c in self.scl.all_concepts]
        
        if self.matrix_snapshot_dir is not None:
            self._save_matrix_snapshots()

        # 2. Make the bet! Score the new observation structured_obs_t
        observed_concepts_xt = self.scl.annotate_observations(unstructured_obs_t)
        logging.info(f"ANNOT unstructured_obs_t {unstructured_obs_t}")
        logging.info(f"observed_concepts_xt {observed_concepts_xt}")
        
        p_t = self.scl.predict_proba(structured_obs_t[None, :], observed=observed_concepts_xt)
        if p_t is None:
            # Extraction failed. Make no bet.
            p_t = 0.5
            p_null = 0.5
        else:    
            p_null = self.scl.predict_proba_null()
        
        # 3. Get the return on the bet. Compute p_null and update e-detector        
        self.e_detector.update(p_t, p_null=p_null)

        # 4. Logging.
        # if (self.concept_lifecycle_recorder is not None
        #         and self.concept_lifecycle_recorder.enabled):
        #     self.concept_lifecycle_recorder.record(
        #         self.scl.build_lifecycle_rows(
        #             self.t, observed_concepts_xt, set())
        #     )
        
        self._record_debug(
            t_hat,
            p_t,
            p_null,
            n_before,
            n_after,
            active_concepts,
            _nan_frac(observed_concepts_xt),
            true_group=unstructured_obs_t.get("true_group"),
            source_idx=unstructured_obs_t.get("source_idx")
        )

        # 6. Append AFTER the e_detector update.
        self._append(structured_obs_t, unstructured_obs_t)
        self.t_hat_history.append(t_hat)
        
    def _append(
        self,
        structured_unstructured_obs_t: np.ndarray,
        unstructured_unstructured_obs_t: dict,
    ) -> None:
        """
        Append structured_unstructured_obs_t to structured_history and unstructured_obs_t to unstructured_history_df.
        """
        x_row = structured_unstructured_obs_t[None, :]
        self.structured_history = x_row.copy() if self.structured_history is None else np.vstack([self.structured_history, x_row])
        obs_row = pd.DataFrame([unstructured_unstructured_obs_t])
        self.unstructured_history_df = (
            obs_row.copy() if self.unstructured_history_df is None
            else pd.concat([self.unstructured_history_df, obs_row], ignore_index=True)
        )

    def _save_matrix_snapshots(self) -> None:
        """Write per-step CSVs of the annotation matrix (0/1/empty) and the concept
        features -- rows/columns are what this step's classifier was fit on.
        Row i = history sample i; first column = its text.
        """
        concepts = [c["concept"] for c in self.scl.all_concepts]
        texts = self.unstructured_history_df[self.scl.text_column].reset_index(drop=True)
        annotated = pd.DataFrame(self.scl.sparse_concept_matrix.to_dense(), columns=concepts)
        features = pd.DataFrame(self.scl.get_concept_features(self.structured_history), columns=concepts)
        for name, df in (("annotated", annotated), ("features", features)):
            df.insert(0, "text", texts)
            fname = f"{name}_t{self.t:04d}.csv"
            df.to_csv(self.matrix_snapshot_dir / fname, index=False)
            self.snapshot_files.append(fname)

    def _record_debug(
        self,
        t_hat: int | None,
        p_t: float,
        p_null: float | None,
        n_before: int,
        n_after: int,
        active_concepts: list[str],
        xt_missing_rate: float,
        true_group: int | None = None,
        source_idx: int | None = None,
    ) -> None:
        """Append one per-step debug record and emit key-event log lines.

        Called after the e_detector step so L_t / M_statistic are current.
        """
        t_hat = self.t_hat_history[-1] if self.t_hat_history else float("nan")
        # Added/removed come from the SCL's explicit per-step lists: with fixed-size
        # removal the count no longer only grows, so all_concepts[n_before:] is wrong.
        new_concepts = [c["concept"] for c in self.scl.last_added_concepts]
        removed_concepts = [c["concept"] for c in self.scl.last_removed_concepts]
        # Mock-only: is_informative is stamped at proposal by MockConceptSelector;
        # other arms' dicts lack the key, leaving these empty -> NaN fractions below.
        # Pre-removal active set = survivors + this step's removals (same dict objects).
        pre_removal = self.scl.all_concepts + self.scl.last_removed_concepts
        active_flags = [c["is_informative"] for c in pre_removal
                        if "is_informative" in c]
        n_noninf_active = sum(not f for f in active_flags)
        n_noninf_removed = sum(not c["is_informative"]
                               for c in self.scl.last_removed_concepts
                               if "is_informative" in c)
        n_weak_active = sum(c.get("strength") == "w" for c in pre_removal)
        n_weak_removed = sum(c.get("strength") == "w"
                             for c in self.scl.last_removed_concepts)
        coef = self.scl.linear_coef()
        n_nonzero = float("nan") if coef is None else int(np.count_nonzero(coef))
        converged = float("nan") if coef is None else bool(self.scl.last_fit_converged)
        L_t = self.e_detector.history_increment[-1]
        M = self.e_detector.MStatistic

        self.debug_records.append({
            "t": self.t,
            "t_hat": t_hat,
            "p_t": p_t,
            "p_null": p_null if p_null is not None else float("nan"),
            "L_t": L_t,
            "L_t_sign": float(np.sign(L_t - 1.0)),  # bet direction: sign(p_t - p_null)
            "true_group": true_group if true_group is not None else float("nan"),
            # Provenance: which CSV row this obs is, and whether it recurs (draws
            # are with replacement, so the same note can be seen more than once).
            "source_idx": source_idx if source_idx is not None else float("nan"),
            "M_statistic": M,
            "n_new_concepts": len(new_concepts),
            "new_concepts": ";".join(new_concepts),
            # active_concepts is the pre-removal set the classifier was fit on for
            # this step's p_t (snapshotted in _step_active), so n_concepts == its size.
            "n_concepts": n_after,
            "active_concepts": ";".join(active_concepts),
            "n_removed_concepts": len(removed_concepts),
            "removed_concepts": ";".join(removed_concepts),
            # Noninformative share of the pre-removal active set, and the fraction of
            # those noninformative concepts this step's removal caught (0 on
            # non-removal steps; NaN when the set has no flags / no noninformative).
            "frac_noninformative_active": (
                n_noninf_active / len(active_flags) if active_flags else float("nan")
            ),
            "frac_noninformative_removed": (
                n_noninf_removed / n_noninf_active if n_noninf_active else float("nan")
            ),
            # Same pair for the weak informative concepts (strength == "w").
            "frac_weak_active": (
                n_weak_active / len(active_flags) if active_flags else float("nan")
            ),
            "frac_weak_removed": (
                n_weak_removed / n_weak_active if n_weak_active else float("nan")
            ),
            "removal_variant": (
                self.scl.removal_rule if self.scl.removal_rule != "greedy"
                else ("greedy_1" if self.scl.greedy_group_size == 1 else "greedy_n")
            ),
            "scl_variant": self.scl.scl_variant,
            # remove_then_detect only; NaN for detect_then_remove and burn-in steps.
            "n_sample_rows_annotated": self.scl.last_fit_debug.get("n_sample_rows_annotated", float("nan")),
            # Anchors added beyond the t_hat window; drops when t_hat sits near an end.
            "n_anchor_rows_annotated": self.scl.last_fit_debug.get(
                "n_anchor_rows_annotated", float("nan")),
            "n_restricted_fit_rows": self.scl.last_fit_debug.get("n_restricted_fit_rows", float("nan")),
            "n_topup_cells": self.scl.last_fit_debug.get("n_topup_cells", float("nan")),
            "n_calib_cells": self.scl.last_fit_debug.get("n_calib_cells", float("nan")),
            # cumulative distinct history cells requested so far (cost-vs-t curve)
            "n_annotation_cells_history": self.scl.last_fit_debug.get(
                "n_annotation_cells_history", float("nan")),
            "restricted_fit_converged": self.scl.last_fit_debug.get("restricted_fit_converged", float("nan")),
            "classifier_n_nonzero": n_nonzero,
            "classifier_converged": converged,
            # What p_null actually scored on (remove_then_detect only; NaN elsewhere).
            # calib_reveal_missing_rate should equal xt_missing_rate wherever
            # n_calib_rows_short == 0 -- that equality is the bet's validity condition.
            "n_calib_reveal_target": self.scl.last_fit_debug.get(
                "n_calib_reveal_target", float("nan")),
            "n_calib_reveal_mean": self.scl.last_fit_debug.get(
                "n_calib_reveal_mean", float("nan")),
            "n_calib_rows_short": self.scl.last_fit_debug.get(
                "n_calib_rows_short", float("nan")),
            "calib_reveal_missing_rate": self.scl.last_fit_debug.get(
                "calib_reveal_missing_rate", float("nan")),
            # Online-SAE warm-start diagnostics on refit steps; NaN otherwise.
            "sae_refit_loss": self.scl.last_fit_debug.get("sae_refit_loss", float("nan")),
            "sae_dead_ratio": self.scl.last_fit_debug.get("sae_dead_ratio", float("nan")),
            "sae_n_train": self.scl.last_fit_debug.get("sae_n_train", float("nan")),
            # SubsetScanSCL only: the scan statistic and selected concept indices.
            "scan_statistic": self.scl.last_fit_debug.get("scan_statistic", float("nan")),
            "selected_subset": self.scl.last_fit_debug.get("selected_subset", ""),
        })

        if new_concepts:
            log.info("proposed %d concepts @ t=%d: %s", len(new_concepts), self.t, new_concepts)
        if removed_concepts:
            log.info("removed %d concepts @ t=%d: %s", len(removed_concepts), self.t, removed_concepts)
        if coef is not None and not self.scl.last_fit_converged:
            log.warning("classifier did not converge @ t=%d", self.t)
        if self.e_detector.is_rejected() and not self._alarmed:
            self._alarmed = True
            log.info("ALARM @ t=%d M=%.3g t_hat=%s", self.t, M, t_hat)

        if self.debug_csv:
            pd.DataFrame(self.debug_records).to_csv(self.debug_csv, index=False)