"""Typed records and CSV serialization for concept lifecycle diagnostics.

debug.csv is keyed on t and collapses the concept dimension into counts and
";"-joined strings. These three artifacts re-key on the concept so per-concept
questions become answerable:

  concept_episodes.csv   one row per concept_id  -- the name dictionary. A
                         concept_id is a residency EPISODE, not a concept: a
                         pool column removed and later re-proposed gets a fresh
                         id, so one `concept` string maps to many ids. Names
                         live here and nowhere else, because the removals table
                         would otherwise repeat them (n_candidates x group_size)
                         times per step.
  concept_lifecycle.csv  one row per (t, concept_id) -- the active set the
                         classifier was fit on for this step's p_t. Written at
                         one uniform point in both scl_variants.
  concept_removals.csv   one row per (t, candidate_id) -- every scored removal
                         candidate, written at the decision point.

Each Row dataclass derives its own CSV header from fields(), so the schema and
the header cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd


@dataclass
class ConceptEpisodeRow:
    """Record one residency episode: a concept_id from proposal to removal."""

    concept_id: int
    concept: str
    proposed_t: int
    removed_t: int | None
    n_steps_active: int
    removal_rule: str
    scl_variant: str
    # 0-based count of prior episodes with the EXACT same `concept` string.
    # Literal churn for the finite oracle/mock pools; ~0 for real, whose LLM
    # re-proposes rejected concepts in paraphrase (cluster post-hoc instead).
    episode_index: int


@dataclass(frozen=True)
class ConceptLifecycleRow:
    """Record one active concept at one monitoring step.

    Every field is measured at monitor step 5c -- after x_t is annotated and
    after the step's classifier fit, but BEFORE detect_then_remove's removal --
    so the rows are always "the concept set this step's p_t was scored on" in
    both scl_variants.

    `concept` is not here; join concept_episodes.csv on concept_id.
    """

    t: int
    scl_variant: str
    removal_rule: str
    fit_index: int
    concept_id: int
    proposed_t: int
    new_this_step: bool
    classifier_coefficient: float | None
    # detect_then_remove decides after this snapshot, so its doomed concepts are
    # still here and flagged. remove_then_detect removed before the fit that
    # produced p_t, so its dropped concepts have no row at all and this is always
    # False -- concept_episodes.removed_t is the variant-agnostic death record.
    removed_this_step: bool
    # The concept_removals.csv candidate that decided this removal, so greedy_n's
    # removed pair shares one id. None for survivors (under greedy_n a survivor
    # sits in many scored groups at once) and whenever that file is not written.
    removal_candidate_id: int | None
    current_observed: bool
    current_annotation: float | None
    history_n: int
    history_n_observed: int
    history_observed_rate: float
    calibration_n: int
    calibration_n_observed: int
    calibration_observed_rate: float


@dataclass(frozen=True)
class RemovalCandidateRow:
    """Record one scored removal candidate: a concept (importance/greedy_1,
    group_size 1) or a group of them (greedy_n)."""

    t: int
    scl_variant: str
    removal_rule: str
    candidate_id: int
    group_size: int
    member_ids: str          # ";"-joined concept_ids; names live in the episodes file
    score: float | None      # higher = stronger evidence to retain; NaN for "random"
    # The feature matrix the score was computed on. detect_then_remove scores the
    # full imputed history, remove_then_detect the restricted annotated rows, so
    # scores are NOT comparable across variants.
    score_basis: str
    selected: bool           # this candidate's members were removed


class CsvRecorder:
    """Append typed rows to a declared CSV artifact.

    The header is written at construction so a declared SCons target always
    exists, even for a disabled recorder or a run that records nothing.
    Appending per step keeps memory flat and leaves a readable partial file if
    a stream dies mid-run.
    """

    def __init__(self, path: str | Path, row_type: type, enabled: bool = True):
        self.path = Path(path)
        self.columns = tuple(field.name for field in fields(row_type))
        self.enabled = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)

    def record(self, rows) -> None:
        """Append one step's rows."""
        if not self.enabled or not rows:
            return
        frame = pd.DataFrame([asdict(row) for row in rows], columns=self.columns)
        frame.to_csv(self.path, mode="a", header=False, index=False)


class ConceptEpisodeRecorder:
    """Track residency episodes in memory and write the name dictionary at end of run.

    Buffered rather than streamed because an episode is only complete once it
    closes, and there are only n_new_concepts_per_update of them per step.
    """

    def __init__(self, path: str | Path, scl_variant: str, enabled: bool = True):
        self.path = Path(path)
        self.scl_variant = scl_variant
        self.enabled = enabled
        self.episodes: dict[int, ConceptEpisodeRow] = {}
        self._n_prior_episodes: dict[str, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[field.name for field in fields(ConceptEpisodeRow)]
        ).to_csv(self.path, index=False)

    def open_episode(self, concept_id: int, concept: str, proposed_t: int,
                     removal_rule: str) -> None:
        """Start an episode; `concept` repeating a past one bumps episode_index."""
        assert concept_id not in self.episodes, f"concept_id {concept_id} reused"
        episode_index = self._n_prior_episodes.get(concept, 0)
        self._n_prior_episodes[concept] = episode_index + 1
        self.episodes[concept_id] = ConceptEpisodeRow(
            concept_id=concept_id,
            concept=concept,
            proposed_t=proposed_t,
            removed_t=None,
            n_steps_active=0,
            removal_rule=removal_rule,
            scl_variant=self.scl_variant,
            episode_index=episode_index,
        )

    def note_active(self, concept_id: int) -> None:
        """Count one concept_lifecycle.csv row against this episode."""
        self.episodes[concept_id].n_steps_active += 1

    def close_episode(self, concept_id: int, removed_t: int) -> None:
        """End an episode at its removal step."""
        self.episodes[concept_id].removed_t = removed_t

    def write(self) -> None:
        """Write every episode, closed or still active, ordered by concept_id."""
        if not self.enabled:
            return
        rows = [self.episodes[cid] for cid in sorted(self.episodes)]
        pd.DataFrame(
            [asdict(row) for row in rows],
            columns=[field.name for field in fields(ConceptEpisodeRow)],
        ).to_csv(self.path, index=False)
