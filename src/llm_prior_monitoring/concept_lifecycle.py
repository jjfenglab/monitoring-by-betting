"""Typed record and CSV serialization for the concept_episodes.csv diagnostic.

debug.csv is keyed on t and collapses the concept dimension into counts and
";"-joined strings. concept_episodes.csv re-keys on the concept: one row per
concept_id. A concept_id is a residency EPISODE, not a concept: a pool column
pruned by the subset scan and later re-proposed gets a fresh id, so one `concept`
string maps to many ids.

The CSV header is derived from the row dataclass's fields(), so the schema and
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
    scl_variant: str
    # 0-based count of prior episodes with the EXACT same `concept` string.
    # Literal churn for the finite oracle/mock pools; ~0 for real, whose LLM
    # re-proposes rejected concepts in paraphrase (cluster post-hoc instead).
    episode_index: int


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

    def open_episode(self, concept_id: int, concept: str, proposed_t: int) -> None:
        """Start an episode; `concept` repeating a past one bumps episode_index."""
        assert concept_id not in self.episodes, f"concept_id {concept_id} reused"
        episode_index = self._n_prior_episodes.get(concept, 0)
        self._n_prior_episodes[concept] = episode_index + 1
        self.episodes[concept_id] = ConceptEpisodeRow(
            concept_id=concept_id,
            concept=concept,
            proposed_t=proposed_t,
            removed_t=None,
            scl_variant=self.scl_variant,
            episode_index=episode_index,
        )

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
