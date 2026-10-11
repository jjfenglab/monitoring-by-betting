from __future__ import annotations

import sys
import asyncio
import logging
import traceback
from typing import Dict, List
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Literal, Optional
from pathlib import Path

from llm_prior_monitoring.concept_selectors import MockConceptState

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from HACHI.src.llm_utils import run_prompts_batched

class SingleExtractResponse(BaseModel):
    reasoning: Optional[str] = Field(description="reasoning")
    answer: float = Field(
        description="binary answer, 1=yes, 0=no, probability if unsure"
    )
class ExtractResponseList(BaseModel):
    extractions: List[SingleExtractResponse] = Field(description="List of extractions")


class SparseExtractionMatrix:
    """Sparse matrix where the implicit background is NaN (unextracted).

    Mapping logic:
    - SciPy 0  -> True NaN (implicitly stored, takes no memory)
    - SciPy (Value)  -> True (Value - 1) (explicitly stored)
    """

    def __init__(self, shape):
        self._matrix = sp.lil_array(shape, dtype=np.float32)
        self.shape = shape

    def __setitem__(self, key, value):
        """Set value using true values (0, 1, np.nan)."""
        if np.isscalar(value):
            self._matrix[key] = value + 1
        else:
            self._matrix[key] = self._encode_value(value)

    def __getitem__(self, key):
        """Get value, returning the true value."""
        raw_val = self._matrix[key]

        if isinstance(raw_val, np.float):
            return self._decode_scalar(raw_val)

        return self._decode_array(raw_val.toarray())

    def _decode_scalar(self, val):
        if val == 0:
            return np.nan
        else:
            return val - 1

    def _decode_array(self, dense_arr):
        """Decode an entire dense numpy array back to original values."""
        float_arr = dense_arr.astype(float)
        nan_mask = float_arr == 0
        float_arr[float_arr > 0] -= 1.0
        float_arr[nan_mask] = np.nan
        return float_arr

    def _encode_value(self, val):
        """Encode an array of true values to internal representation."""
        return val + 1.0

    def resize(self, new_shape):
        """Dynamically change the size of the matrix."""
        self._matrix.resize(new_shape)
        self.shape = new_shape

    def select_columns(self, keep_idx):
        """Keep/reorder columns by `keep_idx` (column indices in the new order).

        A pure column subset/permutation on the raw 0/1/2 encoding, so unextracted
        (NaN) cells stay NaN. Used to drop removed-concept columns while keeping the
        matrix aligned with `all_concepts`.
        """
        keep_idx = np.asarray(keep_idx, dtype=int)
        assert keep_idx.ndim == 1 and (keep_idx < self.shape[1]).all(), (
            f"keep_idx out of range for {self.shape[1]} columns"
        )
        self._matrix = self._matrix.tocsc()[:, keep_idx].tolil()
        self.shape = self._matrix.shape

    def get_filled_entries(self):
        """Return boolean mask of entries that have been extracted."""
        return self._matrix.toarray() > 0

    def to_dense(self) -> np.ndarray:
        """Convert to dense array with true values (0, 1, NaN)."""
        return self._decode_array(self._matrix.toarray())

    def update_from_sparse(self, other: SparseExtractionMatrix):
        """Update this matrix with non-NaN entries from another matrix."""
        other_filled = other._matrix.toarray() > 0
        rows, cols = np.where(other_filled)
        for r, c in zip(rows, cols):
            self._matrix[r, c] = other._matrix[r, c]


class SparseFeatureExtractor:
    """Extract different sets of features for different observations."""

    def __init__(
        self,
        llm_dict: Dict,
        prompt_file: str,
        is_image: bool = False,
        batch_size: int=10,
        max_new_tokens: int = 8000,
        num_retries: int = 0,
        retry_delay_seconds: int = 60,
        retry_prompt_file: Optional[str] = None,
    ):
        self.llm_dict = llm_dict
        self.prompt_file = prompt_file
        self.is_image = is_image
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_retries = num_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_prompt_file = retry_prompt_file
        self._prompt_template = self._load_prompt_template()
        self._retry_prompt_template = self._load_retry_prompt_template()

    def _load_prompt_template(self) -> str:
        with open(self.prompt_file, "r") as f:
            return f.read()

    def _load_retry_prompt_template(self) -> Optional[str]:
        if self.retry_prompt_file and Path(self.retry_prompt_file).exists():
            with open(self.retry_prompt_file, "r") as f:
                return f.read()
        return None

    async def extract(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract: sp.sparray,
        text_column: str = "text",
        error_if_missing: bool = False,
    ) -> SparseExtractionMatrix:
        """
        Extract different features for different observations.

        Args:
            data_df: dataframe with text observations (image paths in image mode)
            all_concepts: list of concept dictionaries with 'concept' key
            obs_concept_to_extract: sparse matrix indicating which (obs, concept) to extract
            text_column: name of column containing text

        Returns:
            SparseExtractionMatrix with extracted values
        """
        n_obs, n_concepts = obs_concept_to_extract.shape
        result = SparseExtractionMatrix((n_obs, n_concepts))

        extraction_mask = obs_concept_to_extract.toarray() if sp.issparse(obs_concept_to_extract) else obs_concept_to_extract
        rows_to_extract = np.where(extraction_mask.any(axis=1))[0]

        if len(rows_to_extract) == 0:
            return result

        logging.info(f"Extracting features for {len(rows_to_extract)} observations")

        prompts = []
        retry_prompts = [] if self._retry_prompt_template else None
        prompt_metadata = []
        image_paths = [] if self.is_image else None

        for obs_idx in rows_to_extract:
            concept_indices = np.where(extraction_mask[obs_idx] > 0)[0]
            # Randomize the order in which annotations are done for concepts, to mitigate temporal
            # shifts in annotation of concepts
            concept_indices = np.random.permutation(concept_indices)
            text = data_df.iloc[obs_idx][text_column]
            concepts_for_obs = [all_concepts[c_idx] for c_idx in concept_indices]

            prompt = self._build_extraction_prompt(text, concepts_for_obs)
            prompts.append(prompt)
            if retry_prompts is not None:
                retry_prompt = self._build_extraction_prompt(
                    text, concepts_for_obs, self._retry_prompt_template
                )
                retry_prompts.append(retry_prompt)
            if self.is_image:
                image_paths.append(str(text))
            prompt_metadata.append({
                "obs_idx": obs_idx,
                "concept_indices": concept_indices,
                "concepts": concepts_for_obs,
            })

        logging.info(f"batch extract {len(prompts)}")
        logging.info(prompts[0])

        # Select LLM and then query
        llm = self.llm_dict.get("extraction")
        llm_outputs = await run_prompts_batched(
            llm,
            prompts,
            image_paths=image_paths,
            retry_prompts=retry_prompts,
            response_format=ExtractResponseList,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            desc="Batch concept extraction",
            num_retries=self.num_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            error_if_missing=error_if_missing
        )

        for metadata, output in zip(prompt_metadata, llm_outputs):
            obs_idx = metadata["obs_idx"]
            concept_indices = metadata["concept_indices"]
            if output is None:
                # failed to extract -- continue
                continue

            for c_local_idx, extraction in enumerate(output.extractions):
                if c_local_idx < len(concept_indices):
                    c_global_idx = concept_indices[c_local_idx]
                    result[obs_idx, c_global_idx] = extraction.answer

        return result

    def _build_extraction_prompt(
        self, text: str, concepts: List[Dict], template: Optional[str] = None
    ) -> str:
        """Build prompt for extracting multiple concepts; in image mode the observation rides as a content part, not in the prompt."""
        template = template or self._prompt_template
        prompt_questions = "\n".join(
            f"{i+1}. {c['concept']}" for i, c in enumerate(concepts)
        )
        if self.is_image:
            return template.format(prompt_questions=prompt_questions)
        return template.format(
            text=text,
            prompt_questions=prompt_questions,
        )

    def extract_sync(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract: sp.sparray,
        text_column: str = "text",
        error_if_missing: bool = False,
    ) -> SparseExtractionMatrix:
        """Synchronous wrapper for extract()."""
        return asyncio.run(
            self.extract(data_df, all_concepts, obs_concept_to_extract, text_column, error_if_missing=error_if_missing)
        )


class SingleConceptSparseFeatureExtractor(SparseFeatureExtractor):
    """Extract one concept per observation per LLM call.

    Unlike SparseFeatureExtractor which batches multiple concepts into one prompt
    per observation, this class makes a separate LLM call for each (obs, concept)
    pair. This ensures all annotations are made under identical conditions,
    avoiding potential artifacts from different batch sizes across observations.
    """

    async def extract(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract: sp.sparray,
        text_column: str = "text",
        error_if_missing: bool = False,
    ) -> SparseExtractionMatrix:
        """Extract one concept per observation per LLM call."""
        n_obs, n_concepts = obs_concept_to_extract.shape
        result = SparseExtractionMatrix((n_obs, n_concepts))

        extraction_mask = (
            obs_concept_to_extract.toarray()
            if sp.issparse(obs_concept_to_extract)
            else obs_concept_to_extract
        )
        rows, cols = np.where(extraction_mask > 0)

        if len(rows) == 0:
            return result

        prompts = []
        retry_prompts = [] if self._retry_prompt_template else None
        prompt_metadata = []
        image_paths = [] if self.is_image else None

        for obs_idx, concept_idx in zip(rows, cols):
            text = data_df.iloc[obs_idx][text_column]
            concept = all_concepts[concept_idx]
            prompt = self._build_extraction_prompt(text, [concept])
            prompts.append(prompt)
            if retry_prompts is not None:
                retry_prompt = self._build_extraction_prompt(
                    text, [concept], self._retry_prompt_template
                )
                retry_prompts.append(retry_prompt)
            if self.is_image:
                image_paths.append(str(text))
            prompt_metadata.append({"obs_idx": obs_idx, "concept_idx": concept_idx})

        llm = self.llm_dict.get("extraction")
        llm_outputs = await run_prompts_batched(
            llm,
            prompts,
            image_paths=image_paths,
            retry_prompts=retry_prompts,
            response_format=SingleExtractResponse, #ExtractResponseList,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            desc="Single-concept extraction",
            num_retries=self.num_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            error_if_missing=error_if_missing,
        )

        for metadata, output in zip(prompt_metadata, llm_outputs):
            obs_idx = metadata["obs_idx"]
            concept_idx = metadata["concept_idx"]
            if output is not None:
                result[obs_idx, concept_idx] = output.answer

        return result


class MockSparseFeatureExtractor:
    """Feature extractor that draws Bernoulli concept values from a shared MockConceptState.

    Replaces LLM calls in SparseFeatureExtractor. The oracle pre/post-change label of each
    observation is read from the frame's `y_true` column (`data_df["y_true"].iloc[obs_idx]`),
    so a standalone x_t / reference frame just carries its own `y_true` — no global index.

    Concept roles come from the concept dicts (strength key set by MockConceptSelector):
      strong ("s"): the observation's latent L(text, id, y), drawn once as
          P(L=1 | y=1) = 1 - noise_rate, P(L=1 | y=0) = noise_rate and cached in
          state.latent_values — a recurring text redraws identically, and the latent
          outlives the concept's removal (it is regenerated on demand by partner_id).
      weak ("w"):  its partner's latent flipped with prob state.weak_flip_rate, so it
          is weakly correlated with the strong twin and weakly informative through it.
      noise:       Bernoulli(0.5) regardless of y_true.
    """

    def __init__(self, state: MockConceptState):
        self.state = state

    def _latent(self, text: str, strong_id: int, y_val: int) -> int:
        key = (text, strong_id, y_val)
        if key not in self.state.latent_values:
            p = 1 - self.state.noise_rate if y_val == 1 else self.state.noise_rate
            self.state.latent_values[key] = int(np.random.random() < p)
        return self.state.latent_values[key]

    def extract_sync(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[Dict],
        obs_concept_to_extract: sp.sparray,
        text_column: str = "text",
        error_if_missing: bool = True,
    ) -> SparseExtractionMatrix:
        n_obs, n_concepts = obs_concept_to_extract.shape
        result = SparseExtractionMatrix((n_obs, n_concepts))
        mask = obs_concept_to_extract.toarray() if sp.issparse(obs_concept_to_extract) else obs_concept_to_extract
        rows, cols = np.where(mask > 0)
        y_true = data_df["y_true"].to_numpy()
        texts = data_df[text_column].to_numpy()
        for obs_idx, concept_idx in zip(rows, cols):
            y_val = int(y_true[obs_idx])
            concept = all_concepts[concept_idx]
            strength = concept.get("strength")
            if strength == "s":
                value = self._latent(texts[obs_idx], concept["id"], y_val)
            elif strength == "w":
                latent = self._latent(texts[obs_idx], concept["partner_id"], y_val)
                value = latent ^ int(np.random.random() < self.state.weak_flip_rate)
            else:
                value = int(np.random.random() < 0.5)
            result[obs_idx, concept_idx] = value
        return result
