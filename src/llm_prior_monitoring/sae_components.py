"""SAE-backed SCL subcomponents: SAE latents are the concepts (HypotheSAEs).

A TopK sparse autoencoder pretrained offline on the disjoint SAE-side pool
(src/train_sae.py) maps each observation's embedding to M latent activations.
Concepts are all M neurons (generate_initial_concepts takes every neuron); their
activations are deterministic functions of the embedding — no LLM calls anywhere.

Two variants share this module, selected by SAEConfig.refit_every:
  frozen (None):            the offline checkpoint is used read-only for the whole stream.
  online (refit_every set): the checkpoint is warm-started at every SCL update
                            (continued training, persistent Adam) on the side pool plus
                            the stream history through t-1; SAEState accepts only None,
                            0 or 1. Neuron *indices* persist across refits; their
                            meanings drift.

Assembled by create_lgm_pipeline.make_sequential_concept_learner for
concept_learner_type == "sae", reusing MockKeyphrase, IdentityConceptSelector,
and IdentitySparseFeatureExtractor for the passthrough surfaces. SAEConceptProjector
maps embeddings to SAE activations.

Known limitation: embeddings are looked up by text (the generator only receives
data_df, never the host's `history` embeddings), so pool texts must map to a
unique embedding — SAEState asserts this at construction.
"""

from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from hypothesaes.sae import load_model
from hypothesaes.select_neurons import select_neurons
from torch.utils.data import DataLoader, TensorDataset


class SAEState:
    """Shared SAE model, pool text-to-embedding lookup, neuron registry, and warm-start refit."""

    def __init__(
        self,
        checkpoint_path: str,
        embeddings_path: str,
        mapping_path: str,
        *,
        side_embeddings_path: str | None = None,
        refit_every: int | None = None,
        refit_epochs: int = 5,
        refit_lr: float = 5e-4,
        refit_batch_size: int = 64,
    ):
        self.checkpoint_path = checkpoint_path
        self.sae = load_model(checkpoint_path, device="cpu")
        emb = np.load(embeddings_path)["emb"]
        mapping = pd.read_csv(mapping_path)
        self.text_to_emb: Dict[str, np.ndarray] = {}
        for text, row_index in zip(mapping["text"], mapping["row_index"]):
            e = emb[int(row_index)]
            prev = self.text_to_emb.get(text)
            # p0/p1 are independent draws over one population and may share texts;
            # batched encoding makes re-embedded duplicates differ at float32 level.
            assert prev is None or np.allclose(prev, e, atol=1e-5), (
                f"duplicate pool text with differing embeddings: {text!r}"
            )
            self.text_to_emb[text] = e
        self.neuron_idxs: List[int] = []  # neuron index per concept, in concept order
        self.refit_every = refit_every
        assert refit_every in [0,1,None]
        self.refit_epochs = refit_epochs
        self.refit_batch_size = refit_batch_size
        if refit_every is not None:
            assert side_embeddings_path is not None, "online SAE needs the side-pool embeddings"
            self.side_emb = np.load(side_embeddings_path)["emb"].astype(np.float32)
            assert self.side_emb.shape[1] == self.sae.input_dim
            # One optimizer for the whole stream: Adam moments carry across refits, so
            # each refit continues the previous one rather than restarting.
            self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=refit_lr)

    def activations(self, embeddings: np.ndarray) -> np.ndarray:
        """Full (n, M) SAE activation matrix for a batch of embeddings.

        hypothesaes' forward bumps its dead-neuron counter on every call, inference
        included, so the several scoring batches per step (history, x_t, the calib
        pool) would reset it for every neuron that fires and sae_dead_ratio would read
        a structural 0; snapshot/restore so it counts training batches only. The
        counter is a diagnostic here and nothing more: these checkpoints are trained
        with aux_k=None, so hypothesaes' dead-neuron revival term is absent and a
        neuron that stops winning the top-K gets no gradient to bring it back.
        """
        counter = self.sae.steps_since_activation.clone()
        acts = self.sae.get_activations(embeddings, show_progress=False)
        self.sae.steps_since_activation.copy_(counter)
        return acts

    def lookup(self, texts) -> np.ndarray:
        """Pool embeddings for the given texts; KeyError = text not in the pool."""
        return np.stack([self.text_to_emb[t] for t in texts])

    def refit(self, history_emb: np.ndarray) -> dict:
        """Warm-start the SAE on side pool ∪ history for refit_epochs; returns diagnostics.

        Reproduces hypothesaes' SparseAutoencoder.fit minus its data re-initialisation
        (fit() calls initialize_weights_ every time, so it cannot continue training).
        sae_refit_loss is the mean batch loss over the whole refit (all epochs).
        """
        train = np.concatenate([self.side_emb, history_emb.astype(np.float32)])
        assert len(train) >= self.refit_batch_size
        # drop_last: hypothesaes normalises each batch's MSE by that batch's own
        # variance, so a 1-row remainder batch divides by zero and poisons the weights.
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train)), batch_size=self.refit_batch_size,
            shuffle=True, drop_last=True,
        )
        self.sae.train()
        losses = []
        for _ in range(self.refit_epochs):
            for (x,) in loader:
                recon, info = self.sae(x)
                loss = self.sae.compute_loss(x, recon, info, aux_coef=1 / 32, multi_coef=0.0)
                # Per batch, before the step: one non-finite batch would poison the
                # persistent Adam moments for the rest of the stream.
                assert torch.isfinite(loss), "SAE refit batch loss is not finite"
                self.optimizer.zero_grad()
                loss.backward()
                self.sae.adjust_decoder_gradient_()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                self.optimizer.step()
                self.sae.normalize_decoder_()
                losses.append(loss.item())
        self.sae.eval()
        mean_loss = float(np.mean(losses))
        dead = self.sae.steps_since_activation > self.sae.dead_neuron_threshold_steps
        return {
            "sae_refit_loss": mean_loss,
            "sae_dead_ratio": float(dead.float().mean()),
            "sae_n_train": len(train),
        }


class SAEConceptGenerator:
    """Proposes SAE neurons discriminative for the pseudo-label y as concepts."""

    def __init__(
        self,
        state: SAEState,
        selection_method: str = "correlation",
        text_column: str = "text",
    ):
        self.state = state
        self.selection_method = selection_method
        self.text_column = text_column
        self.config = SimpleNamespace(
            training=SimpleNamespace(num_greedy_holdout=0, train_frac=1.0),
            concept=SimpleNamespace(max_meta_concepts=0),
        )

    def generate_initial_concepts(self, init_seed, data_df: pd.DataFrame, concept_selector):
        """Select all SAE neurons as concepts."""
        acts = self.state.activations(self.state.lookup(data_df[self.text_column]))
        n_neurons = acts.shape[1]
        self.state.neuron_idxs = list(range(n_neurons))
        return [{"concept": f"sae_neuron_{i}", "neuron_idx": i} for i in range(n_neurons)]

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[Dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """Propose new concepts."""
        # no new concepts, since they are already pointing to the right thing
        if self.state.refit_every is not None:
            # NOTE: assumes that refit_every == 1
            # just refit the concepts if asked to "generate new concepts"
            self.state.refit(self.state.lookup(init_data_df[self.text_column]))
        return []


class SAEConceptProjector:
    """Projects embeddings to SAE activations of selected neurons."""

    def __init__(self, state: SAEState):
        self.state = state

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """SAE activations of selected neurons, shape (n, n_concepts)."""
        return self.state.activations(embeddings)[:, list(self.state.neuron_idxs)]
