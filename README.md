# Monitoring-by-Betting

Code for the manuscript *Learning to Detect Shifts in Unstructured Data Streams via Monitoring-by-Betting* (see `main.tex`).

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd monitoring-by-betting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you did not clone with `--recurse-submodules`:

```bash
git submodule update --init
```

### LLM credentials

The LLM-based bettors call `azure/gpt-4.1-mini-2025-04-14` via `litellm`. Set the usual Azure OpenAI environment variables (`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`) before running those arms, or point to a different model by editing `LLM_MODELS` at the top of each experiment's `sconscript`.

## Running the experiments

The repo contains three experiments, each a nestly+SCons pipeline under its own `exp_<name>/` directory:

- `exp_cebab/` — CEBaB restaurant-review text stream with an injected ambiance-mention shift (Section 4.1).
- `exp_cub/` — CUB-200-2011 bird-image stream with an injected yellow-bill or white-crown shift (Section 4.2).
- `exp_ai_scribe/` — real-world clinical AI-scribe deployment data; **data is restricted and not included.** The pipeline is released so that the methodology is reproducible on comparable datasets (Section 4.3).

Build a single experiment:

```bash
scons exp_cebab/_output
scons exp_cub/_output
```

Or everything:

```bash
scons
```

Pass `--slurm` to launch each stage through `srun`; the sconscripts' `_run_cmd` helper is already wired for a SLURM cluster.

## Code layout

- `src/` — top-level entry scripts invoked by the sconscripts (`run_monitor.py`, `run_ctm_monitor.py`, `run_ensemble_monitor.py`, `assemble_*.py`, `aggregate_*.py`, `plot_*.py`, …).
- `src/llm_prior_monitoring/` — the installable Python package: asymptotic e-detectors, Classifier-based Betting, conformal test martingales for the fixed-monitor baselines, LLM-concept proposer, SAE and BERTopic components, feature extractors, and the sequential-concept-learner driver.
- `src/llm_prior_monitoring/prompts/` — prompt templates used by the package.
- `exp_*/prompts/` — per-experiment prompt templates and drift-direction descriptors.
- `exp_cub/tilt_selection/` — committed JSON+CSV artifacts for the CUB tilt-selection step.
- `HACHI/` — git submodule providing the LLM-call batching utilities (`HACHI.src.llm_utils`).

## Notes

- `exp_cebab` and `exp_cub` build population data from public sources (HuggingFace `CEBaB` parquets, CUB-200-2011); re-running the `build_population` / data-assembly targets will fetch them.
- `exp_ai_scribe` requires IRB-approved clinical data access and will not run without that data in place.
- Embeddings are produced with `sentence-transformers/all-MiniLM-L6-v2` (text) and `openai/clip-vit-base-patch32` (images).
