# Agent Research

Research code for studying how prior assessments, source labels, and upstream agent outputs influence subsequent LLM and human judgments. The repository includes experiments on source attribution, belief firewalls, dose response, low-information upstream agents, and human comparison studies.

This public repository contains source code, configuration templates, documentation, and notebooks with all execution state removed. It does not contain raw research data, patient cards or identifiers, model responses, statistical results, clinician questionnaires, credentials, credential files, intermediate outputs, or test artifacts.

## Repository structure

| Path | Purpose |
| --- | --- |
| `run_full.py` | Immutable system prompt, 17 experimental conditions, and the main entry point |
| `multi_runner.py` | Multi-model scheduling, randomized task order, concurrency, resume support, sentinel checks, output locking, and run metadata |
| `provider_adapters.py` | OpenAI-compatible, Anthropic, and Gemini request adapters and response validation |
| `prepare_experiment_assets.py` | Local generation and verification of prompt fingerprints and fixed sample lists |
| `export.py` | Converts local run records into analysis tables and excludes smoke-test data |
| `analyze_full.py` / `analyze_extra.py` | Effect, clinical metric, token, failure-rate, and latency analyses |
| `pull_vs_accuracy.py` | Relates pull coefficients to accuracy using locally prepared summary tables |
| `providers.yaml` | Provider configuration template containing environment-variable names only |
| `notebooks/` | Low-information pipeline and OpenAI follow-up workflows |
| `doctors/` | Human-comparison questionnaire generation and analysis logic |
| `tools/` | Publication checks and allowlist-based source packaging |

## Installation and offline validation

Python 3.10 or later is recommended. Run all commands from the repository root:

```bash
python -m pip install -r requirements.txt
python tools/check_upload.py
```

Optional notebook and human-study dependencies are available separately:

```bash
python -m pip install -r requirements-notebooks.txt
```

## Preparing local data

Prepare a local `data/cards.json` from data obtained through your own authorized access channel. Each record must provide `RID`, outcome labels, and the card fields used by the selected experiment month. Then create immutable experiment assets:

```bash
python prepare_experiment_assets.py --cards data/cards.json --write
python prepare_experiment_assets.py
```

Asset generation requires at least 300 eligible cases. The original study fingerprints and patient lists are deliberately absent, so first-time users must establish a new baseline from their own data. A newly generated baseline must not be used to overwrite or resume the original study runs.

Inspect the planned call count without contacting a provider:

```bash
python run_full.py --arm deepseek-flash@off --rid-file patients/patients_n100.txt --conds main --repeats 5 --plan
```

`providers.yaml` preserves the experiment configuration as a template. Verify current model identifiers, parameter support, prices, and data terms before any paid run.

Runtime credentials are read from process environment variables named by each provider configuration. They must never be written into repository files, notebooks, command-line arguments, outputs, or Git history. See [RUNBOOK.md](RUNBOOK.md) for the complete workflow.

## Publication safeguards

`SOURCE_MANIFEST.txt` is the explicit publication allowlist. `.gitignore` excludes local data, credentials, run outputs, analysis results, and caches. Add every new source file to the manifest deliberately.

```bash
python tools/check_upload.py
python tools/package_source.py
```

The packaging tool reads only manifest entries, writes the archive outside the repository by default, and never recursively copies experiment directories or Git history. It refuses to overwrite an existing archive.

Before pushing, validate the actual Git index:

```bash
git add .
python tools/check_upload.py --staged
```

If a notebook has been edited or executed, clear its outputs, execution counts, cell metadata, and attachments before staging it. The checker reports only the location of a suspected credential and never prints its value.
