# Local Runbook

Run commands from the repository root. Install dependencies, prepare a local `data/cards.json`, and use `python prepare_experiment_assets.py --write` to establish fingerprints and sample lists for your own data. Do not regenerate a baseline for an existing study run.

## 1. Configuration and planning

Provider endpoints, model identifiers, and request options live in `providers.yaml`. The configuration stores only the name of the process environment variable used at runtime; credential values must remain outside repository files and Git history.

Create an offline plan without contacting a provider:

```bash
python run_full.py --provider-config providers.yaml --arm deepseek-flash@off --rid-file patients/patients_n100.txt --conds main --repeats 5 --plan
```

## 2. Smoke test and full run

After supplying the runtime credential in the current process environment, make three smoke-test calls:

```bash
python run_full.py --arm deepseek-flash@off --rid-file patients/patients_n100.txt --conds main --repeats 5 --smoke 3
```

After the smoke test succeeds, start the full run:

```bash
python run_full.py --arm deepseek-flash@off --rid-file patients/patients_n100.txt --conds main --repeats 5 --workers 8
```

Results are written under `runs/<vendor>/<arm>/`. Repeating the same command resumes an interrupted run. The runner validates completed-record identity and uses output locks to prevent two processes from writing the same file. Smoke-test records live under `runs/_smoke/` and are excluded from normal exports.

Use identical patient lists when comparing models or arms. `--arms arm1,arm2` interleaves multiple arms within one global schedule. Keep prompts, condition names, patient order, and repeat counts fixed.

## 3. Export and analysis

```bash
python export.py --cards data/cards.json --adnimerge data/ADNIMERGE.csv --outdir out_multimodel
python analyze_full.py --outdir out_multimodel
python analyze_extra.py --outdir out_multimodel
```

Omit `--adnimerge` when demographic merging is unnecessary. Keep all exported tables, QC reports, statistics, and figure data local.

`pull_vs_accuracy.py` reads `patient_long.csv`, `perpatient_pull.csv`, and optional `pipeline8_*.csv` files from a local `figdata/` directory. These are additional analysis inputs and are not all produced directly by the main export script.

## 4. Notebooks

Start Jupyter from the repository root:

- `notebooks/model_followup.ipynb`: OpenAI follow-up workflow with plan, smoke test, and full run.
- `notebooks/lowinfo_pipeline_robust.ipynb`: provider-compatible pipeline with throttling, retry, multi-pass recovery, and resume support.
- `notebooks/lowinfo_pipeline_openai.ipynb`: OpenAI low-information upstream pipeline.
- `notebooks/lowinfo_pipeline.ipynb`: earlier pipeline implementation retained for comparison.

Review the model and endpoint configuration before running a notebook. Execution can make real provider requests and create local outputs; do not commit those outputs.

Human-comparison instructions are in [doctors/README.md](doctors/README.md).

## 5. Publication checks

```bash
python tools/check_upload.py
git add .
python tools/check_upload.py --staged
python tools/package_source.py
```

Publish only the isolated `agent-research` Git repository or the allowlist-generated archive. The parent working directory contains private local assets and is outside the publication scope.
