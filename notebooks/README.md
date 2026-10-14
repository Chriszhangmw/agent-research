# Notebook Workflows

All notebooks in this directory have empty outputs, execution counts, cell metadata, and attachments. Prepare local data as described in the root README before running them. Runtime credentials must be supplied through process environment variables or a hidden prompt and must never be saved in the notebook.

Use `lowinfo_pipeline_robust.ipynb` for retry, throttling, and resume support. `lowinfo_pipeline_openai.ipynb` contains the OpenAI-specific low-information workflow, and `model_followup.ipynb` runs the main follow-up experiment. `lowinfo_pipeline.ipynb` retains the earlier pipeline implementation for comparison.

Start Jupyter from the repository root. Executing these notebooks can make real provider requests and create local results. Clear notebook state and run the publication checker before committing any changes.
