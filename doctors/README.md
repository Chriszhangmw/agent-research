# Human Comparison Tools

This directory publishes only spreadsheet-processing and statistical-analysis logic. Case tables, blank questionnaire templates, returned questionnaires, clinician names, allocation records, and reference statistics must remain local.

Run `fill_round2.py` from the `doctors/` directory. It expects a local `cases.csv` file and the workbook named by the `TEMPLATE` constant in the script. Generate one round-two workbook or a batch:

```bash
python fill_round2.py clinician_round1.xlsx A clinician_round2.xlsx
python fill_round2.py --batch round1/ round2/
```

Keep returned round-two workbooks and their matching allocation records in a local `second/` directory. `analysis.ipynb` expects `case`, `RID`, `dx36`, `ai_base`, and `ai_beta` columns in `cases.csv`, plus a local `ai_reference.json` file.

The reference JSON must contain numeric fields `self_`, `other_phys`, `other_ai`, `self_minus_other`, `self_minus_other_se`, `beta_median`, and `disclosure_pct`; a two-element numeric array named `beta_range`; and a string field named `model`. Supply values from your own local analysis. Set `AI_REFERENCE_PATH` to use a different file location.

No actual reference statistics are included. The Excel worksheet structure and column mappings remain in the source code, but the human-comparison workflow cannot run without the local templates and research inputs.
