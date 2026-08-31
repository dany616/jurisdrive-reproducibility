# Results

This directory contains lightweight, reproducible research artifacts.

- `n0_n3_summary.json`: path-free aggregate counts, coverage, and integrity checks.
- `n_stage_counts.csv`: counts and rates for the N-stage filtering diagram.
- `n4_n6_validation_summary.json`: N4-N6 static validation and dry-run acceptance checks.
- `frozen/rq1_selective_metrics_table.csv`: compact aggregate input used to rebuild the
  legacy pending-era paper tables; it contains no judgment text.

Per-record manifests, copied runtime summaries, and machine-local audit reports
are intentionally excluded from Git. Regenerate them under `artifacts/audit/`
with `src/analysis/audit_current_data.py` when the licensed corpus is available.
