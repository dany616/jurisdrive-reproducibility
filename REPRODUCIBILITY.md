# Reproducibility guide

This repository separates checks that can be reproduced from a clean clone
from experiments that require licensed data or external runtime environments.

## Level 1: clean-clone verification

Run:

```bash
python -m pip install -r requirements-lock.txt
python scripts/verify_release.py
```

Expected checks:

- all Python sources compile;
- the repository test suite passes;
- the JurisDrive CLI loads;
- the frozen N0--N3 identity `2,902 + 72,653 + 736 = 76,291` holds;
- the frozen aggregate 400-case graph/contract and 200-case dry-run checks hold; and
- a synthetic record completes graph, contract, and dry-run generation.

## Level 2: licensed-corpus audit

Obtain the **Civil Law LLM Pretraining and Instruction Tuning Data** from
[AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%AF%BC%EC%82%AC%EB%B2%95%20LLM%20%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%20%EB%B0%8F%20Instruction%20Tuning%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71841).
The repository intentionally excludes the raw judgments and full per-record
batch outputs. Use `src/analysis/audit_current_data.py` with explicit paths, as
shown in the main README. New outputs are written below ignored `artifacts/`
directories so that frozen summaries are not overwritten.

## Level 3: graph, contract, and dry-run regeneration

Supply the reproduced N0--N3 `full_run` tree through `--full-run-dir` or the
`JURISDRIVE_FULL_RUN_DIR` environment variable. First run the Level 2 audit to
generate `artifacts/audit/final_car_to_car_manifest.jsonl`. The manifest's
`source_stage`, `result_file`, and relative `source_path` fields are portable;
the per-record manifest remains outside Git.

Use fixed tier counts and preserve every generated `summary.json`, seed,
configuration file, and checksum ledger. Tier C remains review-only and must
not be auto-promoted.

## Level 4: external CARLA and VLM experiments

`requirements-carla.txt` documents the isolated CARLA 0.9.16 port/ablation
profile. Keep simulator and model-serving environments isolated. Record exact
simulator versions, model identifiers, seeds, endpoints, and output checksums
for every new CARLA or VLM execution.
