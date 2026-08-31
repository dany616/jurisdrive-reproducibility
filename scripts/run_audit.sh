#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${PAPER_ROOT}/.." && pwd)}"
LOCAL_LLM_ROOT="${LOCAL_LLM_ROOT:-${WORKSPACE_ROOT}/LocalLLM}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-${PAPER_ROOT}/artifacts/migration_runs/${RUN_STAMP}/n0_n3_audit}"

"${PYTHON_BIN}" "${PAPER_ROOT}/src/analysis/audit_current_data.py" \
  --full-run-dir "${LOCAL_LLM_ROOT}/zeroshot_test/pipelines/car_to_car_filter/full_run" \
  --raw-dir "${LOCAL_LLM_ROOT}/zeroshot_test/inputs/raw" \
  --zeroshot-dir "${LOCAL_LLM_ROOT}/zeroshot_test/outputs/zeroshot_done" \
  --output-dir "${OUTPUT_DIR}"

printf 'AUDIT_OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
