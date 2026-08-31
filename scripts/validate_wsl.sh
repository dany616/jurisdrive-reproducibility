#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PAPER_ROOT}/.." && pwd)"
DEFAULT_VENV_PYTHON="${WORKSPACE_ROOT}/.venvs/jurisdrive/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd -- "${PAPER_ROOT}"

"${PYTHON_BIN}" -m py_compile jurisdrive/*.py scripts/*.py tests/*.py src/analysis/*.py
"${PYTHON_BIN}" -m unittest discover -s tests -v
bash -n scripts/*.sh
"${PYTHON_BIN}" -m jurisdrive --help
