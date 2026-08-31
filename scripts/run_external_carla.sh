#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BUNDLE_DIR [HOST] [PORT]" >&2
  exit 2
fi

BUNDLE_DIR=$1
HOST=${2:-127.0.0.1}
PORT=${3:-2000}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"
python3 scripts/carla_healthcheck.py --host "$HOST" --port "$PORT"
python3 -m jurisdrive run \
  --backend carla \
  --bundle-dir "$BUNDLE_DIR" \
  --host "$HOST" \
  --port "$PORT"
python3 -m jurisdrive evaluate \
  --evaluator mock \
  --bundle-dir "$BUNDLE_DIR" \
  --result "$BUNDLE_DIR/simulation_result.json"
