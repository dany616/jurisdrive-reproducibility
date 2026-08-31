#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: run_experimental_evaluation_400.sh RUN_ROOT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${PAPER_ROOT}/.." && pwd)}"
LOCAL_LLM_ROOT="${LOCAL_LLM_ROOT:-${WORKSPACE_ROOT}/LocalLLM}"
ZEROSHOT_ROOT="${ZEROSHOT_ROOT:-${LOCAL_LLM_ROOT}/zeroshot_test}"
INPUT_DIR="${INPUT_DIR:-${ZEROSHOT_ROOT}/pipelines/car_to_car_filter/full_run/output/ambiguous}"
RUNNER="${RUNNER:-${ZEROSHOT_ROOT}/pipelines/car_to_car_filter/run_ambiguous_llm_filter.py}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:?Set MODEL to the exact ID returned by ${API_BASE}/models}"
VLLM_CONTAINER="${VLLM_CONTAINER:-qwen35-vlm-server}"

mkdir -p "${RUN_ROOT}/concurrency_sweep" "${RUN_ROOT}/full_400_c8"

curl -fsS "${API_BASE}/models" > "${RUN_ROOT}/models.json"
docker inspect "${VLLM_CONTAINER}" > "${RUN_ROOT}/container_inspect.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv > "${RUN_ROOT}/gpu_inventory.csv"
lscpu > "${RUN_ROOT}/cpu_inventory.txt"
free -b > "${RUN_ROOT}/memory_inventory.txt"

export PYTHONPATH="${ZEROSHOT_ROOT}"

# Fixed 16-case workload for latency/throughput scaling. The smoke run is
# excluded so every reported sweep point starts from the same warm server.
for concurrency in 1 2 4 8 16; do
  out_dir="${RUN_ROOT}/concurrency_sweep/c${concurrency}"
  mkdir -p "${out_dir}"
  python3 "${RUNNER}" \
    --input-dir "${INPUT_DIR}" \
    --output-root "${out_dir}" \
    --api-base-url "${API_BASE}" \
    --model-name "${MODEL}" \
    --max-tokens 256 \
    --max-concurrency "${concurrency}" \
    --limit 16 \
    --reset-logs \
    > "${out_dir}/stdout.log" 2> "${out_dir}/stderr.log"
done

# Full 400-request experiment. Resource traces are sampled once per second.
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits \
  --loop=1 > "${RUN_ROOT}/full_400_c8/gpu_samples.csv" &
gpu_monitor_pid=$!

vmstat 1 > "${RUN_ROOT}/full_400_c8/vmstat_samples.txt" &
vmstat_monitor_pid=$!

cleanup_monitors() {
  kill "${gpu_monitor_pid}" "${vmstat_monitor_pid}" 2>/dev/null || true
  wait "${gpu_monitor_pid}" "${vmstat_monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitors EXIT

python3 "${RUNNER}" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${RUN_ROOT}/full_400_c8" \
  --api-base-url "${API_BASE}" \
  --model-name "${MODEL}" \
  --max-tokens 256 \
  --max-concurrency 8 \
  --limit 400 \
  --reset-logs \
  > "${RUN_ROOT}/full_400_c8/stdout.log" 2> "${RUN_ROOT}/full_400_c8/stderr.log"

cleanup_monitors
trap - EXIT

docker logs --since 2h "${VLLM_CONTAINER}" > "${RUN_ROOT}/vllm_server.log" 2>&1
grep -Eai 'out of memory|CUDA OOM|Traceback|engine core.*failed|NCCL (WARN|ERROR)' \
  "${RUN_ROOT}/vllm_server.log" > "${RUN_ROOT}/vllm_error_scan.txt" || true

printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/COMPLETED"
