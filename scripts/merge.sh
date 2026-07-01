#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(uv run python)
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen3-4B-Instruct-2507}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/qwen3-4b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-4b-merged}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

CMD=(
  "${PYTHON_CMD[@]}"
  "merge_lora.py"
  "--model-name-or-path" "${MODEL_NAME_OR_PATH}"
  "--adapter-path" "${ADAPTER_PATH}"
  "--output-dir" "${OUTPUT_DIR}"
  "--torch-dtype" "${TORCH_DTYPE}"
  "--attn-implementation" "${ATTN_IMPLEMENTATION}"
  "--device-map" "${DEVICE_MAP}"
  "--log-level" "${LOG_LEVEL}"
)

if [[ "${SAFE_SERIALIZATION}" == "true" ]]; then
  CMD+=("--safe-serialization")
else
  CMD+=("--no-safe-serialization")
fi

echo "[INFO] Working directory: ${ROOT_DIR}"
echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
