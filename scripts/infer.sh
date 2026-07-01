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
PROMPT="${PROMPT:-请用一句话解释 LoRA。}"
INPUT_TEXT="${INPUT_TEXT:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个专业、可靠、简洁的中文助手。}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
DO_SAMPLE="${DO_SAMPLE:-true}"
NUM_BEAMS="${NUM_BEAMS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SEED="${SEED:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

CMD=(
  "${PYTHON_CMD[@]}"
  "inference.py"
  "--model-name-or-path" "${MODEL_NAME_OR_PATH}"
  "--adapter-path" "${ADAPTER_PATH}"
  "--prompt" "${PROMPT}"
  "--input-text" "${INPUT_TEXT}"
  "--system-prompt" "${SYSTEM_PROMPT}"
  "--torch-dtype" "${TORCH_DTYPE}"
  "--attn-implementation" "${ATTN_IMPLEMENTATION}"
  "--device-map" "${DEVICE_MAP}"
  "--max-new-tokens" "${MAX_NEW_TOKENS}"
  "--temperature" "${TEMPERATURE}"
  "--top-p" "${TOP_P}"
  "--top-k" "${TOP_K}"
  "--repetition-penalty" "${REPETITION_PENALTY}"
  "--num-beams" "${NUM_BEAMS}"
  "--log-level" "${LOG_LEVEL}"
)

if [[ "${DO_SAMPLE}" == "true" ]]; then
  CMD+=("--do-sample")
else
  CMD+=("--no-do-sample")
fi

if [[ -n "${SEED}" ]]; then
  CMD+=("--seed" "${SEED}")
fi

if [[ -n "${OUTPUT_FILE}" ]]; then
  CMD+=("--output-file" "${OUTPUT_FILE}")
fi

echo "[INFO] Working directory: ${ROOT_DIR}"
echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
