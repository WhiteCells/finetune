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

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train.yaml}"
LORA_CONFIG="${LORA_CONFIG:-config/lora.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

CMD=(
  "${PYTHON_CMD[@]}"
  "train.py"
  "--train-config" "${TRAIN_CONFIG}"
  "--lora-config" "${LORA_CONFIG}"
  "--log-level" "${LOG_LEVEL}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=("--resume-from-checkpoint" "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${ADAPTER_PATH}" ]]; then
  CMD+=("--adapter-path" "${ADAPTER_PATH}")
fi

echo "[INFO] Working directory: ${ROOT_DIR}"
echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
