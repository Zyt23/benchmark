#!/usr/bin/env bash
set -euo pipefail

# TiRex-2 requires its own Python>=3.11 / torch>=2.8 environment.  The model
# weights are gated; accept the Hugging Face terms and configure HF_TOKEN first.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TIREX2_PYTHON="${TIREX2_PYTHON:-${HOME}/anaconda3/envs/tirex2/bin/python}"

if [ ! -x "${TIREX2_PYTHON}" ]; then
  echo "TiRex-2 Python not found: ${TIREX2_PYTHON}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
env MODELS=TiRex2 ZERO_SHOT_PYTHON="${TIREX2_PYTHON}" \
  RUN_SUFFIX=20260720_tirex2 \
  LOG_ROOT="experiment_artifacts/QAR_extra_experiments_20260717/server_logs/foundation_context40_tirex2_20260720" \
  bash scripts/experiments/launch_foundation_context40_20260717.sh
