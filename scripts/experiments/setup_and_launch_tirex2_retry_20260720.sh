#!/usr/bin/env bash
set -euo pipefail

# Prepare an isolated TiRex-2 environment and launch the context-length sweep.
# Network/model access is retried at least three times, 30 minutes apart by
# default, because the training server can temporarily lose outbound access.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
BASE_ENV="${BASE_ENV:-${HOME}/anaconda3/envs/chronos2}"
TIREX2_ENV="${TIREX2_ENV:-${HOME}/anaconda3/envs/tirex2}"
CONDA="${CONDA:-${HOME}/anaconda3/bin/conda}"
ATTEMPTS="${ATTEMPTS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"

cd "${PROJECT_ROOT}"

access_ok=0
for attempt in $(seq 1 "${ATTEMPTS}"); do
  echo "[TiRex-2] access attempt ${attempt}/${ATTEMPTS}"
  if "${HOME}/anaconda3/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("NX-AI/TiRex-2", "model-config.yaml"))
PY
  then
    access_ok=1
    break
  fi
  if [ "${attempt}" -lt "${ATTEMPTS}" ]; then
    echo "[TiRex-2] access failed; waiting ${WAIT_SECONDS}s before retry"
    sleep "${WAIT_SECONDS}"
  fi
done

if [ "${access_ok}" -ne 1 ]; then
  echo "[TiRex-2] model access unavailable after ${ATTEMPTS} attempts" >&2
  exit 3
fi

if [ ! -x "${TIREX2_ENV}/bin/python" ]; then
  "${CONDA}" create --yes --prefix "${TIREX2_ENV}" --clone "${BASE_ENV}"
fi

"${TIREX2_ENV}/bin/python" -m pip install --upgrade "tirex-2==0.1.1"
"${TIREX2_ENV}/bin/python" - <<'PY'
import torch
import tirex2
print("torch", torch.__version__)
print("tirex2", getattr(tirex2, "__version__", "installed"))
PY

TIREX2_PYTHON="${TIREX2_ENV}/bin/python" \
  bash scripts/experiments/launch_tirex2_context40_20260720.sh
