#!/usr/bin/env bash
set -euo pipefail

# Re-run PatchTST after fixing Model(configs) to honor patch_len/stride.
# Forecast waves use all eight GPUs in two groups; classification starts only
# after forecasting and uses batch size 2 for the long 2000-point sequence.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
RUN_SUFFIX="${RUN_SUFFIX:-20260720_paramfix}"

cd "${PROJECT_ROOT}"

forecast_wave() {
  local patch_len="$1"
  local gpu_list="$2"
  env PATCH_VALUES="${patch_len}" PATCH_FORECAST_MODELS=PatchTST \
    RUN_CLASSIFICATION=0 RUN_FORECAST=1 \
    GPU_LIST="${gpu_list}" MAX_PARALLEL=4 TSLIB_USE_SDPA=1 \
    LOG_ROOT="${ARTIFACT_ROOT}/server_logs/patchlen_paramfix_forecast_p${patch_len}_20260720" \
    RUN_SUFFIX="${RUN_SUFFIX}" \
    bash scripts/experiments/launch_patchlen_sweep_20260717.sh
}

echo "[wave 1] patch 16 and 8"
forecast_wave 16 "0 1 2 3" &
forecast_wave 8 "4 5 6 7" &
wait

echo "[wave 2] patch 4 and 2"
forecast_wave 4 "0 1 2 3" &
forecast_wave 2 "4 5 6 7" &
wait

echo "[wave 3] patch 1"
forecast_wave 1 "0 1 2 3"

echo "[classification] all patch values"
env PATCH_VALUES="16 8 4 2 1" PATCH_FORECAST_MODELS=PatchTST \
  RUN_CLASSIFICATION=1 RUN_FORECAST=0 \
  GPU_LIST="0 1 2 3 4" MAX_PARALLEL=5 TSLIB_USE_SDPA=1 \
  CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-2}" \
  LOG_ROOT="${ARTIFACT_ROOT}/server_logs/patchlen_paramfix_classification_20260720" \
  RUN_SUFFIX="${RUN_SUFFIX}" \
  bash scripts/experiments/launch_patchlen_sweep_20260717.sh

echo "[done] PatchTST parameter-fix sweep"
