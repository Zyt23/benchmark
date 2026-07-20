#!/usr/bin/env bash
set -euo pipefail

# Repair cells interrupted when the server root filesystem temporarily filled.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
GPU_LIST="${GPU_LIST:-5 6 7}"
RUN_SUFFIX="${RUN_SUFFIX:-20260720_paramfix_repair}"

cd "${PROJECT_ROOT}"
read -r -a gpus <<< "${GPU_LIST}"

run_group() {
  local patch_len="$1"
  local anchor="$2"
  local datasets="$3"
  local gpu="$4"
  local log_root="${ARTIFACT_ROOT}/server_logs/patchlen_paramfix_repair_p${patch_len}_${anchor}_20260720"
  env PATCH_VALUES="${patch_len}" ANCHORS="${anchor}" DATASETS="${datasets}" \
    PATCH_FORECAST_MODELS=PatchTST RUN_CLASSIFICATION=0 RUN_FORECAST=1 \
    GPU_LIST="${gpu}" MAX_PARALLEL=1 TSLIB_USE_SDPA=1 \
    LOG_ROOT="${log_root}" RUN_SUFFIX="${RUN_SUFFIX}" \
    bash scripts/experiments/launch_patchlen_sweep_20260717.sh
}

run_group 4 predict_4_5 "dataset14" "${gpus[0]}" &
run_group 2 predict_2_3 "dataset13" "${gpus[1]}" &
run_group 2 predict_4_5 "dataset13 dataset14" "${gpus[2]}" &
run_group 2 predict_5_6 "dataset12 dataset13 dataset14" "${gpus[0]}" &
run_group 2 predict_8_9 "dataset13" "${gpus[1]}" &
wait

echo "[done] repaired PatchTST parameter-fix forecast cells"
