#!/usr/bin/env bash
set -euo pipefail

# Resume only the experiment cells still missing after the 20260717 sweep.
# Results/logs go to experiment_artifacts; source code stays in the repository.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
BLOCKS="${BLOCKS:-normal_aug_patchtst patch_sweep forecast_anomaly}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
GPU_LIST="${GPU_LIST//,/ }"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
PYTHON_BIN="${PYTHON_BIN:-/home/para/anaconda3/bin/python}"

cd "${PROJECT_ROOT}"

contains_block() {
  [[ " ${BLOCKS} " == *" $1 "* ]]
}

run_normal_aug_patchtst_repair() {
  local datasets="dataset5_normalx2 dataset5_normalx4 dataset6_normalx2 dataset6_normalx4 dataset7_normalx2 dataset7_normalx4 dataset9_normalx2 dataset9_normalx4 dataset10_normalx2 dataset10_normalx4 dataset12_normalx2 dataset12_normalx4"
  local compact_root="datasetall_tsfile_compact_normal_aug_cls_20260717"
  local log_root="${ARTIFACT_ROOT}/server_logs/normal_aug_patchtst_repair_20260720"
  mkdir -p "${log_root}"
  printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${log_root}/expected_jobs.tsv"
  read -r -a gpus <<< "${GPU_LIST}"
  local job_idx=0

  for dataset in ${datasets}; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do sleep 20; done
    local gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    local run_tag="normalx_cls_PatchTST_${dataset}_20260720_repair_b2"
    printf "classification\tnormalx\t\tPatchTST\t%s\t%s\t%s\n" \
      "${dataset}" "${run_tag}" "${compact_root}" >> "${log_root}/expected_jobs.tsv"
    (
      env DATASETS="${dataset}" MODELS=PatchTST COMPACT_ROOT="${compact_root}" \
        RUN_TAG="${run_tag}" CUDA_DEVICES="${gpu}" USE_MULTI_GPU=0 \
        TRAIN_EPOCHS="${CLS_EPOCHS:-30}" PATIENCE="${CLS_PATIENCE:-4}" \
        BATCH_SIZE="${CLS_BATCH_SIZE:-2}" CLASS_WEIGHT=balanced \
        EARLY_STOP_METRIC=macro_f1 QAR_SPLIT_STRATEGY=per_class_chrono \
        CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
        bash scripts/classification/run_QAR_tsfile_shiftN80.sh
    ) > "${log_root}/${run_tag}.launcher.log" 2>&1 < /dev/null &
  done
  wait
}

run_patch_sweep() {
  env PATCH_VALUES="8 4 2 1" GPU_LIST="${GPU_LIST}" MAX_PARALLEL="${MAX_PARALLEL}" \
    LOG_ROOT="${ARTIFACT_ROOT}/server_logs/patchlen_missing_20260720" \
    RUN_SUFFIX=20260720_missing \
    bash scripts/experiments/launch_patchlen_sweep_20260717.sh

  # The old patch=16 classification run missed dataset14.  This also refreshes
  # its eight forecast cells, which is harmless and makes this shard complete.
  env DATASETS=dataset14 PATCH_VALUES=16 GPU_LIST="${GPU_LIST}" MAX_PARALLEL="${MAX_PARALLEL}" \
    LOG_ROOT="${ARTIFACT_ROOT}/server_logs/patchlen16_dataset14_20260720" \
    RUN_SUFFIX=20260720_dataset14 \
    bash scripts/experiments/launch_patchlen_sweep_20260717.sh
}

run_forecast_anomaly() {
  env RUN_MODES=predict80 GPU_LIST="${GPU_LIST}" MAX_PARALLEL="${MAX_PARALLEL}" \
    PYTHON_BIN="${PYTHON_BIN}" FORECAST_ANOMALY_SCORE=auto \
    THRESHOLD_SOURCE=val_mixed_best_f1 \
    LOG_ROOT="${ARTIFACT_ROOT}/server_logs/forecast_head_anomaly_20260720" \
    RUN_SUFFIX=20260720_val_auto \
    bash scripts/experiments/launch_forecast_head_anomaly_20260719.sh
}

contains_block normal_aug_patchtst && run_normal_aug_patchtst_repair
contains_block patch_sweep && run_patch_sweep
contains_block forecast_anomaly && run_forecast_anomaly
