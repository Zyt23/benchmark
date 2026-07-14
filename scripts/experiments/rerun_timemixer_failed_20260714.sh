#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
FORECAST_ROOT="${FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_benchmark_matrix_20260714/server_logs}"
DATASETS_FAILED="${DATASETS_FAILED:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
FORECAST_ANCHORS="${FORECAST_ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

for anchor in ${FORECAST_ANCHORS}; do
  run_tag="matrix_fullshot_${anchor}_timemixer_rerun_20260714"
  log_file="${LOG_ROOT}/timemixer_rerun_${anchor}.log"
  echo "[launch] ${run_tag} -> ${log_file}"
  (
    RUN_TAG="${run_tag}" \
    DATASETS="${DATASETS_FAILED}" \
    MODELS="TimeMixer" \
    COMPACT_ROOT="${FORECAST_ROOT}/${anchor}" \
    QAR_SPLIT_STRATEGY=per_class_chrono \
    PYTHON="${PYTHON}" \
    CUDA_DEVICES="${TIMEMIXER_CUDA_DEVICES:-4}" \
    USE_MULTI_GPU=0 \
    BATCH_SIZE="${BATCH_SIZE:-128}" \
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}" \
    PATIENCE="${PATIENCE:-2}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    SEQ_LEN=60 \
    LABEL_LEN=20 \
    PRED_LEN=20 \
    bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
  ) > "${log_file}" 2>&1 < /dev/null &
  echo $! > "${LOG_ROOT}/timemixer_rerun_${anchor}.pid"
done
