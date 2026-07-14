#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
CLS_ROOT="${CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
FORECAST_ROOT="${FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_custom_conditions_perclass_20260714/server_logs}"

DATASETS_ALL="dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14"
QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}"
STAGE="${STAGE:-all}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

launch() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo "[launch] ${name} -> ${log_file}"
  ("$@") > "${log_file}" 2>&1 < /dev/null &
  echo $! > "${LOG_ROOT}/${name}.pid"
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "classification" ]]; then
launch cls_core env \
  RUN_TAG=perclass_cls_core_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="TimesNet DLinear iTransformer" \
  COMPACT_ROOT="${CLS_ROOT}" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=4,5 \
  USE_MULTI_GPU=1 \
  DEVICES=0,1 \
  BATCH_SIZE=128 \
  TRAIN_EPOCHS=12 \
  PATIENCE=3 \
  NUM_WORKERS=4 \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh

launch cls_transformer env \
  RUN_TAG=perclass_cls_transformer_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="Transformer" \
  COMPACT_ROOT="${CLS_ROOT}" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=6 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=4 \
  TRAIN_EPOCHS=12 \
  PATIENCE=3 \
  NUM_WORKERS=2 \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh

launch cls_patchtst env \
  RUN_TAG=perclass_cls_patchtst_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="PatchTST" \
  COMPACT_ROOT="${CLS_ROOT}" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=1 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=16 \
  TRAIN_EPOCHS=12 \
  PATIENCE=3 \
  NUM_WORKERS=2 \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "forecast" ]]; then
launch forecast_2_3 env \
  RUN_TAG=perclass_forecast_predict_2_3_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="Transformer TimesNet PatchTST DLinear iTransformer" \
  COMPACT_ROOT="${FORECAST_ROOT}/predict_2_3" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=4 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=256 \
  TRAIN_EPOCHS=5 \
  PATIENCE=2 \
  NUM_WORKERS=2 \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh

launch forecast_4_5 env \
  RUN_TAG=perclass_forecast_predict_4_5_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="Transformer TimesNet PatchTST DLinear iTransformer" \
  COMPACT_ROOT="${FORECAST_ROOT}/predict_4_5" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=5 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=256 \
  TRAIN_EPOCHS=5 \
  PATIENCE=2 \
  NUM_WORKERS=2 \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh

launch forecast_5_6 env \
  RUN_TAG=perclass_forecast_predict_5_6_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="Transformer TimesNet PatchTST DLinear iTransformer" \
  COMPACT_ROOT="${FORECAST_ROOT}/predict_5_6" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=6 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=256 \
  TRAIN_EPOCHS=5 \
  PATIENCE=2 \
  NUM_WORKERS=2 \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh

launch forecast_8_9 env \
  RUN_TAG=perclass_forecast_predict_8_9_20260714 \
  DATASETS="${DATASETS_ALL}" \
  MODELS="Transformer TimesNet PatchTST DLinear iTransformer" \
  COMPACT_ROOT="${FORECAST_ROOT}/predict_8_9" \
  QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
  PYTHON="${PYTHON}" \
  CUDA_DEVICES=1 \
  USE_MULTI_GPU=0 \
  BATCH_SIZE=256 \
  TRAIN_EPOCHS=5 \
  PATIENCE=2 \
  NUM_WORKERS=2 \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
fi

echo "[done] launched per-class custom-condition experiments stage=${STAGE}"
