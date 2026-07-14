#!/usr/bin/env bash
set -euo pipefail

# Launch the QAR benchmark-matrix experiments that are directly available in
# this repository.  Missing external models are intentionally not faked; the
# Excel builder will mark them as PENDING until a real implementation is added.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
CLS_ROOT="${CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
FORECAST_ROOT="${FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_benchmark_matrix_20260714/server_logs}"

DATASETS_ALL="${DATASETS_ALL:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
FORECAST_ANCHORS="${FORECAST_ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}"
STAGE="${STAGE:-all}"

# Requested table names -> actual repository model names.
FULLSHOT_MODELS_AVAILABLE="${FULLSHOT_MODELS_AVAILABLE:-TimeMixer TimeXer iTransformer DLinear PatchTST TimesNet Autoformer}"
CLASS_MODELS_AVAILABLE="${CLASS_MODELS_AVAILABLE:-MambaSingleLayer TimesNet PatchTST DLinear iTransformer}"
ANOMALY_MODELS_AVAILABLE="${ANOMALY_MODELS_AVAILABLE:-KANAD AnomalyTransformer TranAD USAD OmniAnomaly}"

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

if [[ "${STAGE}" == "all" || "${STAGE}" == "forecast" ]]; then
  for anchor in ${FORECAST_ANCHORS}; do
    launch "fullshot_forecast_${anchor}" env \
      RUN_TAG="matrix_fullshot_${anchor}_20260714" \
      DATASETS="${DATASETS_ALL}" \
      MODELS="${FULLSHOT_MODELS_AVAILABLE}" \
      COMPACT_ROOT="${FORECAST_ROOT}/${anchor}" \
      QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
      PYTHON="${PYTHON}" \
      CUDA_DEVICES="${FORECAST_CUDA_DEVICES:-4}" \
      USE_MULTI_GPU=0 \
      BATCH_SIZE="${FORECAST_BATCH_SIZE:-128}" \
      TRAIN_EPOCHS="${FORECAST_TRAIN_EPOCHS:-5}" \
      PATIENCE="${FORECAST_PATIENCE:-2}" \
      NUM_WORKERS="${NUM_WORKERS:-2}" \
      SEQ_LEN=60 \
      LABEL_LEN=20 \
      PRED_LEN=20 \
      bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "classification" ]]; then
  launch "classification_available" env \
    RUN_TAG="matrix_cls_available_20260714" \
    DATASETS="${DATASETS_ALL}" \
    MODELS="${CLASS_MODELS_AVAILABLE}" \
    COMPACT_ROOT="${CLS_ROOT}" \
    QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY}" \
    PYTHON="${PYTHON}" \
    CUDA_DEVICES="${CLASS_CUDA_DEVICES:-5}" \
    USE_MULTI_GPU=0 \
    BATCH_SIZE="${CLASS_BATCH_SIZE:-32}" \
    TRAIN_EPOCHS="${CLASS_TRAIN_EPOCHS:-12}" \
    PATIENCE="${CLASS_PATIENCE:-3}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash scripts/classification/run_QAR_tsfile_shiftN80.sh
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "anomaly" ]]; then
  launch "anomaly_available_p95" env \
    RUN_TAG="matrix_anomaly_available_p95_20260714" \
    DATASETS="${DATASETS_ALL}" \
    MODELS="${ANOMALY_MODELS_AVAILABLE}" \
    COMPACT_ROOT="${CLS_ROOT}" \
    PYTHON_BIN="${PYTHON}" \
    CUDA_DEVICE="${ANOMALY_CUDA_DEVICE:-6}" \
    LOCAL_GPU=0 \
    BATCH_SIZE="${ANOMALY_BATCH_SIZE:-32}" \
    TRAIN_EPOCHS="${ANOMALY_TRAIN_EPOCHS:-5}" \
    PATIENCE="${ANOMALY_PATIENCE:-2}" \
    THRESHOLD_PERCENTILE=95 \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash scripts/anomaly_detection/run_QAR_anomaly_shiftN80.sh
fi

echo "[done] launched benchmark matrix available-model experiments stage=${STAGE}"
