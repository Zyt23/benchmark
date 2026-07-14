#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
FORECAST_ROOT="${FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_benchmark_matrix_20260714/server_logs}"
DATASETS_ALL="${DATASETS_ALL:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
FORECAST_ANCHORS="${FORECAST_ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
FORECAST_GPU_LIST="${FORECAST_GPU_LIST:-0 1 2 3}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-external_models/chronos-2}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

read -r -a forecast_gpus <<< "${FORECAST_GPU_LIST}"
anchor_idx=0
for anchor in ${FORECAST_ANCHORS}; do
  gpu="${forecast_gpus[$((anchor_idx % ${#forecast_gpus[@]}))]}"
  anchor_idx=$((anchor_idx + 1))
  log="${LOG_ROOT}/zeroshot_forecast_${anchor}_chronos2.log"
  pid_file="${LOG_ROOT}/zeroshot_forecast_${anchor}_chronos2.pid"
  echo "[launch] ${anchor} gpu=${gpu} -> ${log}"
  (
    env \
      RUN_TAG="matrix_zeroshot_${anchor}_chronos2_20260714" \
      DATASETS="${DATASETS_ALL}" \
      MODELS="Chronos2" \
      COMPACT_ROOT="${FORECAST_ROOT}/${anchor}" \
      QAR_SPLIT_STRATEGY=per_class_chrono \
      PYTHON="${PYTHON}" \
      CUDA_DEVICES="${gpu}" \
      BATCH_SIZE="${ZERO_BATCH_SIZE:-8}" \
      NUM_WORKERS="${NUM_WORKERS:-0}" \
      SEQ_LEN=60 \
      LABEL_LEN=20 \
      PRED_LEN=20 \
      CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH}" \
      bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
  ) > "${log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
done

echo "[done] launched Chronos2 zero-shot forecast jobs"
