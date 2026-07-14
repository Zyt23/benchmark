#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
CLS_ROOT="${CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_benchmark_matrix_20260714/server_logs}"
DATASETS_ALL="${DATASETS_ALL:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS_EXTRA="${MODELS_EXTRA:-LITE}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

log="${LOG_ROOT}/classification_lite.log"
pid_file="${LOG_ROOT}/classification_lite.pid"
echo "[launch] LITE classification -> ${log}"
(
  env \
    RUN_TAG="${RUN_TAG:-matrix_cls_lite_20260714}" \
    DATASETS="${DATASETS_ALL}" \
    MODELS="${MODELS_EXTRA}" \
    COMPACT_ROOT="${CLS_ROOT}" \
    QAR_SPLIT_STRATEGY=per_class_chrono \
    PYTHON="${PYTHON}" \
    CUDA_DEVICES="${CLASS_CUDA_DEVICES:-5}" \
    DEVICES=0 \
    USE_MULTI_GPU=0 \
    BATCH_SIZE="${CLASS_BATCH_SIZE:-32}" \
    TRAIN_EPOCHS="${CLASS_TRAIN_EPOCHS:-12}" \
    PATIENCE="${CLASS_PATIENCE:-3}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    CLASS_WEIGHT=balanced \
    EARLY_STOP_METRIC=macro_f1 \
    bash scripts/classification/run_QAR_tsfile_shiftN80.sh
) > "${log}" 2>&1 < /dev/null &
echo $! > "${pid_file}"

echo "[done] launched LITE classification"
