#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RUN_TAG="${RUN_TAG:-qar_anomaly_oneclass_$(date +%Y%m%d_%H%M%S)}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEQ_LEN="${SEQ_LEN:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "${PROJECT_ROOT}" || exit 1
mkdir -p experiment_artifacts logs/anomaly_detection

start_shard() {
  local shard_id="$1"
  local cuda_device="$2"
  local datasets="$3"
  local summary_dir="logs/anomaly_detection/${RUN_TAG}_shard${shard_id}"
  local log_file="experiment_artifacts/${RUN_TAG}_shard${shard_id}.nohup.log"

  echo "[launch] shard=${shard_id} cuda=${cuda_device} datasets=${datasets} log=${log_file}"
  (
    export RUN_TAG="${RUN_TAG}"
    export SUMMARY_DIR="${summary_dir}"
    export DATASETS="${datasets}"
    export COMPACT_ROOT="${COMPACT_ROOT}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export CUDA_DEVICE="${cuda_device}"
    export LOCAL_GPU=0
    export SEQ_LEN="${SEQ_LEN}"
    export BATCH_SIZE="${BATCH_SIZE}"
    export TRAIN_EPOCHS="${TRAIN_EPOCHS}"
    export PATIENCE="${PATIENCE}"
    export NUM_WORKERS="${NUM_WORKERS}"
    bash scripts/anomaly_detection/run_QAR_anomaly_shiftN80.sh
  ) > "${log_file}" 2>&1 < /dev/null &
  echo $! > "experiment_artifacts/${RUN_TAG}_shard${shard_id}.pid"
}

start_shard 0 "${CUDA0:-4}" "dataset5 dataset6 dataset7"
start_shard 1 "${CUDA1:-5}" "dataset8 dataset8-1 dataset9"
start_shard 2 "${CUDA2:-6}" "dataset10 dataset11 dataset12"
start_shard 3 "${CUDA3:-1}" "dataset13 dataset14"

echo "[done] launched run_tag=${RUN_TAG}"
