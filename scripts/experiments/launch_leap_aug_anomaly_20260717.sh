#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_leap_aug_cls_20260717}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_history80_leap_aug_20260717/server_logs/anomaly}"
DATASETS="${DATASETS:-dataset9_aug0_1000 dataset9_aug0_2000 dataset9_aug0_4000 dataset9_aug0_19119 dataset10_aug0_1000 dataset10_aug0_2000 dataset10_aug0_4000 dataset10_aug0_19119 dataset12_aug0_1000 dataset12_aug0_2000 dataset12_aug0_4000 dataset12_aug0_19119}"
MODELS="${MODELS:-KANAD AnomalyTransformer TranAD USAD OmniAnomaly}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-leap_aug_anomaly_p95}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0

for model in ${MODELS}; do
  gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
  job_idx=$((job_idx + 1))
  run_tag="${RUN_TAG_PREFIX}_${model}_20260717"
  log="${LOG_ROOT}/${run_tag}.launcher.log"
  pid_file="${LOG_ROOT}/${run_tag}.pid"
  echo "[launch] anomaly model=${model} gpu=${gpu} run_tag=${run_tag}"
  (
    env \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      COMPACT_ROOT="${COMPACT_ROOT}" \
      RUN_TAG="${run_tag}" \
      DATASETS="${DATASETS}" \
      MODELS="${model}" \
      PYTHON_BIN="${PYTHON_BIN:-/home/para/anaconda3/bin/python}" \
      CUDA_DEVICE="${gpu}" \
      LOCAL_GPU=0 \
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}" \
      PATIENCE="${PATIENCE:-2}" \
      BATCH_SIZE="${BATCH_SIZE:-32}" \
      THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95.0}" \
      NUM_WORKERS="${NUM_WORKERS:-0}" \
      bash scripts/anomaly_detection/run_QAR_anomaly_shiftN80.sh
  ) > "${log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
done

echo "[done] launched LEAP-augmented anomaly jobs"
