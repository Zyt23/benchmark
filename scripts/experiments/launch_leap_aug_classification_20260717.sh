#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_leap_aug_cls_20260717}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_history80_leap_aug_20260717/server_logs/classification}"
DATASETS="${DATASETS:-dataset9_aug0_1000 dataset9_aug0_2000 dataset9_aug0_4000 dataset9_aug0_19119 dataset10_aug0_1000 dataset10_aug0_2000 dataset10_aug0_4000 dataset10_aug0_19119 dataset12_aug0_1000 dataset12_aug0_2000 dataset12_aug0_4000 dataset12_aug0_19119}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-leap_aug_cls}"

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
  echo "[launch] model=${model} gpu=${gpu} run_tag=${run_tag}"
  (
    env \
      DATASETS="${DATASETS}" \
      MODELS="${model}" \
      COMPACT_ROOT="${COMPACT_ROOT}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${gpu}" \
      USE_MULTI_GPU=0 \
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}" \
      PATIENCE="${PATIENCE:-4}" \
      BATCH_SIZE="${BATCH_SIZE:-96}" \
      CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}" \
      EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}" \
      QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
      CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
      bash scripts/classification/run_QAR_tsfile_shiftN80.sh
  ) > "${log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
done

echo "[done] launched LEAP-augmented classification jobs"
