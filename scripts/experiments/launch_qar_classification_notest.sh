#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe five-model QAR classification benchmark.
# Runs a strict split audit first, then launches one worker per model with a
# memory-safe batch size for the 2235/2920-point classification sequences.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
RUN_SUFFIX="${RUN_SUFFIX:-20260719_v1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-4}"
LOG_ROOT="${LOG_ROOT:-/share/workspace/monren/prsov/qar_benchmark_runs/classification_notest_${RUN_SUFFIX}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/share/workspace/monren/prsov/qar_benchmark_checkpoints/classification_notest_${RUN_SUFFIX}}"
AUDIT_DIR="${AUDIT_DIR:-experiment_artifacts/QAR_leakage_safe_rerun_20260719/split_audit}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${CHECKPOINT_ROOT}" "${AUDIT_DIR}"

"${PYTHON:-/home/para/anaconda3/bin/python}" scripts/analysis/audit_qar_chrono_splits.py \
  --compact_root "${COMPACT_ROOT}" \
  --output_dir "${AUDIT_DIR}" \
  --strict

read -r -a gpus <<< "${GPU_LIST}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

expected="${LOG_ROOT}/expected_jobs.tsv"
printf "model\trun_tag\tbatch_size\tcompact_root\n" > "${expected}"

job_index=0
for model in ${MODELS}; do
  gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
  job_index=$((job_index + 1))
  case "${model}" in
    Transformer) batch_size="${TRANSFORMER_BATCH_SIZE:-4}" ;;
    PatchTST) batch_size="${PATCHTST_BATCH_SIZE:-16}" ;;
    TimesNet) batch_size="${TIMESNET_BATCH_SIZE:-96}" ;;
    *) batch_size="${DEFAULT_BATCH_SIZE:-128}" ;;
  esac
  run_tag="classification_notest_${model}_${RUN_SUFFIX}"
  worker_log="${LOG_ROOT}/${model}.launcher.log"
  pid_file="${LOG_ROOT}/${model}.pid"
  printf "%s\t%s\t%s\t%s\n" \
    "${model}" "${run_tag}" "${batch_size}" "${COMPACT_ROOT}" >> "${expected}"

  (
    env \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      MODELS="${model}" \
      COMPACT_ROOT="${COMPACT_ROOT}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${gpu}" \
      USE_MULTI_GPU=0 \
      TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
      PATIENCE="${PATIENCE}" \
      BATCH_SIZE="${batch_size}" \
      CLASS_WEIGHT=balanced \
      EARLY_STOP_METRIC=macro_f1 \
      QAR_SPLIT_STRATEGY=per_class_chrono \
      SAVE_EPOCH_CHECKPOINTS=0 \
      CHECKPOINTS="${CHECKPOINT_ROOT}/${run_tag}" \
      bash scripts/classification/run_QAR_tsfile_shiftN80.sh
  ) > "${worker_log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
  echo "[launch] model=${model} gpu=${gpu} batch=${batch_size} pid=$(cat "${pid_file}")"
done

echo "[done] launched classification workers; expected jobs: ${expected}"
