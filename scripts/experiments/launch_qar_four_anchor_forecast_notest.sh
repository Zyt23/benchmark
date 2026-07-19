#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe QAR full-shot forecasting benchmark.
#
# One background worker is launched per model. Each worker runs all four flight
# phase transitions sequentially and evaluates TEST once after validation-based
# early stopping. The underlying compact cache provides 60 input points and 20
# target points around each transition.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
FORECAST_ROOT="${FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
RUN_SUFFIX="${RUN_SUFFIX:-20260719_v1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_ROOT="${LOG_ROOT:-/share/workspace/monren/prsov/qar_benchmark_runs/forecast_notest_${RUN_SUFFIX}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/share/workspace/monren/prsov/qar_benchmark_checkpoints/forecast_notest_${RUN_SUFFIX}}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${CHECKPOINT_ROOT}"

read -r -a gpus <<< "${GPU_LIST}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

expected="${LOG_ROOT}/expected_jobs.tsv"
printf "anchor\tmodel\trun_tag\tcompact_root\n" > "${expected}"

job_index=0
for model in ${MODELS}; do
  gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
  job_index=$((job_index + 1))
  worker_log="${LOG_ROOT}/${model}.launcher.log"
  pid_file="${LOG_ROOT}/${model}.pid"

  for anchor in ${ANCHORS}; do
    run_tag="forecast_notest_${anchor}_${model}_${RUN_SUFFIX}"
    printf "%s\t%s\t%s\t%s\n" \
      "${anchor}" "${model}" "${run_tag}" "${FORECAST_ROOT}/${anchor}" >> "${expected}"
  done

  (
    for anchor in ${ANCHORS}; do
      run_tag="forecast_notest_${anchor}_${model}_${RUN_SUFFIX}"
      echo "[run] anchor=${anchor} model=${model} gpu=${gpu} run_tag=${run_tag}"
      env \
        MODELS="${model}" \
        COMPACT_ROOT="${FORECAST_ROOT}/${anchor}" \
        RUN_TAG="${run_tag}" \
        CUDA_DEVICES="${gpu}" \
        USE_MULTI_GPU=0 \
        SEQ_LEN=60 \
        LABEL_LEN=20 \
        PRED_LEN=20 \
        TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
        PATIENCE="${PATIENCE}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        QAR_SPLIT_STRATEGY=per_class_chrono \
        SAVE_EPOCH_CHECKPOINTS=0 \
        CHECKPOINTS="${CHECKPOINT_ROOT}/${run_tag}" \
        bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
    done
  ) > "${worker_log}" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
  echo "[launch] model=${model} gpu=${gpu} pid=$(cat "${pid_file}")"
done

echo "[done] launched forecast workers; expected jobs: ${expected}"
