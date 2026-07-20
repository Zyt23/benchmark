#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for forecast-head anomaly detection.
#
# Modes:
#   predict80:
#     COMPACT_ROOT_BASE/<predict_2_3|predict_4_5|predict_5_6|predict_8_9>
#     seq_len=60 label_len=20 pred_len=20
#   history80:
#     HISTORY_ROOT/histK/<hist80_2_3|hist80_4_5|hist80_5_6|hist80_8_9>
#     seq_len=K*80 label_len=80 pred_len=80

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
COMPACT_ROOT_BASE="${COMPACT_ROOT_BASE:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
HISTORY_ROOT="${HISTORY_ROOT:-datasetall_tsfile_compact_history80_20260719}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/forecast_head_anomaly_20260719}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
PREDICT_ANCHORS="${PREDICT_ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
HISTORY_ANCHORS="${HISTORY_ANCHORS:-hist80_2_3 hist80_4_5 hist80_5_6 hist80_8_9}"
HISTORY_COUNTS="${HISTORY_COUNTS:-1 4 8 12 16}"
RUN_MODES="${RUN_MODES:-predict80 history80}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
RUN_SUFFIX="${RUN_SUFFIX:-20260719}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\tvariant\thistory_count\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\tseq_len\tpred_len\n" > "${expected}"

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

launch_one() {
  local variant="$1"
  local history_count="$2"
  local anchor="$3"
  local model="$4"
  local root="$5"
  local seq_len="$6"
  local label_len="$7"
  local pred_len="$8"
  local run_tag="$9"

  printf "forecast_anomaly_detection\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" "${history_count}" "${anchor}" "${model}" "${DATASETS}" \
    "${run_tag}" "${root}" "${seq_len}" "${pred_len}" >> "${expected}"

  wait_for_slot
  local gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
  job_idx=$((job_idx + 1))
  local log="${LOG_ROOT}/${run_tag}.launcher.log"
  echo "[launch] variant=${variant} hist=${history_count} anchor=${anchor} model=${model} gpu=${gpu}"
  (
    env \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      COMPACT_ROOT="${root}" \
      RUN_TAG="${run_tag}" \
      DATASETS="${DATASETS}" \
      MODELS="${model}" \
      PYTHON_BIN="${PYTHON_BIN:-/home/para/anaconda3/bin/python}" \
      CUDA_DEVICE="${gpu}" \
      LOCAL_GPU=0 \
      SEQ_LEN="${seq_len}" \
      LABEL_LEN="${label_len}" \
      PRED_LEN="${pred_len}" \
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}" \
      PATIENCE="${PATIENCE:-3}" \
      BATCH_SIZE="${BATCH_SIZE:-64}" \
      THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-val_mixed_best_f1}" \
      THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-99.0}" \
      FORECAST_ANOMALY_SCORE="${FORECAST_ANOMALY_SCORE:-auto}" \
      bash scripts/anomaly_detection/run_QAR_forecast_head_anomaly.sh
  ) > "${log}" 2>&1 < /dev/null &
}

if [[ " ${RUN_MODES} " == *" predict80 "* ]]; then
  for anchor in ${PREDICT_ANCHORS}; do
    for model in ${MODELS}; do
      root="${COMPACT_ROOT_BASE}/${anchor}"
      run_tag="forecast_head_anomaly_${anchor}_${model}_${RUN_SUFFIX}"
      launch_one "predict80" "" "${anchor}" "${model}" "${root}" 60 20 20 "${run_tag}"
    done
  done
fi

if [[ " ${RUN_MODES} " == *" history80 "* ]]; then
  for history_count in ${HISTORY_COUNTS}; do
    seq_len=$((history_count * 80))
    for anchor in ${HISTORY_ANCHORS}; do
      for model in ${MODELS}; do
        root="${HISTORY_ROOT}/hist${history_count}/${anchor}"
        run_tag="forecast_head_anomaly_hist${history_count}_${anchor}_${model}_${RUN_SUFFIX}"
        launch_one "history80" "${history_count}" "${anchor}" "${model}" "${root}" "${seq_len}" 80 80 "${run_tag}"
      done
    done
  done
fi

echo "[wait] forecast-head anomaly jobs are running; expected jobs: ${expected}"
wait
echo "[done] forecast-head anomaly jobs"
