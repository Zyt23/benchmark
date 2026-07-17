#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
SOURCE_ROOT="${SOURCE_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
CONTEXT_ROOT="${CONTEXT_ROOT:-datasetall_tsfile_compact_context40_predict23_20260717}"
HISTORY_COUNTS="${HISTORY_COUNTS:-2 3 5 8}"
MODELS="${MODELS:-Chronos2 Toto Moirai TiRex}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/foundation_context40}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

echo "[build] foundation context caches -> ${CONTEXT_ROOT}"
"${PYTHON}" scripts/long_term_forecast/build_qar_history_forecast_compacts.py \
  --source_root "${SOURCE_ROOT}" \
  --output_root "${CONTEXT_ROOT}" \
  --datasets "${DATASETS}" \
  --anchors predict_2_3 \
  --history_counts ${HISTORY_COUNTS} \
  --segment_len 80 \
  --current_context_len 40 \
  --target_len 40

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\thistory_count\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\tseq_len\tpred_len\n" > "${expected}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

batch_for_model() {
  case "$1" in
    TiRex) echo "${TIREX_BATCH_SIZE:-1}" ;;
    Moirai) echo "${MOIRAI_BATCH_SIZE:-4}" ;;
    Toto) echo "${TOTO_BATCH_SIZE:-4}" ;;
    *) echo "${ZERO_BATCH_SIZE:-8}" ;;
  esac
}

for hist in ${HISTORY_COUNTS}; do
  seq_len=$((hist * 80 + 40))
  root="${CONTEXT_ROOT}/hist${hist}/predict_2_3"
  for model in ${MODELS}; do
    wait_for_slot
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    batch_size="$(batch_for_model "${model}")"
    run_tag="context40_hist${hist}_predict_2_3_${model}_${RUN_SUFFIX}"
    log="${LOG_ROOT}/${run_tag}.launcher.log"
    printf "zero_shot_forecast\t%s\tpredict_2_3\t%s\t%s\t%s\t%s\t%s\t40\n" "${hist}" "${model}" "${DATASETS}" "${run_tag}" "${root}" "${seq_len}" >> "${expected}"
    echo "[launch] zero-shot model=${model} hist=${hist} seq_len=${seq_len} gpu=${gpu}"
    (
      env \
        DATASETS="${DATASETS}" \
        MODELS="${model}" \
        COMPACT_ROOT="${root}" \
        RUN_TAG="${run_tag}" \
        CUDA_DEVICES="${gpu}" \
        SEQ_LEN="${seq_len}" \
        LABEL_LEN=40 \
        PRED_LEN=40 \
        BATCH_SIZE="${batch_size}" \
        NUM_WORKERS="${NUM_WORKERS:-0}" \
        QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
        bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
    ) > "${log}" 2>&1 < /dev/null &
  done
done

echo "[wait] foundation context jobs are running; expected jobs: ${expected}"
wait
echo "[done] foundation context40 experiments"
