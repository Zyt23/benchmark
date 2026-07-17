#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
SOURCE_ROOT="${SOURCE_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
UNIVARIATE_ROOT="${UNIVARIATE_ROOT:-datasetall_tsfile_compact_univariate_forecast_20260717}"
TARGET_ALIAS="${TARGET_ALIAS:-manifold_pressure}"
MODELS="${MODELS:-Sundial}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/univariate_foundation}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

for anchor in ${ANCHORS}; do
  echo "[build] univariate ${anchor} target=${TARGET_ALIAS}"
  "${PYTHON}" scripts/data/project_qar_compact_univariate.py \
    --base_root "${SOURCE_ROOT}/${anchor}" \
    --output_root "${UNIVARIATE_ROOT}/${anchor}" \
    --datasets "${DATASETS}" \
    --target "${TARGET_ALIAS}"
done

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\ttarget\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\tseq_len\tpred_len\n" > "${expected}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

for anchor in ${ANCHORS}; do
  root="${UNIVARIATE_ROOT}/${anchor}"
  for model in ${MODELS}; do
    run_tag="univariate_${TARGET_ALIAS}_${anchor}_${model}_${RUN_SUFFIX}"
    printf "zero_shot_forecast\t%s\t%s\t%s\t%s\t%s\t%s\t60\t20\n" "${TARGET_ALIAS}" "${anchor}" "${model}" "${DATASETS}" "${run_tag}" "${root}" >> "${expected}"
  done
done

for anchor in ${ANCHORS}; do
  root="${UNIVARIATE_ROOT}/${anchor}"
  for model in ${MODELS}; do
    wait_for_slot
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    run_tag="univariate_${TARGET_ALIAS}_${anchor}_${model}_${RUN_SUFFIX}"
    log="${LOG_ROOT}/${run_tag}.launcher.log"
    echo "[launch] univariate zero-shot model=${model} anchor=${anchor} gpu=${gpu}"
    (
      env \
        DATASETS="${DATASETS}" \
        MODELS="${model}" \
        COMPACT_ROOT="${root}" \
        RUN_TAG="${run_tag}" \
        CUDA_DEVICES="${gpu}" \
        SEQ_LEN="${SEQ_LEN:-60}" \
        LABEL_LEN="${LABEL_LEN:-20}" \
        PRED_LEN="${PRED_LEN:-20}" \
        BATCH_SIZE="${BATCH_SIZE:-8}" \
        NUM_WORKERS="${NUM_WORKERS:-0}" \
        QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
        bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
    ) > "${log}" 2>&1 < /dev/null &
  done
done

echo "[wait] univariate jobs are running; expected jobs: ${expected}"
wait
echo "[done] univariate foundation experiments"
