#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
PATCH_VALUES="${PATCH_VALUES:-16 8 4 2 1}"
PATCH_FORECAST_MODELS="${PATCH_FORECAST_MODELS:-PatchTST TimeXer}"
RUN_CLASSIFICATION="${RUN_CLASSIFICATION:-1}"
RUN_FORECAST="${RUN_FORECAST:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

BASE_CLS_ROOT="${BASE_CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
BASE_FORECAST_ROOT="${BASE_FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/patchlen}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"
read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\tpatch_len\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${expected}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

stride_for_patch() {
  local p="$1"
  local s=$((p / 2))
  if [ "${s}" -lt 1 ]; then
    s=1
  fi
  echo "${s}"
}

for patch_len in ${PATCH_VALUES}; do
  if [[ "${RUN_CLASSIFICATION}" != "0" ]]; then
    run_tag="patchlen${patch_len}_cls_PatchTST_${RUN_SUFFIX}"
    printf "classification\t%s\t\tPatchTST\t%s\t%s\t%s\n" "${patch_len}" "${DATASETS}" "${run_tag}" "${BASE_CLS_ROOT}" >> "${expected}"
  fi

  if [[ "${RUN_FORECAST}" != "0" ]]; then
    for anchor in ${ANCHORS}; do
      for model in ${PATCH_FORECAST_MODELS}; do
        root="${BASE_FORECAST_ROOT}/${anchor}"
        run_tag="patchlen${patch_len}_forecast_${anchor}_${model}_${RUN_SUFFIX}"
        printf "forecast\t%s\t%s\t%s\t%s\t%s\t%s\n" "${patch_len}" "${anchor}" "${model}" "${DATASETS}" "${run_tag}" "${root}" >> "${expected}"
      done
    done
  fi
done

for patch_len in ${PATCH_VALUES}; do
  stride="$(stride_for_patch "${patch_len}")"

  if [[ "${RUN_CLASSIFICATION}" != "0" ]]; then
    wait_for_slot
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    run_tag="patchlen${patch_len}_cls_PatchTST_${RUN_SUFFIX}"
    log="${LOG_ROOT}/${run_tag}.launcher.log"
    echo "[launch] classification PatchTST patch_len=${patch_len} gpu=${gpu}"
    (
      env \
        DATASETS="${DATASETS}" \
        MODELS="PatchTST" \
        COMPACT_ROOT="${BASE_CLS_ROOT}" \
        RUN_TAG="${run_tag}" \
        CUDA_DEVICES="${gpu}" \
        USE_MULTI_GPU=0 \
        PATCH_LEN="${patch_len}" \
        STRIDE="${stride}" \
        TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}" \
        PATIENCE="${PATIENCE:-4}" \
        BATCH_SIZE="${CLS_BATCH_SIZE:-96}" \
        CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}" \
        EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}" \
        QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
        CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
        bash scripts/classification/run_QAR_tsfile_shiftN80.sh
    ) > "${log}" 2>&1 < /dev/null &
  fi

  if [[ "${RUN_FORECAST}" != "0" ]]; then
    for anchor in ${ANCHORS}; do
      for model in ${PATCH_FORECAST_MODELS}; do
      wait_for_slot
      gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
      job_idx=$((job_idx + 1))
      root="${BASE_FORECAST_ROOT}/${anchor}"
      run_tag="patchlen${patch_len}_forecast_${anchor}_${model}_${RUN_SUFFIX}"
      log="${LOG_ROOT}/${run_tag}.launcher.log"
      echo "[launch] forecast ${model} anchor=${anchor} patch_len=${patch_len} gpu=${gpu}"
      (
        env \
          DATASETS="${DATASETS}" \
          MODELS="${model}" \
          COMPACT_ROOT="${root}" \
          RUN_TAG="${run_tag}" \
          CUDA_DEVICES="${gpu}" \
          USE_MULTI_GPU=0 \
          PATCH_LEN="${patch_len}" \
          STRIDE="${stride}" \
          SEQ_LEN="${SEQ_LEN:-60}" \
          LABEL_LEN="${LABEL_LEN:-20}" \
          PRED_LEN="${PRED_LEN:-20}" \
          TRAIN_EPOCHS="${FORECAST_TRAIN_EPOCHS:-20}" \
          PATIENCE="${FORECAST_PATIENCE:-3}" \
          BATCH_SIZE="${FORECAST_BATCH_SIZE:-96}" \
          QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
          CHECKPOINTS="${HOME}/qar_checkpoint_archive/checkpoints_forecast/${run_tag}" \
          bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
        ) > "${log}" 2>&1 < /dev/null &
      done
    done
  fi
done

echo "[wait] patch-length jobs are running; expected jobs: ${expected}"
wait
echo "[done] patch-length sweep"
