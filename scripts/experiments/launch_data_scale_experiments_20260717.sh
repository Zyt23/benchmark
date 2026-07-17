#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

BASE_CLS_ROOT="${BASE_CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
BASE_FORECAST_ROOT="${BASE_FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
SCALE_CLS_ROOT="${SCALE_CLS_ROOT:-datasetall_tsfile_compact_scale_cls_20260717}"
SCALE_FORECAST_ROOT="${SCALE_FORECAST_ROOT:-datasetall_tsfile_compact_scale_forecast_20260717}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/data_scale}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

echo "[build] classification subset compacts -> ${SCALE_CLS_ROOT}"
"${PYTHON}" scripts/data/build_qar_compact_subsets.py \
  --base_root "${BASE_CLS_ROOT}" \
  --output_root "${SCALE_CLS_ROOT}" \
  --datasets "${DATASETS}" \
  --variants both_keep50:0.5:0.5 both_keep25:0.25:0.25 normal_keep50:0.5:1.0 normal_keep25:0.25:1.0 \
  --skip_existing

for anchor in ${ANCHORS}; do
  echo "[build] forecast subset compacts ${anchor} -> ${SCALE_FORECAST_ROOT}/${anchor}"
  "${PYTHON}" scripts/data/build_qar_compact_subsets.py \
    --base_root "${BASE_FORECAST_ROOT}/${anchor}" \
    --output_root "${SCALE_FORECAST_ROOT}/${anchor}" \
    --datasets "${DATASETS}" \
    --variants both_keep50:0.5:0.5 both_keep25:0.25:0.25 \
    --skip_existing
done

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${expected}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

dataset_variant_list() {
  local variant="$1"
  local out=""
  for dataset in ${DATASETS}; do
    out="${out} ${dataset}_${variant}"
  done
  echo "${out# }"
}

launch_cls() {
  local variant="$1"
  local model="$2"
  local datasets_variant="$3"
  local gpu="$4"
  local run_tag="scale_${variant}_cls_${model}_20260717"
  local log="${LOG_ROOT}/${run_tag}.launcher.log"
  printf "classification\t%s\t\t%s\t%s\t%s\t%s\n" "${variant}" "${model}" "${datasets_variant}" "${run_tag}" "${SCALE_CLS_ROOT}" >> "${expected}"
  echo "[launch] cls variant=${variant} model=${model} gpu=${gpu}"
  (
    env \
      DATASETS="${datasets_variant}" \
      MODELS="${model}" \
      COMPACT_ROOT="${SCALE_CLS_ROOT}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${gpu}" \
      USE_MULTI_GPU=0 \
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}" \
      PATIENCE="${PATIENCE:-4}" \
      BATCH_SIZE="${CLS_BATCH_SIZE:-96}" \
      CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}" \
      EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}" \
      QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
      CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
      bash scripts/classification/run_QAR_tsfile_shiftN80.sh
  ) > "${log}" 2>&1 < /dev/null &
}

launch_forecast() {
  local variant="$1"
  local anchor="$2"
  local model="$3"
  local datasets_variant="$4"
  local gpu="$5"
  local root="${SCALE_FORECAST_ROOT}/${anchor}"
  local run_tag="scale_${variant}_forecast_${anchor}_${model}_20260717"
  local log="${LOG_ROOT}/${run_tag}.launcher.log"
  printf "forecast\t%s\t%s\t%s\t%s\t%s\t%s\n" "${variant}" "${anchor}" "${model}" "${datasets_variant}" "${run_tag}" "${root}" >> "${expected}"
  echo "[launch] forecast variant=${variant} anchor=${anchor} model=${model} gpu=${gpu}"
  (
    env \
      DATASETS="${datasets_variant}" \
      MODELS="${model}" \
      COMPACT_ROOT="${root}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${gpu}" \
      USE_MULTI_GPU=0 \
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
}

for variant in both_keep50 both_keep25 normal_keep50 normal_keep25; do
  datasets_variant="$(dataset_variant_list "${variant}")"
  for model in ${MODELS}; do
    wait_for_slot
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    launch_cls "${variant}" "${model}" "${datasets_variant}" "${gpu}"
  done
done

for variant in both_keep50 both_keep25; do
  datasets_variant="$(dataset_variant_list "${variant}")"
  for anchor in ${ANCHORS}; do
    for model in ${MODELS}; do
      wait_for_slot
      gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
      job_idx=$((job_idx + 1))
      launch_forecast "${variant}" "${anchor}" "${model}" "${datasets_variant}" "${gpu}"
    done
  done
done

echo "[wait] data-scale jobs are running; expected jobs: ${expected}"
wait
echo "[done] data-scale experiments"
