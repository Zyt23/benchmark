#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"

MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

BASE_CLS_ROOT="${BASE_CLS_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
BASE_FORECAST_ROOT="${BASE_FORECAST_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
NORMAL_AUG_CLS_ROOT="${NORMAL_AUG_CLS_ROOT:-datasetall_tsfile_compact_normal_aug_cls_20260717}"
NORMAL_AUG_FORECAST_ROOT="${NORMAL_AUG_FORECAST_ROOT:-datasetall_tsfile_compact_normal_aug_forecast_20260717}"
WORK_ROOT="${WORK_ROOT:-datasetall_tsfile_work_normal_aug_20260717}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/normal_aug}"

LEAP_EXTRA_ZIP="${LEAP_EXTRA_ZIP:-datasetall/data12-0类追加csv数据(1).zip}"
A320_EXTRA_ZIP="${A320_EXTRA_ZIP:-datasetall/320-HPV-正常-追加567.zip}"
V2527_EXTRA_ZIP="${V2527_EXTRA_ZIP:-datasetall/320-V2527-HPV-正常-追加.zip}"
SOV_EXTRA_ZIP="${SOV_EXTRA_ZIP:-datasetall/SOV_normal_5600_hdf5.zip}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

if [ ! -f "${LEAP_EXTRA_ZIP}" ]; then
  LEAP_EXTRA_ZIP="$(find datasetall -maxdepth 1 -name 'data12-0*csv*.zip' -print -quit 2>/dev/null || true)"
fi
if [ ! -f "${A320_EXTRA_ZIP}" ]; then
  A320_EXTRA_ZIP="$(find datasetall -maxdepth 1 -name '320-HPV-*567.zip' -print -quit 2>/dev/null || true)"
fi
if [ ! -f "${V2527_EXTRA_ZIP}" ]; then
  V2527_EXTRA_ZIP="$(find datasetall -maxdepth 1 -name '320-V2527-HPV-*.zip' -print -quit 2>/dev/null || true)"
fi
if [ ! -f "${SOV_EXTRA_ZIP}" ]; then
  SOV_EXTRA_ZIP="$(find datasetall -maxdepth 1 -name 'SOV_normal_5600_hdf5.zip' -print -quit 2>/dev/null || true)"
fi

specs=()
aug_datasets=()
if [ -f "${LEAP_EXTRA_ZIP}" ]; then
  specs+=("${LEAP_EXTRA_ZIP}=dataset9,dataset10,dataset12")
  for dataset in dataset9 dataset10 dataset12; do
    aug_datasets+=("${dataset}_normalx2" "${dataset}_normalx4")
  done
else
  echo "[warn] missing LEAP extra zip: ${LEAP_EXTRA_ZIP}"
fi
if [ -f "${A320_EXTRA_ZIP}" ]; then
  specs+=("${A320_EXTRA_ZIP}=dataset5,dataset6,dataset7")
  for dataset in dataset5 dataset6 dataset7; do
    aug_datasets+=("${dataset}_normalx2" "${dataset}_normalx4")
  done
else
  echo "[warn] missing A320 extra zip: ${A320_EXTRA_ZIP}"
fi
if [ -f "${V2527_EXTRA_ZIP}" ]; then
  specs+=("${V2527_EXTRA_ZIP}=dataset8,dataset8-1")
  for dataset in dataset8 dataset8-1; do
    aug_datasets+=("${dataset}_normalx2" "${dataset}_normalx4")
  done
else
  echo "[warn] missing V2527 extra zip: ${V2527_EXTRA_ZIP}"
fi
if [ -f "${SOV_EXTRA_ZIP}" ]; then
  specs+=("${SOV_EXTRA_ZIP}=dataset14")
  for dataset in dataset14; do
    aug_datasets+=("${dataset}_normalx2" "${dataset}_normalx4")
  done
else
  echo "[warn] missing SOV extra zip: ${SOV_EXTRA_ZIP}"
fi

if [ "${#specs[@]}" -eq 0 ]; then
  echo "[error] no extra-normal zip is available; nothing to run" >&2
  exit 1
fi

echo "[build] normal augmentation compacts"
"${PYTHON}" scripts/data/prepare_normal_extra_augmented_compacts.py \
  --extra_specs "${specs[@]}" \
  --base_classification_root "${BASE_CLS_ROOT}" \
  --base_forecast_segment_root "${BASE_FORECAST_ROOT}" \
  --classification_output_root "${NORMAL_AUG_CLS_ROOT}" \
  --forecast_output_root "${NORMAL_AUG_FORECAST_ROOT}" \
  --work_root "${WORK_ROOT}" \
  --factors 2 4 \
  --tasks classification forecast \
  --anchors "${ANCHORS}"

DATASETS_AUG="${aug_datasets[*]}"
read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${expected}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

for model in ${MODELS}; do
  run_tag="normalx_cls_${model}_${RUN_SUFFIX}"
  printf "classification\tnormalx\t\t%s\t%s\t%s\t%s\n" "${model}" "${DATASETS_AUG}" "${run_tag}" "${NORMAL_AUG_CLS_ROOT}" >> "${expected}"
done

for anchor in ${ANCHORS}; do
  for model in ${MODELS}; do
    root="${NORMAL_AUG_FORECAST_ROOT}/${anchor}"
    run_tag="normalx_forecast_${anchor}_${model}_${RUN_SUFFIX}"
    printf "forecast\tnormalx\t%s\t%s\t%s\t%s\t%s\n" "${anchor}" "${model}" "${DATASETS_AUG}" "${run_tag}" "${root}" >> "${expected}"
  done
done

for model in ${MODELS}; do
  wait_for_slot
  gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
  job_idx=$((job_idx + 1))
  run_tag="normalx_cls_${model}_${RUN_SUFFIX}"
  log="${LOG_ROOT}/${run_tag}.launcher.log"
  echo "[launch] cls normalx model=${model} gpu=${gpu}"
  (
    env \
      DATASETS="${DATASETS_AUG}" \
      MODELS="${model}" \
      COMPACT_ROOT="${NORMAL_AUG_CLS_ROOT}" \
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
done

for anchor in ${ANCHORS}; do
  for model in ${MODELS}; do
    wait_for_slot
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    root="${NORMAL_AUG_FORECAST_ROOT}/${anchor}"
    run_tag="normalx_forecast_${anchor}_${model}_${RUN_SUFFIX}"
    log="${LOG_ROOT}/${run_tag}.launcher.log"
    echo "[launch] forecast normalx anchor=${anchor} model=${model} gpu=${gpu}"
    (
      env \
        DATASETS="${DATASETS_AUG}" \
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
  done
done

echo "[wait] normal-augmentation jobs are running; expected jobs: ${expected}"
wait
echo "[done] normal-augmentation experiments"
