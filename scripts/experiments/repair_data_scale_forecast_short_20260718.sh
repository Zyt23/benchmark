#!/usr/bin/env bash
set -euo pipefail

# Repair launcher for the 20260717 data-scale forecasting sweep.
#
# The first data-scale forecast run used very long RUN_TAG/checkpoint names.
# On Linux this overflowed the file-name limit when TimesNet/Informer-style
# experiment setting strings were appended to the checkpoint directory.  This
# launcher keeps the same compact caches and expected matrix, but uses short
# run tags and short checkpoint roots.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
VARIANTS="${VARIANTS:-both_keep50 both_keep25}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

SCALE_FORECAST_ROOT="${SCALE_FORECAST_ROOT:-datasetall_tsfile_compact_scale_forecast_20260717}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/data_scale_forecast_short}"
RUN_SUFFIX="${RUN_SUFFIX:-20260718}"
FILTER_OLD_EXPECTED="${FILTER_OLD_EXPECTED:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

if [[ "${FILTER_OLD_EXPECTED}" == "1" ]]; then
  echo "[repair] keep only classification rows in old data-scale expected_jobs.tsv"
  "${PYTHON}" - <<'PY'
from pathlib import Path

groups = [
    "data_scale_final_dlinear_b96",
    "data_scale_final_itransformer_b96",
    "data_scale_final_ptb16",
    "data_scale_final_tfb2",
    "data_scale_final_timesnet_b96",
]
root = Path("experiment_artifacts/QAR_extra_experiments_20260717/server_logs")
for group in groups:
    path = root / group / "expected_jobs.tsv"
    if not path.exists():
        continue
    backup = path.with_name("expected_jobs.with_failed_forecast.tsv")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        continue
    header, rows = lines[0], lines[1:]
    kept = [line for line in rows if line.split("\t", 1)[0] == "classification"]
    path.write_text("\n".join([header, *kept]) + "\n", encoding="utf-8")
    print(f"{path}: kept {len(kept)} classification rows, backup={backup}")
PY
fi

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

variant_short() {
  case "$1" in
    both_keep50) echo "b50" ;;
    both_keep25) echo "b25" ;;
    *) echo "$1" ;;
  esac
}

anchor_short() {
  case "$1" in
    predict_2_3) echo "p23" ;;
    predict_4_5) echo "p45" ;;
    predict_5_6) echo "p56" ;;
    predict_8_9) echo "p89" ;;
    *) echo "$1" ;;
  esac
}

model_short() {
  case "$1" in
    Transformer) echo "TF" ;;
    TimesNet) echo "TN" ;;
    PatchTST) echo "PT" ;;
    DLinear) echo "DL" ;;
    iTransformer) echo "IT" ;;
    *) echo "$1" ;;
  esac
}

for variant in ${VARIANTS}; do
  datasets_variant="$(dataset_variant_list "${variant}")"
  for anchor in ${ANCHORS}; do
    root="${SCALE_FORECAST_ROOT}/${anchor}"
    for model in ${MODELS}; do
      run_tag="sf_$(variant_short "${variant}")_$(anchor_short "${anchor}")_$(model_short "${model}")_${RUN_SUFFIX}"
      printf "forecast\t%s\t%s\t%s\t%s\t%s\t%s\n" "${variant}" "${anchor}" "${model}" "${datasets_variant}" "${run_tag}" "${root}" >> "${expected}"
    done
  done
done

launch_forecast() {
  local variant="$1"
  local anchor="$2"
  local model="$3"
  local datasets_variant="$4"
  local gpu="$5"
  local root="${SCALE_FORECAST_ROOT}/${anchor}"
  local run_tag="sf_$(variant_short "${variant}")_$(anchor_short "${anchor}")_$(model_short "${model}")_${RUN_SUFFIX}"
  local log="${LOG_ROOT}/${run_tag}.launcher.log"
  echo "[launch] forecast variant=${variant} anchor=${anchor} model=${model} gpu=${gpu} run_tag=${run_tag}"
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
      CHECKPOINTS="${HOME}/qar_checkpoint_archive/cf/${run_tag}" \
      bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
  ) > "${log}" 2>&1 < /dev/null &
}

for variant in ${VARIANTS}; do
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

echo "[wait] short data-scale forecast repair jobs are running; expected jobs: ${expected}"
wait
echo "[done] short data-scale forecast repair"
