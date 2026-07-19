#!/usr/bin/env bash
set -euo pipefail

# Run the remaining QAR experiments requested on 2026-07-19.
#
# This script intentionally writes expected_jobs.tsv under
# experiment_artifacts/QAR_extra_experiments_20260717/server_logs/* so the
# existing collector can build one combined workbook.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
COLLECT_DIR="${COLLECT_DIR:-${ARTIFACT_ROOT}/collected_20260719}"

RUN_NORMAL_AUG="${RUN_NORMAL_AUG:-1}"
RUN_LEAP_AUG="${RUN_LEAP_AUG:-1}"
RUN_HISTORY80_FORECAST="${RUN_HISTORY80_FORECAST:-1}"
RUN_FORECAST_HEAD_ANOMALY="${RUN_FORECAST_HEAD_ANOMALY:-1}"
RUN_LEAP_AUG_HISTORY80="${RUN_LEAP_AUG_HISTORY80:-1}"
RUN_LEAP_AUG_FORECAST_HEAD_ANOMALY="${RUN_LEAP_AUG_FORECAST_HEAD_ANOMALY:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}" "${COLLECT_DIR}"

LEAP_AUG_DATASETS="${LEAP_AUG_DATASETS:-dataset12_aug0_2000 dataset12_aug0_4000 dataset12_aug0_6000 dataset12_aug0_10000 dataset12_aug0_20000}"
LEAP_EXTRA_ZIP="${LEAP_EXTRA_ZIP:-datasetall/data12-0类追加csv数据(1).zip}"
if [ ! -f "${LEAP_EXTRA_ZIP}" ]; then
  LEAP_EXTRA_ZIP="$(find datasetall -maxdepth 1 -name 'data12-0*csv*.zip' -print -quit 2>/dev/null || true)"
fi

if [ "${RUN_LEAP_AUG}" = "1" ]; then
  echo "[stage] dataset12 extra normal 2000/4000/6000/10000/20000 compacts"
  "${PYTHON}" scripts/data/prepare_leap_normal_augmented_compacts.py \
    --extra_zip "${LEAP_EXTRA_ZIP}" \
    --base_classification_root "datasetall_tsfile_compact_custom_cls_chrono_20260711" \
    --base_forecast_segment_root "datasetall_tsfile_compact_custom_forecast_chrono_20260711" \
    --classification_output_root "datasetall_tsfile_compact_leap_aug_cls_20260717" \
    --forecast_output_root "datasetall_tsfile_compact_leap_aug_hist80_segments_20260717" \
    --work_root "datasetall_tsfile_work_leap_aug_20260717" \
    --datasets "dataset12" \
    --counts 2000 4000 6000 10000 20000 \
    --tasks classification forecast \
    --anchors "hist80_2_3 hist80_4_5 hist80_5_6 hist80_8_9"

  echo "[stage] LEAP extra normal classification"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    COMPACT_ROOT="datasetall_tsfile_compact_leap_aug_cls_20260717" \
    DATASETS="${LEAP_AUG_DATASETS}" \
    bash scripts/experiments/launch_leap_aug_classification_20260717.sh

  echo "[stage] LEAP extra normal reconstruction anomaly"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    COMPACT_ROOT="datasetall_tsfile_compact_leap_aug_cls_20260717" \
    DATASETS="${LEAP_AUG_DATASETS}" \
    bash scripts/experiments/launch_leap_aug_anomaly_20260717.sh
fi

if [ "${RUN_NORMAL_AUG}" = "1" ]; then
  echo "[stage] normal augmentation x2/x4 classification + predict80 forecast"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    RUN_SUFFIX="${RUN_SUFFIX:-20260719}" \
    bash scripts/experiments/launch_normal_aug_experiments_20260717.sh
fi

if [ "${RUN_HISTORY80_FORECAST}" = "1" ]; then
  echo "[stage] history80 full-shot forecast"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    RUN_SUFFIX="${RUN_SUFFIX:-20260719}" \
    bash scripts/experiments/launch_history80_forecast_20260719.sh
fi

if [ "${RUN_LEAP_AUG_HISTORY80}" = "1" ]; then
  echo "[stage] LEAP augmented history80 full-shot forecast"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON="${PYTHON}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    SEGMENT_ROOT="datasetall_tsfile_compact_leap_aug_hist80_segments_20260717" \
    HISTORY_ROOT="datasetall_tsfile_compact_leap_aug_history80_20260719" \
    DATASETS="${LEAP_AUG_DATASETS}" \
    RUN_SUFFIX="${RUN_SUFFIX:-20260719_leap_aug}" \
    bash scripts/experiments/launch_history80_forecast_20260719.sh
fi

if [ "${RUN_FORECAST_HEAD_ANOMALY}" = "1" ]; then
  echo "[stage] forecast-head anomaly detection"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    RUN_SUFFIX="${RUN_SUFFIX:-20260719}" \
    bash scripts/experiments/launch_forecast_head_anomaly_20260719.sh
fi

if [ "${RUN_LEAP_AUG_FORECAST_HEAD_ANOMALY}" = "1" ]; then
  echo "[stage] LEAP augmented forecast-head anomaly detection on history80"
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    HISTORY_ROOT="datasetall_tsfile_compact_leap_aug_history80_20260719" \
    DATASETS="${LEAP_AUG_DATASETS}" \
    RUN_MODES="history80" \
    RUN_SUFFIX="${RUN_SUFFIX:-20260719_leap_aug}" \
    bash scripts/experiments/launch_forecast_head_anomaly_20260719.sh
fi

echo "[stage] collect results"
"${PYTHON}" scripts/analysis/collect_qar_experiment_results_20260717.py \
  --root "${PROJECT_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output-dir "${COLLECT_DIR}"

"${PYTHON}" scripts/analysis/build_qar_extra_matrix_excel_20260717.py \
  --input-dir "${COLLECT_DIR}" \
  --output "${COLLECT_DIR}/QAR_extra_experiments_matrix_20260719.xlsx"

"${PYTHON}" scripts/analysis/build_qar_extra_one_sheet_excel_20260719.py \
  --input-dir "${COLLECT_DIR}" \
  --output "${COLLECT_DIR}/QAR_extra_experiments_one_sheet_20260719.xlsx"

echo "[done] collected outputs:"
echo "  ${COLLECT_DIR}/QAR_extra_experiments_matrix_20260719.xlsx"
echo "  ${COLLECT_DIR}/QAR_extra_experiments_one_sheet_20260719.xlsx"
