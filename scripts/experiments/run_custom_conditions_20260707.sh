#!/usr/bin/env bash
set -euo pipefail

# Batch runner for the July-2026 custom-condition QAR experiments.
#
# Stages:
#   cls_core        classification: TimesNet/DLinear/iTransformer on anchor cache
#   cls_patchtst    classification: PatchTST on anchor cache, small batch
#   cls_transformer classification: Transformer on anchor cache, very small batch
#   cls_attn        compatibility alias: cls_transformer + cls_patchtst
#   forecast_anchor long-term forecast on anchor cache, segment sliding windows
#   forecast_phase80 long-term forecast on phase-start80 cache
#   all             run all stages sequentially

STAGE="${1:-all}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
ANCHOR_ROOT="${ANCHOR_ROOT:-./datasetall_tsfile_compact_anchor_20260707}"
PHASE80_ROOT="${PHASE80_ROOT:-./datasetall_tsfile_compact_phase80_20260707}"

run_cls_core() {
  RUN_TAG="${RUN_TAG:-anchor_cls_core_20260707}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-TimesNet DLinear iTransformer}" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-1}" \
  DEVICES="${DEVICES:-0,1}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  COMPACT_ROOT="${ANCHOR_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_patchtst() {
  RUN_TAG="${RUN_TAG:-anchor_cls_patchtst_20260707}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-PatchTST}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  CUDA_DEVICES="${CUDA_DEVICES:-6}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  COMPACT_ROOT="${ANCHOR_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_transformer() {
  RUN_TAG="${RUN_TAG:-anchor_cls_transformer_20260707}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-Transformer}" \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  CUDA_DEVICES="${CUDA_DEVICES:-7}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  COMPACT_ROOT="${ANCHOR_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_attn() {
  run_cls_transformer
  run_cls_patchtst
}

run_forecast_anchor() {
  RUN_TAG="${RUN_TAG:-anchor_forecast_segment_20260707}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-1}" \
  DEVICES="${DEVICES:-0,1}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  COMPACT_ROOT="${ANCHOR_ROOT}" \
  FORECAST_WINDOW_MODE="${FORECAST_WINDOW_MODE:-segment}" \
  FORECAST_STRIDE="${FORECAST_STRIDE:-80}" \
  PYTHON="${PYTHON}" \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
}

run_forecast_phase80() {
  RUN_TAG="${RUN_TAG:-phase80_forecast_20260707}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  CUDA_DEVICES="${CUDA_DEVICES:-6}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  COMPACT_ROOT="${PHASE80_ROOT}" \
  FORECAST_WINDOW_MODE="${FORECAST_WINDOW_MODE:-segment}" \
  FORECAST_STRIDE="${FORECAST_STRIDE:-80}" \
  PYTHON="${PYTHON}" \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
}

case "${STAGE}" in
  cls_core)
    run_cls_core
    ;;
  cls_attn)
    run_cls_attn
    ;;
  cls_patchtst)
    run_cls_patchtst
    ;;
  cls_transformer)
    run_cls_transformer
    ;;
  forecast_anchor)
    run_forecast_anchor
    ;;
  forecast_phase80)
    run_forecast_phase80
    ;;
  all)
    run_cls_core
    run_cls_transformer
    run_cls_patchtst
    run_forecast_anchor
    run_forecast_phase80
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
