#!/usr/bin/env bash
set -euo pipefail

# Chronological custom-condition QAR experiments.
#
# Key protocol:
#   - compact caches carry sources/time_keys
#   - dataloaders split all samples globally by time as train/val/test = 7/1/2
#   - classification uses custom multi-anchor conditions without 6->8
#   - forecasting uses four separate transition datasets:
#       predict_2_3, predict_4_5, predict_5_6, predict_8_9

STAGE="${1:-all}"

DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
IOTDB_LIB="${IOTDB_LIB:-/data2/peizhongyi/nanhang-iotdb/apache-iotdb-2.0.4-SNAPSHOT-all-bin/lib}"
TSFILE_ZIP="${TSFILE_ZIP:-datasetall_tsfile/tsfile_datasets.zip}"
CLS_ROOT="${CLS_ROOT:-./datasetall_tsfile_compact_custom_cls_chrono_20260711}"
FORECAST_ROOT="${FORECAST_ROOT:-./datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
WORK_ROOT="${WORK_ROOT:-./datasetall_tsfile_work_custom_conditions_chrono_20260711}"
MODELS_ALL="${MODELS_ALL:-Transformer TimesNet PatchTST DLinear iTransformer}"
FORECAST_ANCHORS="${FORECAST_ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"

prepare_cls() {
  "${PYTHON}" prepare_tsfile_compact_custom_conditions.py \
    --task classification \
    --zip_path "${TSFILE_ZIP}" \
    --classification_output_root "${CLS_ROOT}" \
    --work_root "${WORK_ROOT}" \
    --iotdb_lib "${IOTDB_LIB}" \
    --datasets ${DATASETS}
}

prepare_forecast() {
  "${PYTHON}" prepare_tsfile_compact_custom_conditions.py \
    --task forecast \
    --zip_path "${TSFILE_ZIP}" \
    --forecast_output_root "${FORECAST_ROOT}" \
    --work_root "${WORK_ROOT}" \
    --iotdb_lib "${IOTDB_LIB}" \
    --datasets ${DATASETS} \
    --forecast_anchors ${FORECAST_ANCHORS}
}

run_cls_core() {
  RUN_TAG="${RUN_TAG:-chrono_cls_core_20260711}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-TimesNet DLinear iTransformer}" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-1}" \
  DEVICES="${DEVICES:-0,1}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  COMPACT_ROOT="${CLS_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_patchtst() {
  RUN_TAG="${RUN_TAG:-chrono_cls_patchtst_20260711}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-PatchTST}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  CUDA_DEVICES="${CUDA_DEVICES:-6}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  COMPACT_ROOT="${CLS_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_transformer() {
  RUN_TAG="${RUN_TAG:-chrono_cls_transformer_20260711}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-Transformer}" \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  CUDA_DEVICES="${CUDA_DEVICES:-7}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-0}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  COMPACT_ROOT="${CLS_ROOT}" \
  PYTHON="${PYTHON}" \
  bash scripts/classification/run_QAR_tsfile_shiftN80.sh
}

run_cls_all() {
  run_cls_core
  run_cls_transformer
  run_cls_patchtst
}

run_forecast_one() {
  local anchor="$1"
  RUN_TAG="${RUN_TAG:-chrono_forecast_${anchor}_20260711}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS:-${MODELS_ALL}}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  CUDA_DEVICES="${CUDA_DEVICES:-4,5}" \
  USE_MULTI_GPU="${USE_MULTI_GPU:-1}" \
  DEVICES="${DEVICES:-0,1}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  COMPACT_ROOT="${FORECAST_ROOT}/${anchor}" \
  SEQ_LEN="${SEQ_LEN:-60}" \
  LABEL_LEN="${LABEL_LEN:-20}" \
  PRED_LEN="${PRED_LEN:-20}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}" \
  PATIENCE="${PATIENCE:-2}" \
  PYTHON="${PYTHON}" \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
}

run_forecast_all() {
  for anchor in ${FORECAST_ANCHORS}; do
    run_forecast_one "${anchor}"
  done
}

case "${STAGE}" in
  prepare_cls)
    prepare_cls
    ;;
  prepare_forecast)
    prepare_forecast
    ;;
  prepare_all)
    prepare_cls
    prepare_forecast
    ;;
  cls_core)
    run_cls_core
    ;;
  cls_patchtst)
    run_cls_patchtst
    ;;
  cls_transformer)
    run_cls_transformer
    ;;
  cls_attn)
    run_cls_transformer
    run_cls_patchtst
    ;;
  cls_all)
    run_cls_all
    ;;
  forecast_predict_2_3)
    run_forecast_one predict_2_3
    ;;
  forecast_predict_4_5)
    run_forecast_one predict_4_5
    ;;
  forecast_predict_5_6)
    run_forecast_one predict_5_6
    ;;
  forecast_predict_8_9)
    run_forecast_one predict_8_9
    ;;
  forecast_all)
    run_forecast_all
    ;;
  all)
    prepare_cls
    prepare_forecast
    run_cls_all
    run_forecast_all
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
