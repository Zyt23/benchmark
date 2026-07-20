#!/usr/bin/env bash
set -u

# Forecast-head anomaly detection on QAR compact caches.
#
# The model is trained with a forecasting head on normal samples only.  The
# anomaly score is the forecast error on the prediction horizon.  By default the
# threshold is selected on a validation threshold split that contains held-out
# normal samples and held-out fault samples; the test split is never used for
# threshold selection.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_custom_forecast_chrono_20260711/predict_8_9}"
RUN_TAG="${RUN_TAG:-qar_forecast_head_anomaly_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"

SEQ_LEN="${SEQ_LEN:-60}"
LABEL_LEN="${LABEL_LEN:-20}"
PRED_LEN="${PRED_LEN:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
N_HEADS="${N_HEADS:-4}"
PATCH_LEN="${PATCH_LEN:-16}"
STRIDE="${STRIDE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${GPU:-0}}"
LOCAL_GPU="${LOCAL_GPU:-0}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-val_mixed_best_f1}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-99.0}"
FORECAST_ANOMALY_SCORE="${FORECAST_ANOMALY_SCORE:-auto}"

CHECKPOINTS="${CHECKPOINTS:-${HOME}/qar_checkpoint_archive/checkpoints_forecast_anomaly/${RUN_TAG}}"
SUMMARY_DIR="${SUMMARY_DIR:-logs/forecast_anomaly_detection/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-${SUMMARY_DIR}/summary.tsv}"

mkdir -p "${SUMMARY_DIR}" "${CHECKPOINTS}"
cd "${PROJECT_ROOT}" || exit 1

if [ ! -f "${SUMMARY_FILE}" ]; then
  printf "dataset\tmodel\tstatus\tenc_in\troot_path\tlog_file\tresult_dir\n" > "${SUMMARY_FILE}"
fi

for dataset in ${DATASETS}; do
  root_path="${COMPACT_ROOT}/${dataset}"
  cache_path="${root_path}/qar_compact_shiftN80.npz"
  if [ ! -f "${cache_path}" ]; then
    echo "[skip] ${dataset}: missing ${cache_path}"
    for model in ${MODELS}; do
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "2" "" "${root_path}" "" "" >> "${SUMMARY_FILE}"
    done
    continue
  fi

  enc_in=$("${PYTHON_BIN}" - <<PY
import numpy as np
c = np.load(r"${cache_path}", allow_pickle=False)
print(int(c["x"].shape[2]))
PY
)

  for model in ${MODELS}; do
    des="${RUN_TAG}_${dataset}_${model}"
    model_id="${dataset}_QAR_forecast_head_anomaly"
    log_file="${SUMMARY_DIR}/${dataset}_${model}.log"
    echo "[run] dataset=${dataset} model=${model} enc_in=${enc_in} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u run.py \
      --task_name forecast_anomaly_detection \
      --is_training 1 \
      --model_id "${model_id}" \
      --model "${model}" \
      --data QAR_forecast_anomaly \
      --root_path "${root_path}" \
      --data_path qar_compact_shiftN80.npz \
      --features M \
      --target var_0 \
      --freq h \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${PRED_LEN}" \
      --enc_in "${enc_in}" \
      --dec_in "${enc_in}" \
      --c_out "${enc_in}" \
      --d_model "${D_MODEL}" \
      --d_ff "${D_FF}" \
      --e_layers "${E_LAYERS}" \
      --d_layers "${D_LAYERS}" \
      --n_heads "${N_HEADS}" \
      --top_k 5 \
      --num_kernels 6 \
      --patch_len "${PATCH_LEN}" \
      --stride "${STRIDE}" \
      --batch_size "${BATCH_SIZE}" \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --learning_rate "${LEARNING_RATE}" \
      --num_workers "${NUM_WORKERS}" \
      --gpu "${LOCAL_GPU}" \
      --des "${des}" \
      --itr 1 \
      --checkpoints "${CHECKPOINTS}" \
      --anomaly_threshold_source "${THRESHOLD_SOURCE}" \
      --anomaly_threshold_percentile "${THRESHOLD_PERCENTILE}" \
      --forecast_anomaly_score "${FORECAST_ANOMALY_SCORE}" \
      > "${log_file}" 2>&1
    status=$?
    result_dir="$(find -L ./results -maxdepth 1 -type d -name "*${des}_0" -print -quit 2>/dev/null || true)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "${status}" "${enc_in}" "${root_path}" "${log_file}" "${result_dir}" >> "${SUMMARY_FILE}"
  done
done

echo "[done] summary: ${SUMMARY_FILE}"
