#!/usr/bin/env bash
set -u

# QAR one-class anomaly detection.
#
# Train/val use only normal samples (label 0). Test uses held-out normal samples
# plus all fault samples. The threshold is selected from normal validation
# reconstruction errors, so no fault/test labels are used for training or
# thresholding.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
COMPACT_ROOT="${COMPACT_ROOT:-datasetall_tsfile_compact_custom_cls_chrono_20260711}"
RUN_TAG="${RUN_TAG:-qar_anomaly_oneclass_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IS_TRAINING="${IS_TRAINING:-1}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"

SEQ_LEN="${SEQ_LEN:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-99.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${GPU:-0}}"
LOCAL_GPU="${LOCAL_GPU:-0}"

SUMMARY_DIR="${SUMMARY_DIR:-logs/anomaly_detection/${RUN_TAG}}"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_FILE="${SUMMARY_DIR}/summary.tsv"
if [ ! -f "${SUMMARY_FILE}" ]; then
  printf "dataset\tmodel\tstatus\tenc_in\troot_path\tlog_file\n" > "${SUMMARY_FILE}"
fi

cd "${PROJECT_ROOT}" || exit 1

for dataset in ${DATASETS}; do
  root_path="${COMPACT_ROOT}/${dataset}"
  cache_path="${root_path}/qar_compact_shiftN80.npz"
  if [ ! -f "${cache_path}" ]; then
    echo "[skip] ${dataset}: missing ${cache_path}"
    for model in ${MODELS}; do
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "2" "" "${root_path}" "" >> "${SUMMARY_FILE}"
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
    log_file="${SUMMARY_DIR}/${dataset}_${model}.log"
    echo "[run] dataset=${dataset} model=${model} enc_in=${enc_in} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u run.py \
      --task_name anomaly_detection \
      --is_training "${IS_TRAINING}" \
      --model_id "${RUN_TAG}_${dataset}" \
      --model "${model}" \
      --data QAR_anomaly \
      --root_path "${root_path}" \
      --data_path qar_compact_shiftN80.npz \
      --features M \
      --seq_len "${SEQ_LEN}" \
      --pred_len 0 \
      --enc_in "${enc_in}" \
      --dec_in "${enc_in}" \
      --c_out "${enc_in}" \
      --d_model "${D_MODEL}" \
      --d_ff "${D_FF}" \
      --e_layers "${E_LAYERS}" \
      --d_layers 1 \
      --n_heads "${N_HEADS}" \
      --top_k 3 \
      --batch_size "${BATCH_SIZE}" \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --learning_rate "${LEARNING_RATE}" \
      --num_workers "${NUM_WORKERS}" \
      --gpu "${LOCAL_GPU}" \
      --des oneclass_val_threshold \
      --anomaly_threshold_source val \
      --anomaly_threshold_percentile "${THRESHOLD_PERCENTILE}" \
      --anomaly_level window \
      > "${log_file}" 2>&1
    status=$?
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "${status}" "${enc_in}" "${root_path}" "${log_file}" >> "${SUMMARY_FILE}"
  done
done

echo "[done] summary: ${SUMMARY_FILE}"
