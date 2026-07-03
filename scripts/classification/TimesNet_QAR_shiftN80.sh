#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
MODEL="${MODEL:-TimesNet}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-3}"
ROOT_PATH="${ROOT_PATH:-./dataset6/}"
MODEL_ID="${MODEL_ID:-QAR_PRSOVTransformer}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICES="${DEVICES:-0,1}"
USE_MULTI_GPU="${USE_MULTI_GPU:-1}"
DATA="${DATA:-QAR_shift}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

# 单次运行: phase_a_shift = -80 (此前最优 acc=0.7617)
# 使用唯一 --des 标记，避免覆盖之前的 checkpoint 目录
# 训练过程中将逐 epoch 保存所有 checkpoint (修改后的 exp_classification.py)

DES_TAG="${MODEL}_shiftN80_keepall_$(date +%Y%m%d_%H%M%S)"
echo "Des tag: ${DES_TAG}"
echo "Model: ${MODEL}"
echo "Python: ${PYTHON}"
echo "Root path: ${ROOT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

GPU_ARGS=()
if [[ "${USE_MULTI_GPU}" != "0" ]]; then
  GPU_ARGS+=(--use_multi_gpu --devices "${DEVICES}")
fi

$PYTHON -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path "${ROOT_PATH}" \
  --model_id "${MODEL_ID}" \
  --model "${MODEL}" \
  --data "${DATA}" \
  --phase_a_shift -80 \
  --seq_len 2000 \
  --e_layers 3 \
  --batch_size "${BATCH_SIZE}" \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --num_kernels 6 \
  --dropout 0.2 \
  --lradj cosine \
  --des "${DES_TAG}" \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs "${TRAIN_EPOCHS}" \
  --patience "${PATIENCE}" \
  "${GPU_ARGS[@]}"
