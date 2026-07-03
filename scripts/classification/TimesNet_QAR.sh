#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
PYTHON=/home/zhanghuan/miniconda3/envs/py10_cu128/bin/python

# Transformer 二分类 QAR (seq_len=2000, 两段拼接)
# 遍历 phase_a_shift: 5, -5, 10, -10, 15, -15, 30, -30, 50, -50, 80, -80, 100, -100

for shift in 5 -5 10 -10 15 -15 30 -30 50 -50 80 -80 100 -100; do
  echo "=========================================="
  echo "Running with --phase_a_shift ${shift}"
  echo "=========================================="
  $PYTHON -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset6/ \
    --model_id QAR_PRSOVTransformer \
    --model Transformer \
    --data QAR_shift \
    --phase_a_shift ${shift} \
    --seq_len 2000 \
    --e_layers 3 \
    --batch_size 128 \
    --d_model 64 \
    --d_ff 128 \
    --top_k 5 \
    --num_kernels 6 \
    --dropout 0.2 \
    --lradj cosine \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --patience 3 \
    --use_multi_gpu \
    --devices 0,1
done
