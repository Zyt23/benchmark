#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
HISTORY_ROOT="${HISTORY_ROOT:-datasetall_tsfile_compact_history80_20260717}"
LOG_ROOT="${LOG_ROOT:-experiment_artifacts/QAR_history80_zeroshot_20260717/server_logs}"
DATASETS_ALL="${DATASETS_ALL:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
ANCHORS="${ANCHORS:-hist80_2_3 hist80_4_5 hist80_5_6 hist80_8_9}"
HISTORY_COUNTS="${HISTORY_COUNTS:-1 4 8 12 16}"
MODELS="${MODELS:-Chronos2 TiRex}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-history80_zeroshot}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

read -r -a gpus <<< "${GPU_LIST}"
job_idx=0
for history_count in ${HISTORY_COUNTS}; do
  seq_len=$((history_count * 80))
  for anchor in ${ANCHORS}; do
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    run_tag="${RUN_TAG_PREFIX}_hist${history_count}_${anchor}_20260717"
    log="${LOG_ROOT}/${run_tag}.log"
    pid_file="${LOG_ROOT}/${run_tag}.pid"
    echo "[launch] hist=${history_count} anchor=${anchor} models=${MODELS} gpu=${gpu} -> ${log}"
    (
      env \
        RUN_TAG="${run_tag}" \
        DATASETS="${DATASETS_ALL}" \
        MODELS="${MODELS}" \
        COMPACT_ROOT="${HISTORY_ROOT}/hist${history_count}/${anchor}" \
        QAR_SPLIT_STRATEGY=per_class_chrono \
        PYTHON="${PYTHON}" \
        CUDA_DEVICES="${gpu}" \
        BATCH_SIZE="${ZERO_BATCH_SIZE:-4}" \
        NUM_WORKERS="${NUM_WORKERS:-0}" \
        SEQ_LEN="${seq_len}" \
        LABEL_LEN=80 \
        PRED_LEN=80 \
        CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-external_models/chronos-2}" \
        TIREX_MODEL_PATH="${TIREX_MODEL_PATH:-NX-AI/TiRex}" \
        TIREX_BACKEND="${TIREX_BACKEND:-torch}" \
        MOIRAI_MODEL_PATH="${MOIRAI_MODEL_PATH:-external_models/moirai-2.0-R-small}" \
        HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
        bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
    ) > "${log}" 2>&1 < /dev/null &
    echo $! > "${pid_file}"
  done
done

echo "[done] launched history80 zero-shot jobs"
