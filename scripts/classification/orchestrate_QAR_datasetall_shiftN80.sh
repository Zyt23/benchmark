#!/usr/bin/env bash
set -euo pipefail

# Parallel wrapper around run_QAR_datasetall_shiftN80.sh.
#
# Default GPU plan for an 8-GPU machine:
#   wave 1: dataset5 -> 0,1; dataset7 -> 2,3; dataset8 -> 4,5; dataset9 -> 6,7
#   wave 2: dataset10 -> 4,5
#
# Override WORKER_SPECS / TAIL_SPECS if the server is busy. Spec format:
#   dataset:CUDA_VISIBLE_DEVICES:run.py_devices

RUN_TAG="${RUN_TAG:-datasetall_shiftN80_$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-0}"
COMPACT_ROOT="${COMPACT_ROOT:-./datasetall_compact}"
CHECKPOINTS="${CHECKPOINTS:-./checkpoints_datasetall/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./logs/datasetall/${RUN_TAG}}"
WORKER_SPECS="${WORKER_SPECS:-dataset5:0,1:0,1 dataset7:2,3:0,1 dataset8:4,5:0,1 dataset9:6,7:0,1}"
TAIL_SPECS="${TAIL_SPECS:-dataset10:4,5:0,1}"

mkdir -p "${LOG_DIR}" "${CHECKPOINTS}"

run_spec() {
  local spec="$1"
  local dataset cuda_devices devices worker_log summary_file
  IFS=':' read -r dataset cuda_devices devices <<< "${spec}"
  worker_log="${LOG_DIR}/worker_${dataset}.out"
  summary_file="${LOG_DIR}/summary_${dataset}.tsv"

  echo "[$(date '+%F %T')] worker start dataset=${dataset} cuda=${cuda_devices} devices=${devices}"
  RUN_TAG="${RUN_TAG}" \
  DATASETS="${dataset}" \
  MODELS="${MODELS}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  CUDA_DEVICES="${cuda_devices}" \
  DEVICES="${devices}" \
  SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS}" \
  COMPACT_ROOT="${COMPACT_ROOT}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  LOG_DIR="${LOG_DIR}" \
  SUMMARY_FILE="${summary_file}" \
    bash scripts/classification/run_QAR_datasetall_shiftN80.sh > "${worker_log}" 2>&1
  echo "[$(date '+%F %T')] worker done dataset=${dataset}; log=${worker_log}"
}

echo "Run tag: ${RUN_TAG}"
echo "Models: ${MODELS}"
echo "Wave-1 specs: ${WORKER_SPECS}"
echo "Wave-2 specs: ${TAIL_SPECS}"
echo "Logs: ${LOG_DIR}"

status=0
pids=()
for spec in ${WORKER_SPECS}; do
  run_spec "${spec}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

for spec in ${TAIL_SPECS}; do
  if ! run_spec "${spec}"; then
    status=1
  fi
done

{
  printf 'dataset\tmodel\tstatus\tlog\tresult_dir\n'
  for file in "${LOG_DIR}"/summary_*.tsv; do
    [[ -f "${file}" ]] || continue
    tail -n +2 "${file}"
  done
} > "${LOG_DIR}/summary_all.tsv"

echo "Combined summary: ${LOG_DIR}/summary_all.tsv"
exit "${status}"
