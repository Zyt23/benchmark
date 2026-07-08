#!/usr/bin/env bash
set -euo pipefail

# Rerun custom-condition QAR classification after fixing validation leakage.
#
# This script starts three independent background groups:
#   - core:        TimesNet, DLinear, iTransformer on GPUs 4,5
#   - transformer: Transformer on GPU 7
#   - patchtst:    PatchTST on GPU 6
#
# Each group first runs the 11 regular datasets and then dataset12_aug0.
# Results are written under distinct run tags:
#   valfix_cls_core_20260708
#   valfix_cls_transformer_20260708
#   valfix_cls_patchtst_20260708
#   valfix_cls_core_aug0_20260708
#   valfix_cls_transformer_aug0_20260708
#   valfix_cls_patchtst_aug0_20260708

COMMON_DATASETS="${COMMON_DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
PATIENCE="${PATIENCE:-3}"
LOG_DIR="${LOG_DIR:-logs/rerun_valfix_20260708}"
ANCHOR_ROOT_MAIN="${ANCHOR_ROOT_MAIN:-./datasetall_tsfile_compact_anchor_20260707}"
ANCHOR_ROOT_AUG0="${ANCHOR_ROOT_AUG0:-./datasetall_tsfile_compact_anchor_aug0_20260707}"

mkdir -p "${LOG_DIR}"

start_group() {
  local name="$1"
  local stage="$2"
  local run_tag="$3"
  local run_tag_aug0="$4"
  local log_file="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  (
    set -euo pipefail
    echo "[$(date '+%F %T')] START ${name}: ${COMMON_DATASETS}"
    TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
    PATIENCE="${PATIENCE}" \
    RUN_TAG="${run_tag}" \
    DATASETS="${COMMON_DATASETS}" \
    ANCHOR_ROOT="${ANCHOR_ROOT_MAIN}" \
    bash scripts/experiments/run_custom_conditions_20260707.sh "${stage}"

    echo "[$(date '+%F %T')] START ${name}: dataset12_aug0"
    TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
    PATIENCE="${PATIENCE}" \
    RUN_TAG="${run_tag_aug0}" \
    DATASETS="dataset12_aug0" \
    ANCHOR_ROOT="${ANCHOR_ROOT_AUG0}" \
    bash scripts/experiments/run_custom_conditions_20260707.sh "${stage}"

    echo "[$(date '+%F %T')] DONE ${name}"
  ) > "${log_file}" 2>&1 &

  echo "$!" > "${pid_file}"
  echo "${name}_pid=$!"
  echo "${name}_log=${log_file}"
}

status_group() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: not started"
    return
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: running pid=${pid}"
  else
    echo "${name}: finished pid=${pid}"
  fi
  if [[ -f "${log_file}" ]]; then
    tail -n 5 "${log_file}"
  fi
}

case "${1:-start}" in
  start)
    start_group core cls_core valfix_cls_core_20260708 valfix_cls_core_aug0_20260708
    start_group transformer cls_transformer valfix_cls_transformer_20260708 valfix_cls_transformer_aug0_20260708
    start_group patchtst cls_patchtst valfix_cls_patchtst_20260708 valfix_cls_patchtst_aug0_20260708
    ;;
  status)
    status_group core
    status_group transformer
    status_group patchtst
    ;;
  *)
    echo "Usage: $0 [start|status]" >&2
    exit 2
    ;;
esac
