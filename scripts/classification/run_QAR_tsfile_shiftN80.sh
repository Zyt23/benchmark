#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the 11 tsfile-derived QAR compact datasets.
# The actual training command is shared with the datasetall compact sweep.

export DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
export COMPACT_ROOT="${COMPACT_ROOT:-./datasetall_tsfile_compact}"
export RUN_TAG="${RUN_TAG:-tsfile_shiftN80_$(date +%Y%m%d_%H%M%S)}"
export CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}"
export SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-0}"

bash scripts/classification/run_QAR_datasetall_shiftN80.sh
