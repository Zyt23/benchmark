#!/usr/bin/env bash
set -euo pipefail

# Repair failed iTransformer normal_keep rows from data_scale_final_itransformer_b96.
#
# The original run succeeded on the smaller datasets but failed on six larger
# normal_keep datasets.  This reruns only those missing/failed dataset variants
# with the same run tags and a smaller batch size, after removing failed rows
# from the old summary files.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SCALE_CLS_ROOT="${SCALE_CLS_ROOT:-datasetall_tsfile_compact_scale_cls_20260717}"
GPU="${GPU:-0}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-16}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/data_scale_itransformer_b16_fix}"

RUN_TAG_KEEP50="${RUN_TAG_KEEP50:-scale_normal_keep50_cls_iTransformer_20260717_final_itransformer_b96}"
RUN_TAG_KEEP25="${RUN_TAG_KEEP25:-scale_normal_keep25_cls_iTransformer_20260717_final_itransformer_b96}"
DATASETS_KEEP50="${DATASETS_KEEP50:-dataset8-1_normal_keep50 dataset10_normal_keep50 dataset11_normal_keep50 dataset12_normal_keep50 dataset13_normal_keep50 dataset14_normal_keep50}"
DATASETS_KEEP25="${DATASETS_KEEP25:-dataset8-1_normal_keep25 dataset10_normal_keep25 dataset11_normal_keep25 dataset12_normal_keep25 dataset13_normal_keep25 dataset14_normal_keep25}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

reset_failed_summary_rows() {
  local run_tag="$1"
  python - "${run_tag}" <<'PY'
import csv
import sys
from pathlib import Path

run_tag = sys.argv[1]
path = Path("logs/datasetall") / run_tag / "summary.tsv"
if not path.exists():
    raise SystemExit(0)
backup = path.with_name("summary.before_itransformer_b16_fix.tsv")
if not backup.exists():
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = rows[0].keys() if rows else ["dataset", "model", "status", "log", "result_dir"]
kept = [row for row in rows if str(row.get("status", "")).strip() in {"0", "0.0"}]
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t")
    writer.writeheader()
    writer.writerows(kept)
print(f"{path}: kept {len(kept)} successful rows, backup={backup}")
PY
}

run_repair() {
  local run_tag="$1"
  local datasets="$2"
  local log="${LOG_ROOT}/${run_tag}.repair_b${CLS_BATCH_SIZE}.log"
  reset_failed_summary_rows "${run_tag}"
  echo "[launch] iTransformer repair run_tag=${run_tag} batch=${CLS_BATCH_SIZE} gpu=${GPU}"
  (
    env \
      DATASETS="${datasets}" \
      MODELS="iTransformer" \
      COMPACT_ROOT="${SCALE_CLS_ROOT}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${GPU}" \
      USE_MULTI_GPU=0 \
      TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}" \
      PATIENCE="${PATIENCE:-4}" \
      BATCH_SIZE="${CLS_BATCH_SIZE}" \
      CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}" \
      EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}" \
      QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
      CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
      bash scripts/classification/run_QAR_tsfile_shiftN80.sh
  ) > "${log}" 2>&1
}

run_repair "${RUN_TAG_KEEP50}" "${DATASETS_KEEP50}"
run_repair "${RUN_TAG_KEEP25}" "${DATASETS_KEEP25}"

echo "[done] iTransformer normal_keep b${CLS_BATCH_SIZE} repair"
