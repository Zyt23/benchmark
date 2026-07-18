#!/usr/bin/env bash
set -euo pipefail

# Repair failed iTransformer normal_keep rows from data_scale_final_itransformer_b96.
#
# The original run used a very long RUN_TAG, which overflowed the Linux
# per-file-name limit after the classification setting string was appended to
# the checkpoint path.  This launcher removes those two long-run expected rows
# and reruns the full normal_keep50/normal_keep25 iTransformer cells with short
# run tags.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
SCALE_CLS_ROOT="${SCALE_CLS_ROOT:-datasetall_tsfile_compact_scale_cls_20260717}"
GPU="${GPU:-0}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-16}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/data_scale_itransformer_cls_short}"
RUN_SUFFIX="${RUN_SUFFIX:-20260718short}"
FILTER_OLD_EXPECTED="${FILTER_OLD_EXPECTED:-1}"

DATASETS_BASE="${DATASETS_BASE:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
RUN_TAG_KEEP50="${RUN_TAG_KEEP50:-si_n50_IT_${RUN_SUFFIX}}"
RUN_TAG_KEEP25="${RUN_TAG_KEEP25:-si_n25_IT_${RUN_SUFFIX}}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

dataset_variant_list() {
  local variant="$1"
  local out=""
  for dataset in ${DATASETS_BASE}; do
    out="${out} ${dataset}_${variant}"
  done
  echo "${out# }"
}

DATASETS_KEEP50="${DATASETS_KEEP50:-$(dataset_variant_list normal_keep50)}"
DATASETS_KEEP25="${DATASETS_KEEP25:-$(dataset_variant_list normal_keep25)}"

if [[ "${FILTER_OLD_EXPECTED}" == "1" ]]; then
  echo "[repair] remove long iTransformer normal_keep rows from old expected_jobs.tsv"
  "${PYTHON}" - <<'PY'
from pathlib import Path

path = Path("experiment_artifacts/QAR_extra_experiments_20260717/server_logs/data_scale_final_itransformer_b96/expected_jobs.tsv")
if path.exists():
    backup = path.with_name("expected_jobs.before_itransformer_short.tsv")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    kept = []
    for line in rows:
        cols = line.split("\t")
        task = cols[0] if len(cols) > 0 else ""
        variant = cols[1] if len(cols) > 1 else ""
        model = cols[3] if len(cols) > 3 else ""
        if task == "classification" and model == "iTransformer" and variant in {"normal_keep50", "normal_keep25"}:
            continue
        kept.append(line)
    path.write_text("\n".join([header, *kept]) + "\n", encoding="utf-8")
    print(f"{path}: kept {len(kept)} rows, backup={backup}")
PY
fi

expected="${LOG_ROOT}/expected_jobs.tsv"
printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${expected}"
printf "classification\tnormal_keep50\t\tiTransformer\t%s\t%s\t%s\n" "${DATASETS_KEEP50}" "${RUN_TAG_KEEP50}" "${SCALE_CLS_ROOT}" >> "${expected}"
printf "classification\tnormal_keep25\t\tiTransformer\t%s\t%s\t%s\n" "${DATASETS_KEEP25}" "${RUN_TAG_KEEP25}" "${SCALE_CLS_ROOT}" >> "${expected}"

reset_failed_summary_rows() {
  local run_tag="$1"
  "${PYTHON}" - "${run_tag}" <<'PY'
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
      CHECKPOINTS="${HOME}/qar_checkpoint_archive/cc/${run_tag}" \
      bash scripts/classification/run_QAR_tsfile_shiftN80.sh
  ) > "${log}" 2>&1
}

run_repair "${RUN_TAG_KEEP50}" "${DATASETS_KEEP50}"
run_repair "${RUN_TAG_KEEP25}" "${DATASETS_KEEP25}"

echo "[done] iTransformer normal_keep b${CLS_BATCH_SIZE} repair"
