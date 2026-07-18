#!/usr/bin/env bash
set -euo pipefail

# Repair TiRex rows from the context40 zero-shot forecast sweep.
#
# The original launch pointed TIREX_MODEL_PATH at external_models/tirex, while
# the implementation expects the actual checkpoint file.  This reruns the same
# run tags after dropping failed summary rows, so the existing expected manifest
# remains valid.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON="${PYTHON:-/home/para/anaconda3/bin/python}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14}"
CONTEXT_ROOT="${CONTEXT_ROOT:-datasetall_tsfile_compact_context40_predict23_20260717}"
HISTORY_COUNTS="${HISTORY_COUNTS:-2 3 5 8}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717_c40local}"
TIREX_MODEL_PATH="${TIREX_MODEL_PATH:-NX-AI/TiRex}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
GPU="${GPU:-0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/foundation_context40_tirex_fix}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

reset_failed_summary_rows() {
  local run_tag="$1"
  "${PYTHON}" - "${run_tag}" <<'PY'
import csv
import sys
from pathlib import Path

run_tag = sys.argv[1]
path = Path("logs/zero_shot_forecast") / run_tag / "summary.tsv"
if not path.exists():
    return_code = 0
else:
    backup = path.with_name("summary.before_tirex_fix.tsv")
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

for hist in ${HISTORY_COUNTS}; do
  seq_len=$((hist * 80 + 40))
  root="${CONTEXT_ROOT}/hist${hist}/predict_2_3"
  run_tag="context40_hist${hist}_predict_2_3_TiRex_${RUN_SUFFIX}"
  log="${LOG_ROOT}/${run_tag}.launcher.log"
  reset_failed_summary_rows "${run_tag}"
  echo "[launch] TiRex hist=${hist} seq_len=${seq_len} gpu=${GPU} model=${TIREX_MODEL_PATH}"
  (
    env \
      DATASETS="${DATASETS}" \
      MODELS="TiRex" \
      COMPACT_ROOT="${root}" \
      RUN_TAG="${run_tag}" \
      CUDA_DEVICES="${GPU}" \
      SEQ_LEN="${seq_len}" \
      LABEL_LEN=40 \
      PRED_LEN=40 \
      BATCH_SIZE="${TIREX_BATCH_SIZE:-1}" \
      NUM_WORKERS="${NUM_WORKERS:-0}" \
      QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
      TIREX_MODEL_PATH="${TIREX_MODEL_PATH}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
      bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
  ) > "${log}" 2>&1
done

echo "[done] TiRex context40 repair"
