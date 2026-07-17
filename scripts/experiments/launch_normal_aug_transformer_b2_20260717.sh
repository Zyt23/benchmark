#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASETS_AUG="${DATASETS_AUG:-dataset9_normalx2 dataset9_normalx4 dataset10_normalx2 dataset10_normalx4 dataset12_normalx2 dataset12_normalx4 dataset5_normalx2 dataset5_normalx4 dataset6_normalx2 dataset6_normalx4 dataset7_normalx2 dataset7_normalx4}"
NORMAL_AUG_CLS_ROOT="${NORMAL_AUG_CLS_ROOT:-datasetall_tsfile_compact_normal_aug_cls_20260717}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
LOG_ROOT="${LOG_ROOT:-${ARTIFACT_ROOT}/server_logs/normal_aug_transformer_b2}"
RUN_SUFFIX="${RUN_SUFFIX:-20260717_b2}"
GPU="${GPU:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

expected="${LOG_ROOT}/expected_jobs.tsv"
run_tag="normalx_cls_Transformer_${RUN_SUFFIX}"
printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${expected}"
printf "classification\tnormalx\t\tTransformer\t%s\t%s\t%s\n" "${DATASETS_AUG}" "${run_tag}" "${NORMAL_AUG_CLS_ROOT}" >> "${expected}"

log="${LOG_ROOT}/${run_tag}.launcher.log"
echo "[launch] normalx Transformer batch2 gpu=${GPU} run_tag=${run_tag}"
(
  env \
    DATASETS="${DATASETS_AUG}" \
    MODELS="Transformer" \
    COMPACT_ROOT="${NORMAL_AUG_CLS_ROOT}" \
    RUN_TAG="${run_tag}" \
    CUDA_DEVICES="${GPU}" \
    USE_MULTI_GPU=0 \
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}" \
    PATIENCE="${PATIENCE:-4}" \
    BATCH_SIZE="${CLS_BATCH_SIZE:-2}" \
    CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}" \
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}" \
    QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}" \
    CHECKPOINTS="checkpoints_datasetall/${run_tag}" \
    bash scripts/classification/run_QAR_tsfile_shiftN80.sh
) > "${log}" 2>&1 < /dev/null &

echo "[wait] normalx Transformer batch2 is running; expected jobs: ${expected}"
wait
echo "[done] normalx Transformer batch2"
