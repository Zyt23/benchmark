#!/usr/bin/env bash
set -euo pipefail

# Re-run only cells that failed because the original TSLib setting name
# exceeded Linux's per-component filename length limit.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/para/anaconda3/bin/python}"
COMPACT_BASE="${COMPACT_BASE:-datasetall_tsfile_compact_custom_forecast_chrono_20260711}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiment_artifacts/QAR_extra_experiments_20260717}"
GPU_LIST="${GPU_LIST:-1 4 5 6}"
GPU_LIST="${GPU_LIST//,/ }"
RUN_SUFFIX="${RUN_SUFFIX:-20260720_pathfix}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"

cd "${PROJECT_ROOT}"
read -r -a gpus <<< "${GPU_LIST}"
log_root="${ARTIFACT_ROOT}/server_logs/forecast_head_anomaly_pathfix_20260720"
mkdir -p "${log_root}"
printf "task\tvariant\tanchor\tmodel\tdatasets\trun_tag\tcompact_root\n" > "${log_root}/expected_jobs.tsv"

job_idx=0
for anchor in ${ANCHORS}; do
  for model in Transformer iTransformer; do
    if [[ "${model}" == "Transformer" ]]; then
      datasets="dataset8-1 dataset10 dataset11 dataset12 dataset13 dataset14"
    else
      datasets="dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14"
    fi
    gpu="${gpus[$((job_idx % ${#gpus[@]}))]}"
    job_idx=$((job_idx + 1))
    run_tag="forecast_head_anomaly_${anchor}_${model}_${RUN_SUFFIX}"
    printf "forecast_anomaly\tbase\t%s\t%s\t%s\t%s\t%s/%s\n" \
      "${anchor}" "${model}" "${datasets}" "${run_tag}" "${COMPACT_BASE}" "${anchor}" \
      >> "${log_root}/expected_jobs.tsv"
    (
      env PROJECT_ROOT="${PROJECT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
        COMPACT_ROOT="${COMPACT_BASE}/${anchor}" DATASETS="${datasets}" \
        MODELS="${model}" RUN_TAG="${run_tag}" CUDA_DEVICE="${gpu}" LOCAL_GPU=0 \
        FORECAST_ANOMALY_SCORE=auto THRESHOLD_SOURCE=val_mixed_best_f1 \
        TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}" PATIENCE="${PATIENCE:-3}" \
        bash scripts/anomaly_detection/run_QAR_forecast_head_anomaly.sh
    ) > "${log_root}/${run_tag}.launcher.log" 2>&1 < /dev/null &
  done
done

wait
echo "[done] repaired forecast-head anomaly cells"
