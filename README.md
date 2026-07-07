# QAR Benchmark

Clean benchmark repository for QAR tsfile compact-cache experiments.

This repository keeps only runnable code, scripts, and documentation. Large
datasets, compact caches, logs, checkpoints, and experiment artifacts are
excluded by `.gitignore`; version history should be tracked by git rather than
by copying code snapshots into `experiment_artifacts/`.

## Tasks

- Classification: `run.py --task_name classification --data QAR_shift`
- Forecasting: `run.py --task_name long_term_forecast --data QAR_forecast`

Both tasks consume compact caches:

```text
<COMPACT_ROOT>/<dataset>/qar_compact_shiftN80.npz
```

The compact cache contains `x`, `mask`, `labels`, `feature_cols`,
`class_names`, and `phase_a_shift`.

## Current version

- The training code is versioned directly by git. `experiment_artifacts/` is
  used only for collected outputs, not for code snapshots.
- Classification and forecasting both read compact caches from
  `<COMPACT_ROOT>/<dataset>/qar_compact_shiftN80.npz`; the model code does not
  connect to IoTDB during training.
- Dataset-specific custom flight-condition logic is applied when building the
  compact cache:
  - `dataset13`: anchors from `build_dataset15_1.py`;
  - `dataset5`-`dataset12`, `dataset8-1`, and `dataset14`: anchors from
    `320321gongkuang.py`;
  - `dataset12_aug0`: original `dataset12` plus extra class-0 CSV flights,
    built with the same `320321gongkuang.py` anchors.
- Forecasting does not define separate flight conditions, but if it reads a
  compact cache built with the custom-condition script, it uses the same custom
  windows as classification. The forecasting loader expands each compact
  flight/window into sliding prediction windows. By default it slides within
  each anchor segment (`forecast_window_mode=segment`) with
  `forecast_stride=80`, so it does not predict across artificial concatenation
  boundaries between flight-condition snippets.

## Dataset mapping

See [docs/DATASETS.md](docs/DATASETS.md).

## Prepare compact caches from tsfile zip

Standard 2→3 shiftN80 windows:

```bash
python tools/prepare/prepare_tsfile_compact_from_zip.py \
  --zip_path /path/to/tsfile_datasets.zip \
  --output_root /path/to/datasetall_tsfile_compact \
  --iotdb_lib /path/to/iotdb/lib \
  --java_src scripts/tsfile/TsFileWindowDumper.java
```

Dataset-specific custom conditions for all datasets:

```bash
python tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path /path/to/tsfile_datasets.zip \
  --output_root /path/to/datasetall_tsfile_compact_anchor \
  --iotdb_lib /path/to/iotdb/lib \
  --java_src scripts/tsfile/TsFileWindowDumperAnchors.java \
  --mode_set anchor \
  --datasets dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14
```

To build the dataset12 extra-normal version:

```bash
python tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path /path/to/tsfile_datasets.zip \
  --output_root /path/to/datasetall_tsfile_compact_anchor \
  --iotdb_lib /path/to/iotdb/lib \
  --java_src scripts/tsfile/TsFileWindowDumperAnchors.java \
  --mode_set anchor \
  --datasets dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14 \
  --dataset12_aug0_csv_zip /path/to/data12-0类追加csv数据(1).zip
```

The custom-condition script uses:

- `dataset13_anchors`: phase anchors from `build_dataset15_1.py`
- `standard_320321_anchors`: standard 16 QAR features plus phase anchors from
  `320321gongkuang.py`
- `dataset14_anchors`: native dataset14 features plus phase anchors from
  `320321gongkuang.py`

Phase-start forecasting compact caches:

```bash
python tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path /path/to/tsfile_datasets.zip \
  --output_root /path/to/datasetall_tsfile_compact_phase_start80 \
  --iotdb_lib /path/to/iotdb/lib \
  --java_src scripts/tsfile/TsFileWindowDumperAnchors.java \
  --mode_set phase_start80 \
  --datasets dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14 \
  --dataset12_aug0_csv_zip /path/to/data12-0类追加csv数据(1).zip
```

For CSV source datasets, a parameterized helper is available:

```bash
python scripts/preprocess/build_phase_anchor_dataset.py \
  --src_root /path/to/source_dataset \
  --dst_root /path/to/output_dataset \
  --preset dataset15_1
```

or:

```bash
python scripts/preprocess/build_phase_anchor_dataset.py \
  --src_root /path/to/source_dataset \
  --dst_root /path/to/output_dataset \
  --preset 320321
```

## Run classification

```bash
COMPACT_ROOT=/path/to/datasetall_tsfile_compact \
RUN_TAG=qar_cls_$(date +%Y%m%d_%H%M%S) \
PYTHON=/home/para/anaconda3/bin/python \
CUDA_DEVICES=5,6,7 \
DEVICES=0,1,2 \
bash scripts/classification/run_QAR_tsfile_shiftN80.sh
```

Useful debug/smoke-test override:

```bash
DATASETS=dataset13 MODELS=DLinear TRAIN_EPOCHS=1 PATIENCE=1 BATCH_SIZE=256 \
COMPACT_ROOT=/path/to/datasetall_tsfile_compact \
bash scripts/classification/run_QAR_tsfile_shiftN80.sh
```

Classification defaults:

- models: `Transformer TimesNet PatchTST DLinear iTransformer`
- `class_weight=balanced`
- `early_stop_metric=macro_f1`
- result metrics include `accuracy`, `macro_f1`, `weighted_f1`, `true_counts`,
  `pred_counts`, and binary `TN/FP/FN/TP` when applicable.

## Run forecasting

```bash
COMPACT_ROOT=/path/to/datasetall_tsfile_compact \
RUN_TAG=qar_forecast_$(date +%Y%m%d_%H%M%S) \
PYTHON=/home/para/anaconda3/bin/python \
CUDA_DEVICES=5,6,7 \
DEVICES=0,1,2 \
bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
```

Useful debug/smoke-test override:

```bash
DATASETS=dataset13 MODELS=DLinear TRAIN_EPOCHS=1 PATIENCE=1 BATCH_SIZE=256 \
COMPACT_ROOT=/path/to/datasetall_tsfile_compact \
bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
```

Forecasting defaults:

- `seq_len=60`, `label_len=20`, `pred_len=20`
- `forecast_window_mode=segment`, `forecast_stride=80`; this uses multiple
  windows across each compact flight/window instead of only the first 80 points.
- `features=M`, multivariate-to-multivariate forecasting
- metrics: `mae`, `mse`, `rmse`, `mape`, `mspe`
- `mape/mspe` can be `inf/nan` if true values contain zero; prefer
  `mae/mse/rmse` for comparison.

## Collect result tables

Classification:

```bash
python tools/collect/collect_qar_metrics_tables.py \
  --run_tags <RUN_TAG> \
  --output_dir experiment_artifacts/<RUN_TAG> \
  --compact_root /path/to/datasetall_tsfile_compact
```

Forecasting:

```bash
python tools/collect/collect_qar_forecast_metrics_tables.py \
  --run_tags <RUN_TAG> \
  --output_dir experiment_artifacts/<RUN_TAG> \
  --compact_root /path/to/datasetall_tsfile_compact
```

## Notes

- This repo intentionally does not track datasets or experiment outputs.
- Root-level files are kept to the minimum runnable entry points. Dataset/cache
  builders live under `tools/prepare/`; result table collectors live under
  `tools/collect/`.
- This repo does not include direct IoTDB connection credentials. Training and
  forecasting consume compact caches; rebuild those caches outside the repo by
  passing explicit source paths and IoTDB/TSFile Java library paths.
- On shared servers, checkpoints default to `$HOME/qar_checkpoint_archive/...`
  to avoid filling `/data`.
- dataset13/dataset14 should use the custom-condition compact caches when
  available; the old 16-feature dataset13 cache can be all-zero due to field
  mismatch.
