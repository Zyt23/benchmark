# QAR experiment manifest 20260717

This manifest records the experiment matrix requested on 2026-07-17.  It is
kept in git so the benchmark can be relaunched without relying on an
`experiment_artifacts` code snapshot.

## Shared datasets

Base datasets:

`dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14`

Fault descriptions:

- dataset5: 320-感压管路故障
- dataset6: 320-HPV活门故障
- dataset7: 320-PRV活门故障
- dataset8: 320-管道漏气
- dataset8-1: 320-PRV活门漏气
- dataset9: 321-HPV故障（LEAP）
- dataset10: 321-感压管路故障（LEAP）
- dataset11: 321-管道漏气
- dataset12: 321-PRV故障（LEAP）
- dataset13: 787机型空气压缩机故障
- dataset14: 777机型PRSOV故障

## Base split and condition logic

- Classification compact root: `datasetall_tsfile_compact_custom_cls_chrono_20260711`.
- Forecast compact root: `datasetall_tsfile_compact_custom_forecast_chrono_20260711`.
- Classification uses the custom multi-anchor condition logic.  For dataset5~12
  and dataset14, the 6→8 cruise→approach anchor is excluded because many
  flights do not contain that transition.  dataset13 uses
  `datasetall_tsfile/build_dataset15_1.py`-style variables and anchors.
- Forecast uses four independent phase-transition datasets:
  `predict_2_3`, `predict_4_5`, `predict_5_6`, `predict_8_9`.
  Each segment has 80 points: 30 before and 50 after the transition.
- The compact loader splits by `time_keys`/`sources`: train 70%, validation 10%,
  test 20%, with train earlier than validation and validation earlier than test.
  For `per_class_chrono`, the chronological split is done separately inside
  each class.

## Experiment blocks

### A. Data scale down

Launcher: `scripts/experiments/launch_data_scale_experiments_20260717.sh`

Classification:

- Both normal and fault kept at 50%: `both_keep50`
- Both normal and fault kept at 25%: `both_keep25`
- Normal kept at 50%, fault unchanged: `normal_keep50`
- Normal kept at 25%, fault unchanged: `normal_keep25`

Forecast:

- Both normal and fault kept at 50%: `both_keep50`
- Both normal and fault kept at 25%: `both_keep25`
- All four forecast anchors are run.

Models: `Transformer TimesNet PatchTST DLinear iTransformer`.

### B. Normal data scale up

Launcher: `scripts/experiments/launch_normal_aug_experiments_20260717.sh`

Normal class is expanded to 200% and 400%; fault samples stay unchanged.

Available extra-normal packages:

- `data12-0类追加csv数据(1).zip` applies to dataset9/dataset10/dataset12.
- `320-HPV-正常-追加567.zip` applies to dataset5/dataset6/dataset7.

Other datasets are intentionally blank/NA until corresponding normal packages
are available.

Tasks: classification and four-anchor forecasting.

Models: `Transformer TimesNet PatchTST DLinear iTransformer`.

### C. Patch length sweep

Launcher: `scripts/experiments/launch_patchlen_sweep_20260717.sh`

Patch lengths: `16 8 4 2 1`.

- Classification: PatchTST only.  The current TimeXer implementation has no
  classification head.
- Forecast: PatchTST and TimeXer, all four forecast anchors.

### D. Foundation-model history context

Launcher: `scripts/experiments/launch_foundation_context40_20260717.sh`

Anchor: 2→3 only.  Each historical flight segment has 80 points, using
30-before/50-after around the transition.  For the target flight, the first 40
points are given as current context and the last 40 are predicted.

History counts: `2 3 5 8`.

Therefore:

- `hist2`: `seq_len=200`, `pred_len=40`
- `hist3`: `seq_len=280`, `pred_len=40`
- `hist5`: `seq_len=440`, `pred_len=40`
- `hist8`: `seq_len=680`, `pred_len=40`

Zero-shot models: `Chronos2 Toto Moirai TiRex`.

### E. Univariate foundation forecasting

Launcher: `scripts/experiments/launch_univariate_foundation_20260717.sh`

The compact cache is projected to one pressure-like variable.  Default target
alias: `manifold_pressure`, resolved in order:

`PRECOOL_PRESS1`, `PRECOOL_PRESS2`, `BMPS1`, `BMPS2`.

Default model: `Sundial`.

### F. Anomaly detection diagnosis

Current one-class anomaly protocol:

- Train only on chronological normal training flights.
- Select threshold from normal validation reconstruction error.
- Test on held-out normal flights plus all fault flights.

The previous p95 results have low recall, so accuracy can look misleading.  If
rerun is needed, use a threshold sweep (e.g. p80/p85/p90/p95) and/or longer
`SEQ_LEN` to avoid losing short fault cues through resampling.
