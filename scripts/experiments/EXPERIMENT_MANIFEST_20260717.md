# QAR 扩展实验清单（2026-07-20 核对版）

本文件记录数据缩放、patch 消融、时序基础模型、单变量预测和预测误差异常检测实验。代码保存在仓库根目录，运行结果可以放在 `experiment_artifacts`。

## 共用数据与切分

数据集为：

`dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset9 dataset10 dataset11 dataset12 dataset13 dataset14`

- 分类 compact：`datasetall_tsfile_compact_custom_cls_chrono_20260711`。
- 预测 compact：`datasetall_tsfile_compact_custom_forecast_chrono_20260711`。
- 分类使用新的多锚点工况逻辑，并删除缺失较多的 6→8 工况。
- dataset13 使用 `datasetall_tsfile/build_dataset15_1.py` 对应的字段与锚点。
- 按正常/故障类别分别进行时间顺序 7:1:2 切分；TRAIN、VAL、TEST 源航班不重叠，且 TEST 保留故障样本。

## 预测切片的准确含义

四个工况分别建集：`2→3`、`4→5`、`5→6`、`8→9`。

每个样本共 80 点：

- 转换前 30 点；
- 转换后 50 点；
- 模型输入前 60 点，即“转换前 30 + 转换后前 30”；
- 模型预测最后 20 点，即“转换后第 31～50 点”。

因此，`seq_len=60, pred_len=20` 没有丢掉后 20 点；此前“前30+后30”的说法只是对完整切片描述不准确。

## A. 数据量缩减

启动器：`scripts/experiments/launch_data_scale_experiments_20260717.sh`

分类：

- `both_keep50`：正常和故障均保留 50%；
- `both_keep25`：正常和故障均保留 25%；
- `normal_keep50`：正常保留 50%，故障不变；
- `normal_keep25`：正常保留 25%，故障不变。

预测只运行 `both_keep50` 和 `both_keep25`，覆盖四个工况。模型为 Transformer、TimesNet、PatchTST、DLinear、iTransformer。

## B. 正常样本扩增到 200%/400%

启动器：`scripts/experiments/launch_normal_aug_experiments_20260717.sh`

- dataset5/6/7 使用 `datasetall/320-HPV-正常-追加567.zip`；
- dataset9/10/12 使用 `datasetall/data12-0类追加csv数据(1).zip`；
- 故障样本保持不变；
- 其他数据集暂留空，直到有对应机型的正常追加包。

分类和四工况预测均运行五个模型。PatchTST 分类失败单元由 `scripts/experiments/launch_missing_requested_20260720.sh` 低 batch 修复。

## C. Patch 长度消融

启动器：`scripts/experiments/launch_patchlen_sweep_20260717.sh`

patch 长度依次为 `16 8 4 2 1`。

- 分类：PatchTST；当前 TimeXer 没有分类头，不能做同构分类消融。
- 预测：PatchTST 和 TimeXer，覆盖四个工况。
- 缺失的 patch=8/4/2/1 以及 dataset14 的 patch=16 由 `launch_missing_requested_20260720.sh` 补跑。

## D. 时序基础模型长上下文

启动器：`scripts/experiments/launch_foundation_context40_20260717.sh`

- 只使用 2→3 工况；每个历史航班是前30+后50，共 80 点。
- 历史航班数为 2、3、5、8；用户描述中的 `5*89` 按与其他设置一致的 `5*80` 执行。
- 当前目标航班给出前 40 点，预测后 40 点。
- 对应 `seq_len` 分别为 200、280、440、680，`pred_len=40`。
- 模型：Chronos-2、Toto-2.0、Moirai、TiRex-2。

TiRex-2 需要独立 Python>=3.11、torch>=2.8 环境和已授权的 Hugging Face token，启动器为 `scripts/experiments/launch_tirex2_context40_20260720.sh`。

## E. 单变量基础模型预测

启动器：`scripts/experiments/launch_univariate_foundation_20260717.sh`

默认目标别名为 `manifold_pressure`，按以下顺序为每个数据集解析存在的压力变量：

`PRECOOL_PRESS1 → PRECOOL_PRESS2 → BMPS1 → BMPS2`

默认模型为 Sundial，覆盖四个工况。

## F. 预测误差异常检测

启动器：`scripts/experiments/launch_forecast_head_anomaly_20260719.sh`

- 五个预测模型只用正常 TRAIN 拟合预测能力；
- early stopping 只使用正常 VAL 的预测损失；
- 阈值选择使用独立的正常+故障 VAL，不接触 TEST；
- `forecast_anomaly_score=auto` 在 MSE、MAE、最大通道 MSE、最大时间点 MSE 之间按 VAL F1 自动选择；
- 每种分数的候选阈值和 VAL 指标保存到 `threshold_sweep.csv`；
- TEST 只评估最终选择一次，并输出 balanced accuracy、F1、AUROC、AUPRC 与 TN/FP/FN/TP。

预测 MSE 与异常检测指标的相关性由 `scripts/analysis/analyze_forecast_anomaly_correlation.py` 计算，不使用 TEST 反向调参。
