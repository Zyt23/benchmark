#!/usr/bin/env python
"""Assemble the leakage-safe QAR analysis sweep into final local tables.

The server collector intentionally keeps failed attempts for auditability.  This
script resolves repair shards by selecting the latest successful logical cell,
checks the requested experiment grid, combines it with the stable base tables,
and prepares the input directory consumed by the one-sheet Excel builder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def successful(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return frame.copy()
    status = pd.to_numeric(frame["status"], errors="coerce")
    return frame[status.eq(0)].copy()


def dedupe(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    existing = [key for key in keys if key in frame]
    return frame.drop_duplicates(existing, keep="last").reset_index(drop=True)


def derive_normalx_variant(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dataset = out["dataset"].astype(str)
    out.loc[dataset.str.endswith("_normalx2"), "variant"] = "normalx2"
    out.loc[dataset.str.endswith("_normalx4"), "variant"] = "normalx4"
    return out


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def assert_rows(name: str, frame: pd.DataFrame, expected: int) -> None:
    if len(frame) != expected:
        raise RuntimeError(f"{name}: expected {expected} rows, got {len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_cls = read(args.base_dir / "all_classification_metrics.csv")
    base_fc = read(args.base_dir / "all_forecast_metrics.csv")
    base_zero = read(args.base_dir / "all_zero_shot_metrics.csv")
    base_anomaly = read(args.base_dir / "all_anomaly_metrics.csv")

    cls = successful(read(args.extra_dir / "all_classification_metrics.csv"))
    fc = successful(read(args.extra_dir / "all_forecast_metrics.csv"))
    zero = successful(read(args.extra_dir / "all_zero_shot_metrics.csv"))
    forecast_anomaly = successful(
        read(args.extra_dir / "all_forecast_anomaly_metrics.csv"))

    scale_variants = ["both_keep25", "both_keep50", "normal_keep25", "normal_keep50"]
    scale_cls = dedupe(
        cls[cls["variant"].isin(scale_variants)],
        ["variant", "dataset", "model"],
    )

    normal_aug_cls = derive_normalx_variant(
        cls[cls["experiment_group"].astype(str).str.contains("normal_aug")])
    normal_aug_cls = dedupe(normal_aug_cls, ["variant", "dataset", "model"])

    patch_cls = cls[cls["experiment_group"].astype(str).str.contains("patchlen")].copy()
    patch_cls["variant"] = "patchlen"
    patch_cls = dedupe(patch_cls, ["patch_len", "dataset", "model"])

    scale_fc = dedupe(
        fc[fc["variant"].isin(["both_keep25", "both_keep50"])],
        ["variant", "anchor", "dataset", "model"],
    )

    normal_aug_fc = derive_normalx_variant(
        fc[fc["experiment_group"].astype(str).str.contains("normal_aug")])
    normal_aug_fc = dedupe(
        normal_aug_fc, ["variant", "anchor", "dataset", "model"])

    patch_fc = fc[fc["experiment_group"].astype(str).str.contains("patchlen")].copy()
    patch_fc["variant"] = "patchlen"
    patch_fc = dedupe(patch_fc, ["patch_len", "anchor", "dataset", "model"])

    context = zero[zero["experiment_group"].astype(str).eq("foundation_context40")].copy()
    # TiRex here is the older TiRex release.  The requested TiRex-2 checkpoint
    # is gated and was not substituted with this different model.
    context = context[context["model"].isin(["Chronos2", "Toto", "Moirai"])]
    context = dedupe(context, ["history_count", "dataset", "model"])

    univariate = zero[
        zero["experiment_group"].astype(str).eq("univariate_foundation")]
    univariate = dedupe(univariate, ["anchor", "dataset", "model", "target"])

    forecast_anomaly = dedupe(
        forecast_anomaly, ["anchor", "dataset", "model"])

    assert_rows("classification data scale", scale_cls, 220)
    assert_rows("classification normal augmentation", normal_aug_cls, 60)
    assert_rows("classification patch sweep", patch_cls, 55)
    assert_rows("forecast data scale", scale_fc, 440)
    assert_rows("forecast normal augmentation", normal_aug_fc, 240)
    assert_rows("forecast patch sweep", patch_fc, 440)
    assert_rows("foundation context (three available models)", context, 132)
    assert_rows("univariate Sundial", univariate, 44)
    assert_rows("forecast-head anomaly detection", forecast_anomaly, 220)

    requested_cls = pd.concat([scale_cls, normal_aug_cls, patch_cls], ignore_index=True, sort=False)
    requested_fc = pd.concat([scale_fc, normal_aug_fc, patch_fc], ignore_index=True, sort=False)
    requested_zero = pd.concat([context, univariate], ignore_index=True, sort=False)

    combined_cls = pd.concat([base_cls, requested_cls], ignore_index=True, sort=False)
    combined_fc = pd.concat([base_fc, requested_fc], ignore_index=True, sort=False)
    combined_zero = pd.concat([base_zero, requested_zero], ignore_index=True, sort=False)

    write(combined_cls, args.output_dir / "all_classification_metrics.csv")
    write(combined_fc, args.output_dir / "all_forecast_metrics.csv")
    write(combined_zero, args.output_dir / "all_zero_shot_metrics.csv")
    write(base_anomaly, args.output_dir / "all_anomaly_metrics.csv")
    write(forecast_anomaly, args.output_dir / "all_forecast_anomaly_metrics.csv")

    write(scale_cls, args.output_dir / "classification_data_scale.csv")
    write(normal_aug_cls, args.output_dir / "classification_normal_augmentation.csv")
    write(patch_cls, args.output_dir / "classification_patch_length.csv")
    write(scale_fc, args.output_dir / "forecast_data_scale.csv")
    write(normal_aug_fc, args.output_dir / "forecast_normal_augmentation.csv")
    write(patch_fc, args.output_dir / "forecast_patch_length.csv")
    write(context, args.output_dir / "foundation_context_forecast.csv")
    write(univariate, args.output_dir / "univariate_sundial_forecast.csv")

    def mean_table(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
        work = frame.copy()
        available = [metric for metric in metrics if metric in work]
        for metric in available:
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
        return work.groupby(groups, dropna=False, as_index=False)[available].mean()

    write(
        mean_table(scale_cls, ["variant", "model"], ["acc", "macro_f1", "weighted_f1"]),
        args.output_dir / "summary_classification_data_scale.csv",
    )
    write(
        mean_table(normal_aug_cls, ["variant", "model"], ["acc", "macro_f1", "weighted_f1"]),
        args.output_dir / "summary_classification_normal_augmentation.csv",
    )
    write(
        mean_table(patch_cls, ["patch_len", "model"], ["acc", "macro_f1", "weighted_f1"]),
        args.output_dir / "summary_classification_patch_length.csv",
    )
    write(
        mean_table(scale_fc, ["variant", "anchor", "model"], ["mae", "mse", "rmse"]),
        args.output_dir / "summary_forecast_data_scale.csv",
    )
    write(
        mean_table(normal_aug_fc, ["variant", "anchor", "model"], ["mae", "mse", "rmse"]),
        args.output_dir / "summary_forecast_normal_augmentation.csv",
    )
    write(
        mean_table(patch_fc, ["patch_len", "anchor", "model"], ["mae", "mse", "rmse"]),
        args.output_dir / "summary_forecast_patch_length.csv",
    )
    write(
        mean_table(context, ["history_count", "model"], ["mae", "mse", "rmse"]),
        args.output_dir / "summary_foundation_context.csv",
    )
    write(
        mean_table(univariate, ["anchor", "model", "target"], ["mae", "mse", "rmse"]),
        args.output_dir / "summary_univariate_sundial.csv",
    )
    write(
        mean_table(
            forecast_anomaly,
            ["anchor", "model"],
            ["accuracy", "balanced_accuracy", "f1", "macro_f1", "auroc", "auprc"],
        ),
        args.output_dir / "summary_forecast_head_anomaly.csv",
    )

    for name in [
        "dataset12_augmentation_manifest.csv",
        "shortcut_audit_summary.csv",
        "split_audit.csv",
    ]:
        source = args.base_dir / name
        if source.exists():
            shutil.copy2(source, args.output_dir / name)

    coverage = pd.DataFrame([
        ("分类：两类同比缩减 + 仅正常缩减", 220, len(scale_cls), "complete"),
        ("分类：正常样本扩增 x2/x4", 60, len(normal_aug_cls), "complete"),
        ("分类：PatchTST patch=16/8/4/2/1", 55, len(patch_cls), "complete"),
        ("预测：两类同比缩减", 440, len(scale_fc), "complete"),
        ("预测：正常样本扩增 x2/x4", 240, len(normal_aug_fc), "complete"),
        ("预测：PatchTST/TimeXer patch=16/8/4/2/1", 440, len(patch_fc), "complete"),
        ("大模型上下文：Chronos-2/Toto-2.0/Moirai", 132, len(context), "complete"),
        ("大模型上下文：TiRex-2", 44, 0, "gated weights + server egress unavailable"),
        ("单变量 Sundial", 44, len(univariate), "complete"),
        ("预测头异常检测", 220, len(forecast_anomaly), "complete"),
    ], columns=["experiment", "expected_rows", "success_rows", "status"])
    write(coverage, args.output_dir / "coverage_audit.csv")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
