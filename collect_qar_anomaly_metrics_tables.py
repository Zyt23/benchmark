#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect QAR anomaly-detection metrics into dataset/model tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = ["KANAD", "AnomalyTransformer", "TranAD", "USAD", "OmniAnomaly"]

DATASET_NAMES = {
    "dataset5": "320-感压管路故障",
    "dataset6": "320-HPV活门故障",
    "dataset7": "320-PRV活门故障",
    "dataset8": "320-管道漏气",
    "dataset8-1": "320-PRV活门漏气",
    "dataset9": "321-HPV故障",
    "dataset10": "321-感压管路故障",
    "dataset11": "321-管道漏气",
    "dataset12": "321-PRV故障",
    "dataset13": "787机型空气压缩机故障",
    "dataset14": "777机型PRSOV故障",
}


def parse_setting(setting: str, run_tag: str, models: list[str]) -> tuple[str, str] | None:
    prefixes = [
        f"anomaly_detection_{run_tag}_",
        f"forecast_anomaly_detection_{run_tag}_",
    ]
    rest = None
    for prefix in prefixes:
        if setting.startswith(prefix):
            rest = setting[len(prefix):]
            break
    if rest is None:
        return None

    for model in sorted(models, key=len, reverse=True):
        markers = [
            f"_{model}_QAR_anomaly_",
            f"_{model}_QAR_forecast_anomaly_",
            f"_{model}_QAR_forecast_head_anomaly_",
            f"_{model}_QAR_forecast_",
        ]
        for marker in markers:
            if marker in rest:
                return rest.split(marker, 1)[0], model
    return None


def read_metric(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run_tag")
    run_group.add_argument(
        "--run_tags",
        nargs="+",
        help="Collect several model-specific run tags into one table.",
    )
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--metric_file_name", default="anomaly_metrics.csv",
                        help="metric filename, e.g. anomaly_metrics.csv or forecast_anomaly_metrics.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    run_tags = args.run_tags or [args.run_tag]
    records: list[dict] = []
    for metric_path in results_root.glob(f"*/{args.metric_file_name}"):
        row = read_metric(metric_path)
        if not row:
            continue
        parsed = None
        matched_run_tag = None
        for run_tag in run_tags:
            parsed = parse_setting(row.get("setting", metric_path.parent.name), run_tag, args.models)
            if parsed is None:
                parsed = parse_setting(metric_path.parent.name, run_tag, args.models)
            if parsed is not None:
                matched_run_tag = run_tag
                break
        if parsed is None:
            continue
        dataset, model = parsed
        if dataset not in args.datasets or model not in args.models:
            continue
        records.append(
            {
                "dataset": dataset,
                "fault": DATASET_NAMES.get(dataset, ""),
                "model": model,
                "run_tag": matched_run_tag,
                "accuracy": float(row.get("accuracy", "nan")),
                "balanced_accuracy": float(row.get("balanced_accuracy", "nan")),
                "precision": float(row.get("precision", "nan")),
                "recall": float(row.get("recall", "nan")),
                "f1": float(row.get("f1", "nan")),
                "macro_f1": float(row.get("macro_f1", "nan")),
                "roc_auc": float(row.get("roc_auc", "nan")),
                "pr_auc": float(row.get("pr_auc", "nan")),
                "raw_roc_auc": float(row.get("raw_roc_auc", "nan")),
                "raw_pr_auc": float(row.get("raw_pr_auc", "nan")),
                "true_counts": row.get("true_counts", ""),
                "pred_counts": row.get("pred_counts", ""),
                "TN": int(float(row.get("TN", "nan"))),
                "FP": int(float(row.get("FP", "nan"))),
                "FN": int(float(row.get("FN", "nan"))),
                "TP": int(float(row.get("TP", "nan"))),
                "threshold": float(row.get("threshold", "nan")),
                "threshold_source": row.get("threshold_source", ""),
                "threshold_percentile": float(row.get("threshold_percentile", "nan")),
                "level": row.get("level", row.get("score", "")),
                "score_direction": row.get("score_direction", ""),
                "normal_score_mean": float(row.get("normal_score_mean", "nan")),
                "fault_score_mean": float(row.get("fault_score_mean", "nan")),
                "normal_score_median": float(row.get("normal_score_median", "nan")),
                "fault_score_median": float(row.get("fault_score_median", "nan")),
                "normal_score_p95": float(row.get("normal_score_p95", "nan")),
                "fault_score_p95": float(row.get("fault_score_p95", "nan")),
                "threshold_val_accuracy": float(row.get("threshold_val_accuracy", "nan")),
                "threshold_val_precision": float(row.get("threshold_val_precision", "nan")),
                "threshold_val_recall": float(row.get("threshold_val_recall", "nan")),
                "threshold_val_f1": float(row.get("threshold_val_f1", "nan")),
                "setting": row.get("setting", metric_path.parent.name),
                "metric_file": str(metric_path),
            }
        )

    if not records:
        raise SystemExit(f"No anomaly metrics found for run_tags={run_tags!r} under {results_root}")

    df = pd.DataFrame(records)
    df["dataset_order"] = df["dataset"].map({d: i for i, d in enumerate(args.datasets)})
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(args.models)})
    df = df.sort_values(["dataset_order", "model_order"]).drop(columns=["dataset_order", "model_order"])

    all_csv = output_dir / "all_anomaly_metrics.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    for dataset, sub in df.groupby("dataset", sort=False):
        sub.to_csv(tables_dir / f"{dataset}_anomaly_metrics.csv", index=False, encoding="utf-8-sig")

    readme = output_dir / "README.md"
    with readme.open("w", encoding="utf-8") as f:
        f.write("# QAR anomaly detection\n\n")
        f.write("- 训练：默认只使用正常类 0。\n")
        f.write("- 阈值：来自正常验证集分位数，或来自验证集正常+故障的 best-F1 阈值；不使用测试集选阈值。\n")
        f.write("- 测试：按窗口级别统计 accuracy / precision / recall / f1 / TN / FP / FN / TP。\n\n")
        f.write(f"- run_tags: `{', '.join(run_tags)}`\n")
        f.write(f"- all metrics: `{all_csv.name}`\n\n")
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if sub.empty:
                continue
            f.write(f"## {dataset}: {DATASET_NAMES.get(dataset, '')}\n\n")
            cols = [
                "model", "accuracy", "precision", "recall", "f1",
                "true_counts", "pred_counts", "TN", "FP", "FN", "TP",
            ]
            f.write(sub[cols].to_markdown(index=False))
            f.write("\n\n")

    xlsx = output_dir / "QAR_anomaly_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_anomaly_metrics", index=False)
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if not sub.empty:
                safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", dataset)[:31]
                sub.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"Wrote {all_csv}")
    print(f"Wrote {readme}")
    print(f"Wrote {xlsx}")


if __name__ == "__main__":
    main()
