#!/usr/bin/env python
"""Collect QAR one-class anomaly-detection metrics into dataset/model tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


MODELS = ["Transformer", "TimesNet", "PatchTST", "DLinear", "iTransformer"]


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


def parse_setting(setting: str, run_tag: str) -> tuple[str, str] | None:
    prefix = f"anomaly_detection_{run_tag}_"
    if not setting.startswith(prefix):
        return None
    rest = setting[len(prefix):]
    for model in MODELS:
        marker = f"_{model}_QAR_anomaly_"
        if marker in rest:
            dataset = rest.split(marker, 1)[0]
            return dataset, model
    return None


def read_metric(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_NAMES))
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for metric_path in results_root.glob("*/anomaly_metrics.csv"):
        row = read_metric(metric_path)
        if not row:
            continue
        parsed = parse_setting(row.get("setting", metric_path.parent.name), args.run_tag)
        if parsed is None:
            parsed = parse_setting(metric_path.parent.name, args.run_tag)
        if parsed is None:
            continue
        dataset, model = parsed
        if dataset not in args.datasets or model not in args.models:
            continue
        record = {
            "dataset": dataset,
            "fault": DATASET_NAMES.get(dataset, ""),
            "model": model,
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "true_counts": row["true_counts"],
            "pred_counts": row["pred_counts"],
            "TN": int(row["TN"]),
            "FP": int(row["FP"]),
            "FN": int(row["FN"]),
            "TP": int(row["TP"]),
            "threshold": float(row["threshold"]),
            "threshold_source": row["threshold_source"],
            "threshold_percentile": float(row["threshold_percentile"]),
            "level": row["level"],
            "setting": row["setting"],
            "metric_file": str(metric_path),
        }
        records.append(record)

    if not records:
        raise SystemExit(f"No anomaly metrics found for run_tag={args.run_tag!r} under {results_root}")

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
        f.write("# QAR one-class anomaly detection\n\n")
        f.write("实验设置：训练和验证只使用正常类 0；测试使用保留正常类 + 全部故障类 1。")
        f.write("异常分数为重构误差，阈值来自正常验证集 99% 分位；指标按整条工况窗口/window 计算。\n\n")
        f.write(f"- run_tag: `{args.run_tag}`\n")
        f.write(f"- all metrics: `{all_csv.name}`\n\n")
        for dataset in args.datasets:
            sub = df[df["dataset"] == dataset]
            if sub.empty:
                continue
            f.write(f"## {dataset}：{DATASET_NAMES.get(dataset, '')}\n\n")
            cols = ["model", "accuracy", "precision", "recall", "f1", "true_counts", "pred_counts", "TN", "FP", "FN", "TP"]
            f.write(sub[cols].to_markdown(index=False))
            f.write("\n\n")

    xlsx = output_dir / "QAR_oneclass_anomaly_results.xlsx"
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
