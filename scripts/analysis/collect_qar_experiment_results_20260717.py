#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect and audit QAR 20260717 experiment results.

The launchers under ``scripts/experiments`` write ``expected_jobs.tsv`` files.
This collector reads those manifests first, then checks the actual summary TSVs
and result files.  Missing cells are therefore explicit instead of silently
disappearing from the final workbook.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORECAST_METRICS = ["mae", "mse", "rmse", "mape", "mspe"]
MODEL_DISPLAY_NAMES = {"TiRex2": "TiRex-2"}
ANOMALY_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "macro_f1", "weighted_f1", "specificity", "auroc", "auprc",
    "true_counts", "pred_counts",
    "TN", "FP", "FN", "TP", "threshold", "threshold_source",
    "threshold_percentile", "level", "score",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_expected(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted(artifact_root.glob("server_logs/*/expected_jobs.tsv")):
        group = path.parent.name
        for row in read_tsv(path):
            row["experiment_group"] = group
            row["expected_file"] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def grab(text: str, pattern: str, default: str = "") -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_int(value: Any) -> Any:
    try:
        return int(float(value))
    except Exception:
        return ""


def parse_classification_result(result_file: Path) -> dict[str, Any]:
    text = result_file.read_text(encoding="utf-8", errors="replace") if result_file.exists() else ""
    values = {
        "acc": grab(text, r"^accuracy:([^\n]+)"),
        "accuracy": grab(text, r"^accuracy:([^\n]+)"),
        "macro_f1": grab(text, r"^macro F1:([^\n]+)"),
        "weighted_f1": grab(text, r"^weighted F1:([^\n]+)"),
        "true_counts": grab(text, r"^true counts:([^\n]+)"),
        "pred_counts": grab(text, r"^pred counts:([^\n]+)"),
        "TN": grab(text, r"^TN:([^\n]+)"),
        "FP": grab(text, r"^FP:([^\n]+)"),
        "FN": grab(text, r"^FN:([^\n]+)"),
        "TP": grab(text, r"^TP:([^\n]+)"),
    }
    raw_matrix = grab(text, r"^confusion matrix:(.+)$")
    if raw_matrix and any(values[k] == "" for k in ("TN", "FP", "FN", "TP")):
        try:
            matrix = ast.literal_eval(raw_matrix)
            if len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2:
                values["TN"], values["FP"] = int(matrix[0][0]), int(matrix[0][1])
                values["FN"], values["TP"] = int(matrix[1][0]), int(matrix[1][1])
        except Exception:
            pass
    return values


def parse_forecast_result(result_dir: Path) -> dict[str, Any]:
    metrics_path = result_dir / "metrics.npy"
    if not metrics_path.exists():
        return {name: np.nan for name in FORECAST_METRICS}
    values = np.load(metrics_path, allow_pickle=False).astype(float).tolist()
    return dict(zip(FORECAST_METRICS, values))


def parse_forecast_anomaly_result(result_dir: Path) -> dict[str, Any]:
    metrics_path = result_dir / "forecast_anomaly_metrics.csv"
    if not metrics_path.exists():
        return {name: np.nan for name in ANOMALY_METRICS}
    rows = pd.read_csv(metrics_path)
    if rows.empty:
        return {name: np.nan for name in ANOMALY_METRICS}
    row = rows.iloc[-1].to_dict()
    return {name: row.get(name, np.nan) for name in ANOMALY_METRICS}


def parse_anomaly_result(result_dir: Path) -> dict[str, Any]:
    metrics_path = result_dir / "anomaly_metrics.csv"
    if not metrics_path.exists():
        return {name: np.nan for name in ANOMALY_METRICS}
    rows = pd.read_csv(metrics_path)
    if rows.empty:
        return {name: np.nan for name in ANOMALY_METRICS}
    row = rows.iloc[-1].to_dict()
    return {name: row.get(name, np.nan) for name in ANOMALY_METRICS}


def find_result_dir(root: Path, task: str, run_tag: str, dataset: str, model: str,
                    result_dir_raw: str) -> Path | None:
    if result_dir_raw:
        candidate = (root / result_dir_raw.lstrip("./")).resolve()
        if candidate.exists():
            return candidate

    if task == "classification":
        base = root / "results"
        marker = "result_classification.txt"
    elif task == "forecast_anomaly_detection":
        base = root / "results"
        marker = "forecast_anomaly_metrics.csv"
    elif task == "anomaly_detection":
        base = root / "results"
        marker = "anomaly_metrics.csv"
    else:
        base = root / "results"
        marker = "metrics.npy"

    if not base.exists():
        return None

    dataset_token = f"_{dataset}_"
    model_token = f"_{model}_"
    candidates = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if run_tag in name and dataset_token in name and model_token in name and (path / marker).exists():
            candidates.append(path)
    if not candidates:
        for path in base.iterdir():
            if not path.is_dir():
                continue
            name = path.name
            if run_tag in name and dataset in name and model in name and (path / marker).exists():
                candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(p.name))[0].resolve()


def summary_dir_for_task(root: Path, task: str, run_tag: str) -> Path:
    if task == "classification":
        return root / "logs" / "datasetall" / run_tag
    if task == "forecast":
        return root / "logs" / "long_term_forecast" / run_tag
    if task == "zero_shot_forecast":
        return root / "logs" / "zero_shot_forecast" / run_tag
    if task == "forecast_anomaly_detection":
        return root / "logs" / "forecast_anomaly_detection" / run_tag
    if task == "anomaly_detection":
        return root / "logs" / "anomaly_detection" / run_tag
    return root / "logs" / run_tag


def collect(root: Path, expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cls_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    forecast_anomaly_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    if expected.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for _, exp in expected.iterrows():
        task = str(exp.get("task", ""))
        run_tag = str(exp.get("run_tag", ""))
        model = str(exp.get("model", ""))
        expected_datasets = [x for x in str(exp.get("datasets", "")).split() if x]
        summary_dir = summary_dir_for_task(root, task, run_tag)
        summary_rows = read_tsv(summary_dir / "summary.tsv")
        # Launchers append to summary.tsv so an interrupted/retried shard can
        # contain the same dataset/model more than once.  The last row is the
        # authoritative retry result and each requested cell must appear once
        # in the final workbook.
        latest_summary_rows: dict[tuple[str, str], dict[str, str]] = {}
        for summary_row in summary_rows:
            key = (summary_row.get("dataset", ""), summary_row.get("model", ""))
            latest_summary_rows[key] = summary_row
        summary_rows = list(latest_summary_rows.values())
        seen = {(row.get("dataset", ""), row.get("model", "")) for row in summary_rows}
        success = 0
        failed = 0

        for row in summary_rows:
            status = parse_int(row.get("status", ""))
            if status == 0:
                success += 1
            else:
                failed += 1
            result_dir_raw = str(row.get("result_dir", "")).strip()
            dataset_name = row.get("dataset", "")
            row_model = row.get("model", model)
            result_dir = find_result_dir(root, task, run_tag, dataset_name, row_model, result_dir_raw)
            result_dir_text = str(result_dir) if result_dir is not None else ""
            common = {
                "experiment_group": exp.get("experiment_group", ""),
                "task": task,
                "variant": exp.get("variant", ""),
                "patch_len": exp.get("patch_len", ""),
                "history_count": exp.get("history_count", ""),
                "target": exp.get("target", ""),
                "anchor": exp.get("anchor", ""),
                "dataset": dataset_name,
                "model": MODEL_DISPLAY_NAMES.get(row_model, row_model),
                "status": status,
                "run_tag": run_tag,
                "summary_dir": str(summary_dir),
                "result_dir": result_dir_text,
            }
            if task == "classification":
                result = parse_classification_result(result_dir / "result_classification.txt") if result_dir else {}
                cls_rows.append({**common, **result})
            elif task == "forecast":
                result = parse_forecast_result(result_dir) if result_dir else {name: np.nan for name in FORECAST_METRICS}
                forecast_rows.append({**common, **result})
            elif task == "zero_shot_forecast":
                result = parse_forecast_result(result_dir) if result_dir else {name: np.nan for name in FORECAST_METRICS}
                zero_rows.append({**common, **result})
            elif task == "forecast_anomaly_detection":
                result = parse_forecast_anomaly_result(result_dir) if result_dir else {name: np.nan for name in ANOMALY_METRICS}
                forecast_anomaly_rows.append({**common, **result})
            elif task == "anomaly_detection":
                result = parse_anomaly_result(result_dir) if result_dir else {name: np.nan for name in ANOMALY_METRICS}
                anomaly_rows.append({**common, **result})

        missing = [d for d in expected_datasets if (d, model) not in seen]
        audit_rows.append({
            "experiment_group": exp.get("experiment_group", ""),
            "task": task,
            "variant": exp.get("variant", ""),
            "patch_len": exp.get("patch_len", ""),
            "history_count": exp.get("history_count", ""),
            "target": exp.get("target", ""),
            "anchor": exp.get("anchor", ""),
            "model": MODEL_DISPLAY_NAMES.get(model, model),
            "run_tag": run_tag,
            "expected_count": len(expected_datasets),
            "summary_count": len(summary_rows),
            "success_count": success,
            "failed_count": failed,
            "missing_count": len(missing),
            "missing_datasets": " ".join(missing),
            "summary_file": str(summary_dir / "summary.tsv"),
            "expected_file": exp.get("expected_file", ""),
        })

    return (
        pd.DataFrame(cls_rows),
        pd.DataFrame(forecast_rows),
        pd.DataFrame(zero_rows),
        pd.DataFrame(anomaly_rows),
        pd.DataFrame(forecast_anomaly_rows),
        pd.DataFrame(audit_rows),
    )


def write_outputs(output_dir: Path, expected: pd.DataFrame, cls: pd.DataFrame,
                  forecast: pd.DataFrame, zero: pd.DataFrame,
                  anomaly: pd.DataFrame, forecast_anomaly: pd.DataFrame,
                  audit: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_map = {
        "expected_jobs.csv": expected,
        "all_classification_metrics.csv": cls,
        "all_forecast_metrics.csv": forecast,
        "all_zero_shot_metrics.csv": zero,
        "all_anomaly_metrics.csv": anomaly,
        "all_forecast_anomaly_metrics.csv": forecast_anomaly,
        "job_audit.csv": audit,
    }
    for name, df in csv_map.items():
        df.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    xlsx = output_dir / "QAR_extra_experiments_results_20260717.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        expected.to_excel(writer, index=False, sheet_name="expected_jobs")
        audit.to_excel(writer, index=False, sheet_name="audit")
        cls.to_excel(writer, index=False, sheet_name="classification_long")
        forecast.to_excel(writer, index=False, sheet_name="forecast_long")
        zero.to_excel(writer, index=False, sheet_name="zero_shot_long")
        anomaly.to_excel(writer, index=False, sheet_name="anomaly_long")
        forecast_anomaly.to_excel(writer, index=False, sheet_name="forecast_anomaly_long")

        incomplete = audit[(audit.get("missing_count", 0) != 0) | (audit.get("failed_count", 0) != 0)] if not audit.empty else audit
        incomplete.to_excel(writer, index=False, sheet_name="incomplete")

    return xlsx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=Path("experiment_artifacts/QAR_extra_experiments_20260717"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_artifacts/QAR_extra_experiments_20260717/collected"))
    args = parser.parse_args()

    root = args.root.resolve()
    artifact_root = (root / args.artifact_root).resolve() if not args.artifact_root.is_absolute() else args.artifact_root
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    expected = read_expected(artifact_root)
    cls, forecast, zero, anomaly, forecast_anomaly, audit = collect(root, expected)
    xlsx = write_outputs(output_dir, expected, cls, forecast, zero, anomaly, forecast_anomaly, audit)
    print(xlsx)


if __name__ == "__main__":
    main()
