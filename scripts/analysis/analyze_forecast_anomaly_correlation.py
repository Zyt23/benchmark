#!/usr/bin/env python
"""Measure whether better QAR forecasting aligns with better anomaly detection.

The forecast model is compared with the same architecture, dataset and phase
transition after it is trained on normal flights and scored by forecast error.
All anomaly thresholds are selected on validation data; this script never
retunes anything on TEST.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["Transformer", "TimesNet", "PatchTST", "DLinear", "iTransformer"]
ANOMALY_METRICS = [
    "balanced_accuracy", "f1", "macro_f1", "recall", "auroc", "auprc"
]


def normalize_anchor(value: object) -> str:
    text = str(value).strip().lower().replace("predict_", "")
    return text.replace("->", "_").replace("→", "_")


def latest_success(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    work = df.copy()
    if "status" in work:
        status = pd.to_numeric(work["status"], errors="coerce")
        successful = work[status.eq(0)]
        if not successful.empty:
            work = successful
    return work.drop_duplicates(keys, keep="last")


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return float("nan")
    return float(left[valid].rank().corr(right[valid].rank()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--anomaly-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast_csv)
    anomaly = pd.read_csv(args.anomaly_csv)
    forecast = forecast[forecast["model"].isin(MODELS)].copy()
    anomaly = anomaly[anomaly["model"].isin(MODELS)].copy()
    forecast["anchor_key"] = forecast["anchor"].map(normalize_anchor)
    anomaly["anchor_key"] = anomaly["anchor"].map(normalize_anchor)

    keys = ["dataset", "anchor_key", "model"]
    forecast = latest_success(forecast, keys)
    anomaly = latest_success(anomaly, keys)
    merged = forecast[keys + ["mae", "mse", "rmse"]].merge(
        anomaly[keys + [c for c in ANOMALY_METRICS if c in anomaly]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged["forecast_quality"] = -pd.to_numeric(merged["mse"], errors="coerce")

    rows = []
    groups = [("global", merged)]
    groups.extend((f"anchor_{anchor}", part) for anchor, part in merged.groupby("anchor_key"))
    model_mean = merged.groupby("model", as_index=False).mean(numeric_only=True)
    groups.append(("model_mean", model_mean))
    for group_name, part in groups:
        for metric in ANOMALY_METRICS:
            if metric not in part:
                continue
            x = pd.to_numeric(part["forecast_quality"], errors="coerce")
            y = pd.to_numeric(part[metric], errors="coerce")
            valid = x.notna() & y.notna()
            rows.append({
                "group": group_name,
                "anomaly_metric": metric,
                "n": int(valid.sum()),
                "pearson_forecast_quality": float(x[valid].corr(y[valid])) if valid.sum() >= 3 else np.nan,
                "spearman_forecast_quality": rank_correlation(x, y),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "forecast_anomaly_matched_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(
        args.output_dir / "forecast_anomaly_correlations.csv", index=False, encoding="utf-8-sig")
    print("matched rows:", len(merged))


if __name__ == "__main__":
    main()
