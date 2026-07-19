#!/usr/bin/env python
"""Audit QAR signal separability and filename-metadata shortcuts.

This is a diagnostic, not a replacement classifier.  Every probe is fitted on
the exact per-class chronological TRAIN split with fixed hyperparameters and is
evaluated on the untouched TEST split.  No test score is used for selection.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META_FIELDS = [
    "tail", "date", "month", "hour", "flight_no", "origin",
    "destination", "route", "has_tail", "has_time", "pattern",
]


def parse_source(source: str) -> dict:
    """Parse only diagnostic metadata from a QAR source filename."""
    name = Path(str(source)).name
    stem = name[:-7] if name.endswith(".tsfile") else Path(name).stem
    parts = stem.split("_")
    tail = date = time = flight_no = route = "UNKNOWN"
    pattern = "unknown"
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    compact_date_re = re.compile(r"^\d{8}$")
    compact_time_re = re.compile(r"^\d{6}$")
    time_re = re.compile(r"^\d{2}-\d{2}-\d{2}$")
    tail_re = re.compile(r"^[A-Z]-[A-Z0-9]+$")

    if len(parts) >= 5 and tail_re.match(parts[0]) and date_re.match(parts[1]) and time_re.match(parts[2]):
        tail, date, time, flight_no, route = parts[:5]
        pattern = "tail_date_time_flight_route"
    elif len(parts) >= 5 and tail_re.match(parts[0]) and compact_date_re.match(parts[2]) and compact_time_re.match(parts[3]):
        tail = parts[0]
        date = f"{parts[2][0:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
        time = f"{parts[3][0:2]}-{parts[3][2:4]}-{parts[3][4:6]}"
        flight_no = parts[4]
        pattern = "tail_archive_compactdate_time_flight"
    elif len(parts) >= 4 and date_re.match(parts[0]) and time_re.match(parts[1]):
        date, time, flight_no, route = parts[:4]
        pattern = "date_time_flight_route"
    elif len(parts) >= 3 and date_re.match(parts[0]):
        date, flight_no, route = parts[:3]
        pattern = "date_flight_route"
    elif len(parts) >= 4 and tail_re.match(parts[0]) and date_re.match(parts[1]):
        tail, date, flight_no, route = parts[:4]
        pattern = "tail_date_flight_route"
    elif len(parts) >= 2:
        flight_no, route = parts[-2:]

    origin = destination = "UNKNOWN"
    export_match = re.match(
        r"^(export-\d+)-([A-Z]-[A-Z0-9]+)-(\d{4}-\d{2}-\d{2})-([A-Z0-9]{4})-([A-Z0-9]{4})$",
        stem,
    )
    if pattern == "unknown" and export_match:
        flight_no, tail, date, origin, destination = export_match.groups()
        route = f"{origin}-{destination}"
        pattern = "export_tail_date_route"
    elif "-" in route:
        origin, destination = route.split("-", 1)

    month = "UNKNOWN"
    try:
        month = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m")
    except ValueError:
        pass
    hour = "UNKNOWN"
    if time != "UNKNOWN":
        try:
            hour = f"{int(time.split('-')[0]):02d}"
        except (ValueError, IndexError):
            pass
    return {
        "tail": tail,
        "date": date,
        "month": month,
        "hour": hour,
        "flight_no": flight_no,
        "origin": origin or "UNKNOWN",
        "destination": destination or "UNKNOWN",
        "route": route,
        "has_tail": "yes" if tail != "UNKNOWN" else "no",
        "has_time": "yes" if time != "UNKNOWN" else "no",
        "pattern": pattern,
    }


def split_bounds(n: int) -> tuple[int, int]:
    if n <= 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = max(1, min(int(n * 0.7), n - 2))
    val_end = max(train_end + 1, min(int(n * 0.8), n - 1))
    return train_end, val_end


def split_indices(labels: np.ndarray, time_keys: np.ndarray,
                  sources: np.ndarray) -> dict[str, np.ndarray]:
    result = {"train": [], "val": [], "test": []}
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label).tolist()
        indices.sort(key=lambda i: (int(time_keys[i]), str(sources[i]), int(i)))
        train_end, val_end = split_bounds(len(indices))
        result["train"].extend(indices[:train_end])
        result["val"].extend(indices[train_end:val_end])
        result["test"].extend(indices[val_end:])
    return {
        name: np.asarray(sorted(values, key=lambda i: (int(time_keys[i]), str(sources[i]), int(i))), dtype=np.int64)
        for name, values in result.items()
    }


def signal_features(x: np.ndarray, mask: np.ndarray, grid_points: int) -> np.ndarray:
    """Create a fixed, interpretable probe representation per flight."""
    mask3 = mask[:, :, None].astype(np.float32)
    denom = np.maximum(mask3.sum(axis=1), 1.0)
    mean = (x * mask3).sum(axis=1) / denom
    centered = (x - mean[:, None, :]) * mask3
    std = np.sqrt((centered * centered).sum(axis=1) / denom)
    valid_min = np.where(mask3 > 0, x, np.inf).min(axis=1)
    valid_max = np.where(mask3 > 0, x, -np.inf).max(axis=1)
    valid_min[~np.isfinite(valid_min)] = 0.0
    valid_max[~np.isfinite(valid_max)] = 0.0

    grid = np.rint(np.linspace(0, x.shape[1] - 1, grid_points)).astype(np.int64)
    sampled = x[:, grid, :].reshape(x.shape[0], -1)
    delta = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([sampled, mean, std, valid_min, valid_max, delta], axis=1).astype(np.float32)


def fixed_logistic(x_train, y_train, x_test, y_test, sparse=False) -> dict:
    scaler = StandardScaler(with_mean=not sparse)
    model = Pipeline([
        ("scale", scaler),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=600,
            class_weight="balanced",
            solver="liblinear",
            random_state=20260719,
        )),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def metadata_probe(sources: np.ndarray, labels: np.ndarray,
                   train: np.ndarray, test: np.ndarray) -> dict:
    frame = pd.DataFrame([parse_source(source) for source in sources])
    values = frame[META_FIELDS].fillna("UNKNOWN").astype(str)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pipeline = Pipeline([
        ("encode", ColumnTransformer([("metadata", encoder, META_FIELDS)])),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=600,
            class_weight="balanced",
            solver="liblinear",
            random_state=20260719,
        )),
    ])
    pipeline.fit(values.iloc[train], labels[train])
    prediction = pipeline.predict(values.iloc[test])
    score = pipeline.predict_proba(values.iloc[test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels[test], prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels[test], prediction)),
        "macro_f1": float(f1_score(labels[test], prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels[test], score)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def cohen_d(values: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = values[indices]
    y = labels[indices]
    zero = selected[y == 0]
    one = selected[y == 1]
    mean_diff = one.mean(axis=0) - zero.mean(axis=0)
    pooled = np.sqrt((zero.var(axis=0) + one.var(axis=0)) / 2.0)
    return np.divide(mean_diff, pooled, out=np.zeros_like(mean_diff), where=pooled > 1e-8)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--grid-points", type=int, default=32)
    args = parser.parse_args()

    cache_paths = sorted(args.compact_root.glob("dataset*/qar_compact_shiftN80.npz"))
    if args.datasets:
        wanted = set(args.datasets)
        cache_paths = [path for path in cache_paths if path.parent.name in wanted]
    if not cache_paths:
        raise SystemExit(f"No caches under {args.compact_root}")

    summary_rows: list[dict] = []
    sensor_rows: list[dict] = []
    for cache_path in cache_paths:
        dataset = cache_path.parent.name
        with np.load(cache_path, allow_pickle=False) as cache:
            x = np.nan_to_num(cache["x"].astype(np.float32), copy=False)
            labels = cache["labels"].astype(np.int64)
            mask = cache["mask"].astype(np.float32) if "mask" in cache else np.ones(x.shape[:2], np.float32)
            sources = cache["sources"].astype(str)
            time_keys = cache["time_keys"].astype(np.int64)
            feature_names = cache["feature_cols"].astype(str) if "feature_cols" in cache else np.asarray([f"var_{i}" for i in range(x.shape[2])])

        split = split_indices(labels, time_keys, sources)
        train, test = split["train"], split["test"]
        features = signal_features(x, mask, args.grid_points)
        signal = fixed_logistic(features[train], labels[train], features[test], labels[test])
        metadata = metadata_probe(sources, labels, train, test)

        per_flight_mean = (x * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1)[:, None], 1.0)
        train_d = cohen_d(per_flight_mean, labels, train)
        test_d = cohen_d(per_flight_mean, labels, test)
        top = np.argsort(np.abs(train_d))[::-1]
        for rank, channel in enumerate(top, 1):
            sensor_rows.append({
                "dataset": dataset,
                "rank": rank,
                "sensor": str(feature_names[channel]),
                "train_cohen_d": float(train_d[channel]),
                "test_cohen_d": float(test_d[channel]),
                "direction_consistent": bool(np.sign(train_d[channel]) == np.sign(test_d[channel])),
            })

        summary_rows.append({
            "dataset": dataset,
            "samples": int(labels.size),
            "train_samples": int(train.size),
            "test_samples": int(test.size),
            "test_class0": int((labels[test] == 0).sum()),
            "test_class1": int((labels[test] == 1).sum()),
            **{f"signal_{key}": value for key, value in signal.items()},
            **{f"metadata_{key}": value for key, value in metadata.items()},
            "top_sensor": str(feature_names[top[0]]),
            "top_sensor_train_abs_d": float(abs(train_d[top[0]])),
            "top_sensor_test_abs_d": float(abs(test_d[top[0]])),
            "top_sensor_direction_consistent": bool(np.sign(train_d[top[0]]) == np.sign(test_d[top[0]])),
        })
        print(dataset, "signal", signal, "metadata", metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "shortcut_audit_summary.csv", summary_rows)
    write_csv(args.output_dir / "sensor_effects.csv", sensor_rows)
    (args.output_dir / "README.md").write_text(
        "# QAR 时间不重叠捷径审计\n\n"
        "- 划分与正式实验一致：每个类别内部按时间 7:1:2，TRAIN/VAL/TEST 航班源文件不重叠。\n"
        "- signal probe：固定参数 Logistic Regression，仅使用传感器统计量与 32 个等距时间采样点。\n"
        "- metadata probe：固定参数 Logistic Regression，仅使用文件名解析的飞机号、日期、航班号、航线和命名格式等。\n"
        "- 两个 probe 均只在 TRAIN 拟合；TEST 不参与调参。metadata probe 高表示仍有来源/采样捷径风险，不能直接解释成物理故障信号。\n"
        "- sensor_effects.csv 比较 TRAIN/TEST 的单传感器 Cohen's d；方向一致且两边都大，才支持稳定物理差异的解释。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
