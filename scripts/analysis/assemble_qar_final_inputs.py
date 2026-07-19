#!/usr/bin/env python
"""Assemble selected QAR result CSVs for the final one-sheet report.

Input order is significant: when the same task/dataset/model/variant/anchor is
present more than once, the last supplied CSV wins.  This lets a leakage-safe
rerun replace an older result without silently mixing checkpoints.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


AUGMENT_PATTERN = re.compile(r"_(aug0_(?:2000|4000|6000|10000|19119|20000))$")
ANCHOR_PATTERN = re.compile(r"predict_(\d+_\d+)")


def read_many(paths: list[Path], task: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for order, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_input_order"] = order
        frame["source_csv"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    if "dataset" not in result or "model" not in result:
        raise ValueError(f"{task} input must contain dataset and model columns")

    if "variant" not in result:
        result["variant"] = "base"
    result["variant"] = result.apply(_variant, axis=1)
    if "anchor" not in result:
        result["anchor"] = ""
    result["anchor"] = result.apply(_anchor, axis=1)

    keys = ["dataset", "model", "variant"]
    if task in {"forecast", "zero_shot"}:
        keys.append("anchor")
    result = result.sort_values("_input_order").drop_duplicates(keys, keep="last")
    return result.drop(columns=["_input_order"]).reset_index(drop=True)


def _variant(row: pd.Series) -> str:
    current = str(row.get("variant", "")).strip()
    if current and current.lower() not in {"nan", "base"}:
        return current
    match = AUGMENT_PATTERN.search(str(row.get("dataset", "")))
    return match.group(1) if match else "base"


def _anchor(row: pd.Series) -> str:
    current = str(row.get("anchor", "")).strip().replace("predict_", "")
    if current and current.lower() != "nan":
        return current
    match = ANCHOR_PATTERN.search(str(row.get("run_tag", "")))
    return match.group(1) if match else ""


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", nargs="*", type=Path, default=[])
    parser.add_argument("--forecast", nargs="*", type=Path, default=[])
    parser.add_argument("--zero-shot", nargs="*", type=Path, default=[])
    parser.add_argument("--anomaly", nargs="*", type=Path, default=[])
    parser.add_argument("--split-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_classification_metrics.csv": read_many(args.classification, "classification"),
        "all_forecast_metrics.csv": read_many(args.forecast, "forecast"),
        "all_zero_shot_metrics.csv": read_many(args.zero_shot, "zero_shot"),
        "all_anomaly_metrics.csv": read_many(args.anomaly, "anomaly"),
    }
    for name, frame in outputs.items():
        write_frame(frame, args.output_dir / name)
        print(f"{name}: {len(frame)} rows")

    if args.split_audit:
        if not args.split_audit.is_file():
            raise FileNotFoundError(args.split_audit)
        shutil.copy2(args.split_audit, args.output_dir / "split_audit.csv")


if __name__ == "__main__":
    main()
