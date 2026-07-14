#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write QAR compact split diagnostics for classification/forecast artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DATASETS = [
    "dataset5", "dataset6", "dataset7", "dataset8", "dataset8-1",
    "dataset9", "dataset10", "dataset11", "dataset12", "dataset13", "dataset14",
]
ANCHORS = ["predict_2_3", "predict_4_5", "predict_5_6", "predict_8_9"]
SPLITS = ["train", "val", "test"]
RATIOS = (0.7, 0.1, 0.2)


def split_bounds_keep_test(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = int(n * RATIOS[0])
    val_end = int(n * (RATIOS[0] + RATIOS[1]))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return train_end, val_end


def split_indices(labels: np.ndarray, time_keys: np.ndarray, sources: np.ndarray, strategy: str) -> dict[str, list[int]]:
    all_indices = list(range(labels.shape[0]))

    def key(i: int):
        return int(time_keys[i]), str(sources[i]), int(i)

    if strategy == "chrono":
        ordered = sorted(all_indices, key=key)
        n = len(ordered)
        train_end = int(n * RATIOS[0])
        val_end = int(n * (RATIOS[0] + RATIOS[1]))
        return {
            "train": ordered[:train_end],
            "val": ordered[train_end:val_end],
            "test": ordered[val_end:],
        }

    if strategy != "per_class_chrono":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    out = {name: [] for name in SPLITS}
    for lab in sorted(np.unique(labels).tolist()):
        group = sorted([i for i in all_indices if int(labels[i]) == int(lab)], key=key)
        train_end, val_end = split_bounds_keep_test(len(group))
        out["train"].extend(group[:train_end])
        out["val"].extend(group[train_end:val_end])
        out["test"].extend(group[val_end:])
    for name in SPLITS:
        out[name] = sorted(out[name], key=key)
    return out


def load_cache(cache_path: Path):
    cache = np.load(cache_path, allow_pickle=False)
    labels = cache["labels"].astype(np.int64) if "labels" in cache.files else np.zeros(cache["x"].shape[0], dtype=np.int64)
    if "sources" in cache.files:
        sources = cache["sources"].astype(str)
    else:
        sources = np.asarray([str(i) for i in range(labels.shape[0])])
    if "time_keys" in cache.files:
        time_keys = cache["time_keys"].astype(np.int64)
    else:
        time_keys = np.arange(labels.shape[0], dtype=np.int64)
    return labels, time_keys, sources


def summarize(task: str, dataset: str, split: str, indices: list[int], labels, time_keys, sources, cache_path: Path):
    if indices:
        labs = labels[indices]
        first_i = indices[0]
        last_i = indices[-1]
        first_time = int(time_keys[first_i])
        last_time = int(time_keys[last_i])
        first_source = str(sources[first_i])
        last_source = str(sources[last_i])
    else:
        labs = np.asarray([], dtype=np.int64)
        first_time = last_time = ""
        first_source = last_source = ""
    return {
        "task": task,
        "dataset": dataset,
        "split": split,
        "n": int(len(indices)),
        "class0": int((labs == 0).sum()),
        "class1": int((labs == 1).sum()),
        "first_time_key": first_time,
        "last_time_key": last_time,
        "first_source": first_source,
        "last_source": last_source,
        "cache_file": str(cache_path),
    }


def collect_rows(root: Path, task: str, datasets: list[str], strategy: str) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        cache_path = root / dataset / "qar_compact_shiftN80.npz"
        if not cache_path.exists():
            continue
        labels, time_keys, sources = load_cache(cache_path)
        split_map = split_indices(labels, time_keys, sources, strategy)
        for split in SPLITS:
            rows.append(summarize(task, dataset, split, split_map[split], labels, time_keys, sources, cache_path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-root", type=Path)
    parser.add_argument("--forecast-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-strategy", default="per_class_chrono", choices=["chrono", "per_class_chrono"])
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--anchors", nargs="*", default=ANCHORS)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.classification_root:
        rows.extend(collect_rows(args.classification_root, "classification", args.datasets, args.split_strategy))
    if args.forecast_root:
        for anchor in args.anchors:
            rows.extend(collect_rows(args.forecast_root / anchor, f"forecast_{anchor}", args.datasets, args.split_strategy))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "dataset", "split", "n", "class0", "class1",
        "first_time_key", "last_time_key", "first_source", "last_source", "cache_file",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
