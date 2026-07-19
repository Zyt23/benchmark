#!/usr/bin/env python
"""Audit leakage-sensitive properties of QAR per-class chronological splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def split_bounds(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = max(1, min(int(n * 0.7), n - 2))
    val_end = max(train_end + 1, min(int(n * 0.8), n - 1))
    return train_end, val_end


def dataset_key(name: str) -> tuple[int, int, str]:
    stem = name.removeprefix("dataset")
    head, _, tail = stem.partition("-")
    return int(head) if head.isdigit() else 10**9, int(tail) if tail.isdigit() else -1, name


def audit_cache(cache_path: Path) -> tuple[dict, list[dict]]:
    data = np.load(cache_path, allow_pickle=False)
    labels = data["labels"].astype(np.int64)
    sources = data["sources"].astype(str)
    time_keys = data["time_keys"].astype(np.int64)
    split_ids = {"train": [], "val": [], "test": []}
    range_rows = []

    for label in sorted(np.unique(labels).tolist()):
        ordered = np.flatnonzero(labels == label).tolist()
        ordered.sort(key=lambda i: (int(time_keys[i]), str(sources[i]), int(i)))
        train_end, val_end = split_bounds(len(ordered))
        class_splits = {
            "train": ordered[:train_end],
            "val": ordered[train_end:val_end],
            "test": ordered[val_end:],
        }
        for split, indices in class_splits.items():
            split_ids[split].extend(indices)
            keys = time_keys[indices] if indices else np.asarray([], dtype=np.int64)
            range_rows.append({
                "dataset": cache_path.parent.name,
                "label": int(label),
                "split": split,
                "samples": len(indices),
                "first_time_key": int(keys.min()) if keys.size else "",
                "last_time_key": int(keys.max()) if keys.size else "",
            })

    source_sets = {
        split: {str(sources[i]) for i in indices}
        for split, indices in split_ids.items()
    }
    test_counts = np.bincount(labels[split_ids["test"]], minlength=2)
    overlap_train_val = len(source_sets["train"] & source_sets["val"])
    overlap_train_test = len(source_sets["train"] & source_sets["test"])
    overlap_val_test = len(source_sets["val"] & source_sets["test"])
    duplicate_sources = int(len(sources) - len(set(sources.tolist())))
    unknown_time_keys = int((time_keys >= 99999999999999).sum())
    passed = (
        duplicate_sources == 0
        and unknown_time_keys == 0
        and overlap_train_val == 0
        and overlap_train_test == 0
        and overlap_val_test == 0
        and int(test_counts[0]) > 0
        and int(test_counts[1]) > 0
    )
    summary = {
        "dataset": cache_path.parent.name,
        "samples": int(labels.size),
        "class0": int((labels == 0).sum()),
        "class1": int((labels == 1).sum()),
        "train_samples": len(split_ids["train"]),
        "val_samples": len(split_ids["val"]),
        "test_samples": len(split_ids["test"]),
        "test_class0": int(test_counts[0]),
        "test_class1": int(test_counts[1]),
        "duplicate_sources": duplicate_sources,
        "unknown_time_keys": unknown_time_keys,
        "train_val_source_overlap": overlap_train_val,
        "train_test_source_overlap": overlap_train_test,
        "val_test_source_overlap": overlap_val_test,
        "passed": passed,
        "cache_file": str(cache_path),
    }
    return summary, range_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cache_paths = sorted(
        args.compact_root.glob("dataset*/qar_compact_shiftN80.npz"),
        key=lambda p: dataset_key(p.parent.name),
    )
    if not cache_paths:
        raise SystemExit(f"No compact caches found under {args.compact_root}")

    summaries, ranges = [], []
    for cache_path in cache_paths:
        summary, range_rows = audit_cache(cache_path)
        summaries.append(summary)
        ranges.extend(range_rows)

    write_csv(args.output_dir / "split_audit.csv", summaries)
    write_csv(args.output_dir / "split_time_ranges.csv", ranges)
    failed = [row["dataset"] for row in summaries if not row["passed"]]
    print(f"audited={len(summaries)} failed={failed}")
    if args.strict and failed:
        raise SystemExit("Split audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
