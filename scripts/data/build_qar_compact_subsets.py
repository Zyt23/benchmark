#!/usr/bin/env python
"""Build deterministic QAR compact-cache subset variants.

The script works for both classification and forecast compact roots because it
only filters samples along the first dimension and preserves every npz array
whose first dimension matches N.

Examples
--------
Classification, keep both classes at 50% and 25%:

    python scripts/data/build_qar_compact_subsets.py \
      --base_root datasetall_tsfile_compact_custom_cls_chrono_20260711 \
      --output_root datasetall_tsfile_compact_scale_cls_20260717 \
      --variants both_keep50:0.5:0.5 both_keep25:0.25:0.25

Forecast anchor, normal-only downsample:

    python scripts/data/build_qar_compact_subsets.py \
      --base_root datasetall_tsfile_compact_hist80_segments_20260717/hist80_2_3 \
      --output_root datasetall_tsfile_compact_scale_forecast_20260717/hist80_2_3 \
      --variants normal_keep50:0.5:1.0 normal_keep25:0.25:1.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DATASETS = [
    "dataset5",
    "dataset6",
    "dataset7",
    "dataset8",
    "dataset8-1",
    "dataset9",
    "dataset10",
    "dataset11",
    "dataset12",
    "dataset13",
    "dataset14",
]

DEFAULT_VARIANTS = [
    "both_keep50:0.5:0.5",
    "both_keep25:0.25:0.25",
    "normal_keep50:0.5:1.0",
    "normal_keep25:0.25:1.0",
]


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def parse_variant(spec: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Variant must be name:normal_keep:fault_keep, got {spec!r}")
    name, normal_keep, fault_keep = parts
    normal_keep = float(normal_keep)
    fault_keep = float(fault_keep)
    if not (0 < normal_keep <= 1) or not (0 < fault_keep <= 1):
        raise ValueError(f"Keep ratios must be in (0,1], got {spec!r}")
    return name, normal_keep, fault_keep


def load_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def sorted_indices(labels: np.ndarray, time_keys: np.ndarray, sources: np.ndarray, wanted_label: int) -> np.ndarray:
    idx = np.flatnonzero(labels == wanted_label)
    if idx.size == 0:
        return idx
    order = sorted(idx.tolist(), key=lambda i: (int(time_keys[i]), str(sources[i]), int(i)))
    return np.asarray(order, dtype=np.int64)


def choose_keep(indices: np.ndarray, keep_ratio: float) -> np.ndarray:
    if indices.size == 0:
        return indices
    count = max(1, int(np.floor(indices.size * keep_ratio)))
    count = min(count, indices.size)
    return indices[:count]


def subset_cache(cache: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    n = int(cache["x"].shape[0])
    out: dict[str, np.ndarray] = {}
    for name, value in cache.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[name] = arr[selected]
        else:
            out[name] = arr
    return out


def build_one(base_cache: Path, out_cache: Path, variant_name: str, normal_keep: float, fault_keep: float) -> dict:
    cache = load_cache(base_cache)
    x = np.asarray(cache["x"])
    labels = np.asarray(cache.get("labels", np.zeros(x.shape[0], dtype=np.int64)), dtype=np.int64)
    sources = np.asarray(cache.get("sources", np.asarray([str(i) for i in range(x.shape[0])]))).astype(str)
    time_keys = np.asarray(cache.get("time_keys", np.arange(x.shape[0], dtype=np.int64)), dtype=np.int64)

    normal = sorted_indices(labels, time_keys, sources, 0)
    fault = np.flatnonzero(labels != 0)
    if fault.size:
        fault = np.asarray(sorted(fault.tolist(), key=lambda i: (int(time_keys[i]), str(sources[i]), int(i))), dtype=np.int64)

    selected = np.concatenate([choose_keep(normal, normal_keep), choose_keep(fault, fault_keep)])
    selected = np.asarray(sorted(selected.tolist(), key=lambda i: (int(time_keys[i]), str(sources[i]), int(i))), dtype=np.int64)
    out = subset_cache(cache, selected)

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **out)

    out_labels = np.asarray(out.get("labels", np.zeros(selected.shape[0], dtype=np.int64)), dtype=np.int64)
    return {
        "variant": variant_name,
        "base_cache": str(base_cache),
        "cache_file": str(out_cache),
        "samples": int(out_labels.shape[0]),
        "class0": int((out_labels == 0).sum()),
        "class1": int((out_labels != 0).sum()),
        "normal_keep": normal_keep,
        "fault_keep": fault_keep,
        "first_time_key": int(np.asarray(out.get("time_keys", [0]), dtype=np.int64).min()),
        "last_time_key": int(np.asarray(out.get("time_keys", [0]), dtype=np.int64).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--suffix_separator", default="_")
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DEFAULT_DATASETS)
    variants = [parse_variant(spec) for spec in args.variants]
    rows = []
    for dataset in datasets:
        base_cache = args.base_root / dataset / "qar_compact_shiftN80.npz"
        if not base_cache.exists():
            print(f"[skip] missing {base_cache}", flush=True)
            continue
        for variant_name, normal_keep, fault_keep in variants:
            out_dataset = f"{dataset}{args.suffix_separator}{variant_name}"
            out_cache = args.output_root / out_dataset / "qar_compact_shiftN80.npz"
            print(f"build {out_dataset}", flush=True)
            row = build_one(base_cache, out_cache, variant_name, normal_keep, fault_keep)
            row["dataset"] = dataset
            row["out_dataset"] = out_dataset
            rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "subset_manifest.csv"
    if rows:
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "subset_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
