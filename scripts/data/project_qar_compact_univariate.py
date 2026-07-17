#!/usr/bin/env python
"""Project QAR compact caches to a single target variable.

The QAR compact cache stores x with shape (N, T, C).  Some foundation models
are cleaner to evaluate as univariate forecasters.  This script selects one
feature channel by name or index and writes a compatible compact cache with
shape (N, T, 1), preserving labels, sources, time_keys, and masks.
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


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def pick_feature(feature_cols: list[str], target: str) -> int:
    if target.isdigit():
        idx = int(target)
        if 0 <= idx < len(feature_cols):
            return idx
        raise ValueError(f"target index {idx} out of range for {len(feature_cols)} features")
    normalized = {name.lower(): i for i, name in enumerate(feature_cols)}
    key = target.lower()
    if key in normalized:
        return normalized[key]
    # Useful aliases for QAR pressure-like variables.
    aliases = {
        "manifold_pressure": [
            "PRECOOL_PRESS1", "PRECOOL_PRESS2", "BMPS1", "BMPS2",
            "casrcac1oupresmp_01", "caslcac1oupresmp_01",
            "casrcac2oupresmp_01", "caslcac2oupresmp_01",
        ],
        "precool_press": ["PRECOOL_PRESS1", "PRECOOL_PRESS2"],
        "bmps": ["BMPS1", "BMPS2"],
        "cac_out_pressure": [
            "casrcac1oupresmp_01", "caslcac1oupresmp_01",
            "casrcac2oupresmp_01", "caslcac2oupresmp_01",
        ],
    }
    for name in aliases.get(key, []):
        if name.lower() in normalized:
            return normalized[name.lower()]
    raise ValueError(f"target {target!r} not found. Available: {feature_cols}")


def project_one(base_cache: Path, out_cache: Path, target: str) -> dict:
    data = np.load(base_cache, allow_pickle=False)
    cache = {name: data[name] for name in data.files}
    x = np.asarray(cache["x"], dtype=np.float32)
    feature_cols = cache["feature_cols"].astype(str).tolist() if "feature_cols" in cache else [f"var_{i}" for i in range(x.shape[2])]
    idx = pick_feature(feature_cols, target)
    cache["x"] = x[:, :, idx:idx + 1]
    cache["feature_cols"] = np.asarray([feature_cols[idx]])
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **cache)
    labels = np.asarray(cache.get("labels", np.zeros(x.shape[0], dtype=np.int64)), dtype=np.int64)
    return {
        "cache_file": str(out_cache),
        "source_cache": str(base_cache),
        "target": target,
        "target_index": idx,
        "target_feature": feature_cols[idx],
        "samples": int(labels.shape[0]),
        "class0": int((labels == 0).sum()),
        "class1": int((labels != 0).sum()),
        "seq_len": int(x.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target", default="manifold_pressure")
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DEFAULT_DATASETS)
    rows = []
    for dataset in datasets:
        base_cache = args.base_root / dataset / "qar_compact_shiftN80.npz"
        if not base_cache.exists():
            print(f"[skip] missing {base_cache}", flush=True)
            continue
        out_cache = args.output_root / dataset / "qar_compact_shiftN80.npz"
        print(f"project {dataset}", flush=True)
        row = project_one(base_cache, out_cache, args.target)
        row["dataset"] = dataset
        rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "univariate_manifest.csv"
    if rows:
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "univariate_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
