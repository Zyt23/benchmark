#!/usr/bin/env python
"""Build QAR compact caches with extra normal CSV flights.

This is a generic wrapper around the CSV conversion logic used for LEAP normal
data.  It supports multiple zip-to-dataset mappings and writes normalx2 /
normalx4 variants:

    class0_new = class0_base + min(extra_available, (factor - 1) * class0_base)
    class1_new = class1_base

Classification and forecast segment compacts are both supported.  Forecast
segment roots should be organized as:

    <base_forecast_root>/<anchor>/<dataset>/qar_compact_shiftN80.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.prepare_leap_normal_augmented_compacts import (
    CLASSIFICATION_ANCHORS,
    FORECAST_ANCHORS,
    build_extra_cache,
    parse_list,
    write_merged,
)


def parse_specs(values: list[str]) -> list[tuple[Path, list[str]]]:
    specs: list[tuple[Path, list[str]]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"extra spec must be zip_path=datasetA,datasetB, got {value!r}")
        raw_zip, raw_datasets = value.split("=", 1)
        datasets = [x.strip() for x in raw_datasets.replace(";", ",").split(",") if x.strip()]
        if not datasets:
            raise ValueError(f"no datasets in {value!r}")
        specs.append((Path(raw_zip), datasets))
    return specs


def normal_count(cache_path: Path) -> int:
    with np.load(cache_path, allow_pickle=False) as data:
        labels = data["labels"].astype(np.int64)
        return int((labels == 0).sum())


def feature_cols(cache_path: Path) -> list[str]:
    with np.load(cache_path, allow_pickle=False) as data:
        if "feature_cols" not in data.files:
            return [f"var_{i}" for i in range(int(data["x"].shape[2]))]
        return data["feature_cols"].astype(str).tolist()


def feature_tag(cols: list[str]) -> str:
    digest = hashlib.sha1("\n".join(cols).encode("utf-8")).hexdigest()[:10]
    return f"f{len(cols)}_{digest}"


def factor_to_count(base_cache: Path, extra_count: int, factor: int) -> int:
    if factor < 2:
        raise ValueError(f"factor must be >=2, got {factor}")
    need = normal_count(base_cache) * (factor - 1)
    return int(min(need, extra_count))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra_specs", nargs="+", required=True,
                        help="zip_path=dataset5,dataset6,dataset7")
    parser.add_argument("--base_classification_root", type=Path, default=Path("datasetall_tsfile_compact_custom_cls_chrono_20260711"))
    parser.add_argument("--base_forecast_segment_root", type=Path, default=Path("datasetall_tsfile_compact_hist80_segments_20260717"))
    parser.add_argument("--classification_output_root", type=Path, default=Path("datasetall_tsfile_compact_normal_aug_cls_20260717"))
    parser.add_argument("--forecast_output_root", type=Path, default=Path("datasetall_tsfile_compact_normal_aug_hist80_segments_20260717"))
    parser.add_argument("--work_root", type=Path, default=Path("datasetall_tsfile_work_normal_aug_20260717"))
    parser.add_argument("--factors", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--tasks", nargs="+", default=["classification", "forecast"], choices=["classification", "forecast"])
    parser.add_argument("--anchors", default=" ".join(FORECAST_ANCHORS.keys()))
    args = parser.parse_args()

    specs = parse_specs(args.extra_specs)
    anchors = parse_list(args.anchors, FORECAST_ANCHORS.keys())
    args.work_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    extra_cache_by_key: dict[tuple[str, str, str], dict] = {}

    def cached_extra(zip_tag: str, zip_path: Path, anchor_key: str,
                     anchors: list[tuple[int, int, int, int]], cols: list[str]) -> dict:
        key = (zip_tag, anchor_key, tuple(cols))
        if key not in extra_cache_by_key:
            cache_path = args.work_root / f"{zip_tag}_{anchor_key}_{feature_tag(cols)}_all.npz"
            extra_cache_by_key[key] = build_extra_cache(
                zip_path,
                cache_path,
                anchors,
                max_count=None,
                feature_cols=cols,
            )
        return extra_cache_by_key[key]

    for spec_idx, (zip_path, datasets) in enumerate(specs):
        zip_tag = f"extra{spec_idx}"
        if "classification" in args.tasks:
            for dataset in datasets:
                base_cache = args.base_classification_root / dataset / "qar_compact_shiftN80.npz"
                cols = feature_cols(base_cache)
                extra = cached_extra(zip_tag, zip_path, "classification", CLASSIFICATION_ANCHORS, cols)
                extra_count = int(extra["x"].shape[0])
                for factor in args.factors:
                    take = factor_to_count(base_cache, extra_count, factor)
                    out_dataset = f"{dataset}_normalx{factor}"
                    out_path = args.classification_output_root / out_dataset / "qar_compact_shiftN80.npz"
                    print(f"classification {out_dataset}: take_extra={take}", flush=True)
                    stats = write_merged(base_cache, extra, take, out_path)
                    rows.append({
                        "task": "classification",
                        "anchor": "",
                        "dataset": dataset,
                        "out_dataset": out_dataset,
                        "factor": factor,
                        "extra_zip": str(zip_path),
                        "cache_file": str(out_path),
                        **stats,
                    })

        if "forecast" in args.tasks:
            for anchor in anchors:
                for dataset in datasets:
                    base_cache = args.base_forecast_segment_root / anchor / dataset / "qar_compact_shiftN80.npz"
                    cols = feature_cols(base_cache)
                    extra = cached_extra(zip_tag, zip_path, anchor, FORECAST_ANCHORS[anchor], cols)
                    extra_count = int(extra["x"].shape[0])
                    for factor in args.factors:
                        take = factor_to_count(base_cache, extra_count, factor)
                        out_dataset = f"{dataset}_normalx{factor}"
                        out_path = args.forecast_output_root / anchor / out_dataset / "qar_compact_shiftN80.npz"
                        print(f"forecast {anchor} {out_dataset}: take_extra={take}", flush=True)
                        stats = write_merged(base_cache, extra, take, out_path)
                        rows.append({
                            "task": "forecast_segment",
                            "anchor": anchor,
                            "dataset": dataset,
                            "out_dataset": out_dataset,
                            "factor": factor,
                            "extra_zip": str(zip_path),
                            "cache_file": str(out_path),
                            **stats,
                        })

    manifest = args.work_root / "normal_aug_manifest.csv"
    write_manifest(manifest, rows)
    (args.work_root / "normal_aug_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(manifest)


if __name__ == "__main__":
    main()
