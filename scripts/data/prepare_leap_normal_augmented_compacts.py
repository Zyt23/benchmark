#!/usr/bin/env python
"""Prepare LEAP normal-data augmentation compacts for dataset9/10/12.

The extra normal package is CSV-based and can be used as class-0 data for:

    dataset9  - 321 HPV fault (LEAP)
    dataset10 - 321 pressure-line fault (LEAP)
    dataset12 - 321 PRV fault (LEAP)

This script builds two kinds of augmented compact caches:

1. classification/anomaly caches:
   <classification_output_root>/<dataset>_aug0_<count>/qar_compact_shiftN80.npz

2. forecast segment caches for history80 construction:
   <forecast_output_root>/<anchor>/<dataset>_aug0_<count>/qar_compact_shiftN80.npz

The actual history-context forecast caches can then be built by
``scripts/long_term_forecast/build_qar_history_forecast_compacts.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "N21", "N22", "BMPS1", "BMPS2",
    "PRECOOL_PRESS1", "PRECOOL_PRESS2",
    "PRV_ENG1_R", "PRV_ENG2_R",
    "HPV_ENG1_R", "HPV_ENG2_R",
    "PRECOOL_TEMP1", "PRECOOL_TEMP2",
    "PACK1_RAM_I_DR", "PACK1_RAM_O_DR",
    "PACK2_RAM_I_DR", "PACK2_RAM_O_DR",
]

CLASSIFICATION_ANCHORS = [
    (0, 1, 30, 100),
    (1, 2, 30, 80),
    (2, 3, 30, 30),
    (4, 5, 30, 500),
    (5, 6, 200, 200),
    (8, 9, 200, 250),
    (9, 11, 200, 80),
    (11, 12, 5, 40),
    (12, 13, 30, 200),
]

FORECAST_ANCHORS = {
    "hist80_2_3": [(2, 3, 40, 40)],
    "hist80_4_5": [(4, 5, 40, 40)],
    "hist80_5_6": [(5, 6, 40, 40)],
    "hist80_8_9": [(8, 9, 40, 40)],
}

DEFAULT_DATASETS = ["dataset9", "dataset10", "dataset12"]


def time_key_from_source(source: str) -> int:
    base = os.path.basename(str(source).replace("\\", "/"))
    candidates = []

    def add(year, month, day, hour=0, minute=0, second=0):
        try:
            year = int(year)
            month = int(month)
            day = int(day)
            hour = int(hour)
            minute = int(minute)
            second = int(second)
        except Exception:
            return
        if 2000 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            candidates.append((year, month, day, hour, minute, second))

    for m in re.finditer(r"(20\d{2})[-_](\d{2})[-_](\d{2})[ T_](\d{2})[-_:](\d{2})[-_:](\d{2})", base):
        add(*m.groups())
    for m in re.finditer(r"(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})", base):
        add(*m.groups())
    for m in re.finditer(r"(20\d{2})[-_](\d{2})[-_](\d{2})", base):
        add(*m.groups())
    for m in re.finditer(r"(20\d{2})(\d{2})(\d{2})", base):
        add(*m.groups())

    if not candidates:
        return 99999999999999
    y, mo, d, h, mi, s = min(candidates)
    return int(f"{y:04d}{mo:02d}{d:02d}{h:02d}{mi:02d}{s:02d}")


def instance_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    out = np.zeros_like(x, dtype=np.float32)
    if valid.sum() > 1:
        mu = x[valid].mean(axis=0)
        sigma = x[valid].std(axis=0) + 1e-5
        out[valid] = (x[valid] - mu) / sigma
    return out


def convert_df(df: pd.DataFrame, anchors: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if "FLIGHT_PHASE" not in df.columns:
        raise ValueError("missing FLIGHT_PHASE")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError("missing features: " + ",".join(missing[:8]))

    df = df.fillna(0.0)
    phase = df["FLIGHT_PHASE"].to_numpy(dtype=np.int64)
    feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    pieces = []
    masks = []
    for fr, to, pre, post in anchors:
        hits = np.flatnonzero((phase[:-1] == fr) & (phase[1:] == to))
        if hits.size == 0:
            raise ValueError(f"missing {fr}->{to}")
        center = int(hits[0] + 1)
        if center < pre:
            raise ValueError(f"pre_short {fr}->{to}")
        if feat.shape[0] - center < post:
            raise ValueError(f"post_short {fr}->{to}")
        pieces.append(feat[center - pre:center + post])
        masks.append(np.ones(pre + post, dtype=np.float32))
    x = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    mask = np.concatenate(masks, axis=0).astype(np.float32, copy=False)
    x = instance_norm(x, mask)
    return x, mask


def parse_count_token(token: str, valid_count: int) -> int:
    token_l = str(token).strip().lower()
    if token_l in {"all", "全部", "-1"}:
        return int(valid_count)
    return int(token)


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def build_extra_cache(extra_zip: Path, cache_path: Path, anchors: list[tuple[int, int, int, int]], max_count: int | None) -> dict:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if max_count is None or data["x"].shape[0] >= max_count:
            return {name: data[name] for name in data.files}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = []
    with zipfile.ZipFile(extra_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        members.sort(key=lambda n: (time_key_from_source(n), os.path.basename(n)))
        for member in members:
            if max_count is not None and len(rows) >= max_count:
                break
            try:
                with zf.open(member) as handle:
                    df = pd.read_csv(handle)
                x, mask = convert_df(df, anchors)
                rows.append((x, mask, member, time_key_from_source(member)))
                status = "OK"
            except Exception as exc:
                status = f"{type(exc).__name__}: {exc}"
            stats.append((member, status))
            if len(rows) and len(rows) % 500 == 0:
                print(f"{cache_path.name}: converted {len(rows)} valid / scanned {len(stats)}", flush=True)

    if not rows:
        raise RuntimeError(f"No valid extra CSV samples converted for {cache_path}")
    x = np.stack([r[0] for r in rows], axis=0).astype(np.float32)
    mask = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    sources = np.asarray([r[2] for r in rows])
    time_keys = np.asarray([r[3] for r in rows], dtype=np.int64)
    np.savez_compressed(
        cache_path,
        x=x,
        mask=mask,
        labels=labels,
        class_names=np.asarray(["0", "1"]),
        feature_cols=np.asarray(FEATURE_COLS),
        phase_a_shift=np.asarray([-80], dtype=np.int64),
        sources=sources,
        time_keys=time_keys,
    )
    with (cache_path.parent / f"{cache_path.stem}_conversion_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "status"])
        writer.writerows(stats)
    return {
        "x": x,
        "mask": mask,
        "labels": labels,
        "class_names": np.asarray(["0", "1"]),
        "feature_cols": np.asarray(FEATURE_COLS),
        "phase_a_shift": np.asarray([-80], dtype=np.int64),
        "sources": sources,
        "time_keys": time_keys,
    }


def load_base(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {name: data[name] for name in data.files}
    if "sources" not in out:
        out["sources"] = np.asarray([f"base_{i}" for i in range(out["x"].shape[0])])
    if "time_keys" not in out:
        out["time_keys"] = np.asarray([time_key_from_source(s) for s in out["sources"]], dtype=np.int64)
    return out


def write_merged(base_cache: Path, extra: dict, count: int, out_path: Path) -> dict:
    base = load_base(base_cache)
    take = min(count, int(extra["x"].shape[0]))
    x = np.concatenate([base["x"].astype(np.float32), extra["x"][:take].astype(np.float32)], axis=0)
    mask = np.concatenate([base["mask"].astype(np.float32), extra["mask"][:take].astype(np.float32)], axis=0)
    labels = np.concatenate([base["labels"].astype(np.int64), extra["labels"][:take].astype(np.int64)], axis=0)
    sources = np.concatenate([base["sources"].astype(str), extra["sources"][:take].astype(str)])
    time_keys = np.concatenate([base["time_keys"].astype(np.int64), extra["time_keys"][:take].astype(np.int64)])
    order = np.lexsort((sources, time_keys))
    x = x[order]
    mask = mask[order]
    labels = labels[order]
    sources = sources[order]
    time_keys = time_keys[order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x,
        mask=mask,
        labels=labels,
        class_names=np.asarray(["0", "1"]),
        feature_cols=np.asarray(FEATURE_COLS),
        phase_a_shift=np.asarray([-80], dtype=np.int64),
        sources=sources,
        time_keys=time_keys,
    )
    return {
        "samples": int(labels.shape[0]),
        "class0": int((labels == 0).sum()),
        "class1": int((labels == 1).sum()),
        "extra_count": int(take),
        "first_time_key": int(time_keys[0]),
        "last_time_key": int(time_keys[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra_zip", type=Path, required=True)
    parser.add_argument("--base_classification_root", type=Path, default=Path("datasetall_tsfile_compact_custom_cls_chrono_20260711"))
    parser.add_argument("--base_forecast_segment_root", type=Path, default=Path("datasetall_tsfile_compact_hist80_segments_20260717"))
    parser.add_argument("--classification_output_root", type=Path, default=Path("datasetall_tsfile_compact_leap_aug_cls_20260717"))
    parser.add_argument("--forecast_output_root", type=Path, default=Path("datasetall_tsfile_compact_leap_aug_hist80_segments_20260717"))
    parser.add_argument("--work_root", type=Path, default=Path("datasetall_tsfile_work_leap_aug_20260717"))
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    parser.add_argument("--counts", nargs="+", default=["1000", "2000", "4000", "all"])
    parser.add_argument("--tasks", nargs="+", default=["classification", "forecast"], choices=["classification", "forecast"])
    parser.add_argument("--anchors", default=" ".join(FORECAST_ANCHORS.keys()))
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DEFAULT_DATASETS)
    anchors = parse_list(args.anchors, FORECAST_ANCHORS.keys())
    args.work_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    if "classification" in args.tasks:
        # Convert all valid extra classification samples once so "all" is exact.
        cls_extra = build_extra_cache(
            args.extra_zip,
            args.work_root / "extra_classification_all.npz",
            CLASSIFICATION_ANCHORS,
            max_count=None,
        )
        counts = [parse_count_token(c, int(cls_extra["x"].shape[0])) for c in args.counts]
        for dataset in datasets:
            base_cache = args.base_classification_root / dataset / "qar_compact_shiftN80.npz"
            for count in counts:
                out_dataset = f"{dataset}_aug0_{count}"
                out_path = args.classification_output_root / out_dataset / "qar_compact_shiftN80.npz"
                stats = write_merged(base_cache, cls_extra, count, out_path)
                manifest_rows.append({
                    "task": "classification",
                    "anchor": "",
                    "dataset": dataset,
                    "aug_dataset": out_dataset,
                    "cache_file": str(out_path),
                    **stats,
                })
                print(f"wrote {out_path}", flush=True)

    if "forecast" in args.tasks:
        for anchor in anchors:
            extra = build_extra_cache(
                args.extra_zip,
                args.work_root / f"extra_{anchor}_all.npz",
                FORECAST_ANCHORS[anchor],
                max_count=None,
            )
            counts = [parse_count_token(c, int(extra["x"].shape[0])) for c in args.counts]
            for dataset in datasets:
                base_cache = args.base_forecast_segment_root / anchor / dataset / "qar_compact_shiftN80.npz"
                for count in counts:
                    out_dataset = f"{dataset}_aug0_{count}"
                    out_path = args.forecast_output_root / anchor / out_dataset / "qar_compact_shiftN80.npz"
                    stats = write_merged(base_cache, extra, count, out_path)
                    manifest_rows.append({
                        "task": "forecast_segment",
                        "anchor": anchor,
                        "dataset": dataset,
                        "aug_dataset": out_dataset,
                        "cache_file": str(out_path),
                        **stats,
                    })
                    print(f"wrote {out_path}", flush=True)

    manifest_path = args.work_root / "leap_aug_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.work_root / "leap_aug_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
