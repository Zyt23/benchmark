#!/usr/bin/env python
"""Build dataset12 normal-class augmentation compact caches.

The extra package is CSV, not TsFile.  This script reuses the already-built
dataset12 custom-condition compact cache, converts the extra normal CSV flights
with the same classification anchors, and writes multiple augmented datasets:

    dataset12_aug0_2000
    dataset12_aug0_4000
    dataset12_aug0_6000
    dataset12_aug0_10000
    dataset12_aug0_20000

Each augmented cache keeps ``sources`` and ``time_keys`` so the QAR dataloader
can still apply the strict chronological 7:1:2 split.
"""

import argparse
import csv
import os
import re
import zipfile
from pathlib import Path

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

# Same standard classification anchors used by TsFileWindowDumperAnchors.
# 6->8 is intentionally excluded.
ANCHORS = [
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
SEQ_LEN = sum(pre + post for _, _, pre, post in ANCHORS)


def time_key_from_source(source):
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


def instance_norm(x, mask):
    valid = mask > 0
    out = np.zeros_like(x, dtype=np.float32)
    if valid.sum() > 1:
        mu = x[valid].mean(axis=0)
        sigma = x[valid].std(axis=0) + 1e-5
        out[valid] = (x[valid] - mu) / sigma
    return out


def convert_csv_member(zf, member):
    with zf.open(member) as handle:
        df = pd.read_csv(handle)
    if "FLIGHT_PHASE" not in df.columns:
        return None, "missing FLIGHT_PHASE"
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, "missing features: " + ",".join(missing[:5])

    df = df.fillna(0.0)
    phase = df["FLIGHT_PHASE"].to_numpy(dtype=np.int64)
    feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    pieces = []
    masks = []
    for fr, to, pre, post in ANCHORS:
        hits = np.flatnonzero((phase[:-1] == fr) & (phase[1:] == to))
        if hits.size == 0:
            return None, f"missing {fr}->{to}"
        anchor = int(hits[0] + 1)
        if anchor < pre:
            return None, f"pre_short {fr}->{to}"
        if feat.shape[0] - anchor < post:
            return None, f"post_short {fr}->{to}"
        pieces.append(feat[anchor - pre: anchor + post])
        masks.append(np.ones(pre + post, dtype=np.float32))
    x = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    mask = np.concatenate(masks, axis=0).astype(np.float32, copy=False)
    if x.shape[0] != SEQ_LEN:
        return None, f"bad_len {x.shape[0]}"
    x = instance_norm(x, mask)
    return (x, mask), "OK"


def load_or_build_extra(extra_zip, extra_cache, max_count):
    if extra_cache.exists():
        data = np.load(extra_cache, allow_pickle=False)
        if data["x"].shape[0] >= max_count:
            return data

    extra_cache.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = []
    with zipfile.ZipFile(extra_zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        members.sort(key=lambda n: (time_key_from_source(n), os.path.basename(n)))
        for member in members:
            if len(rows) >= max_count:
                break
            converted, status = convert_csv_member(zf, member)
            stats.append((member, status))
            if converted is None:
                continue
            x, mask = converted
            rows.append((x, mask, member, time_key_from_source(member)))
            if len(rows) % 500 == 0:
                print(f"converted extra csv: {len(rows)} valid / scanned {len(stats)}", flush=True)

    if not rows:
        raise RuntimeError("No valid extra CSV samples were converted")
    x = np.stack([r[0] for r in rows], axis=0).astype(np.float32)
    mask = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    sources = np.array([r[2] for r in rows])
    time_keys = np.array([r[3] for r in rows], dtype=np.int64)
    np.savez_compressed(
        extra_cache,
        x=x,
        mask=mask,
        labels=labels,
        class_names=np.array(["0", "1"]),
        feature_cols=np.array(FEATURE_COLS),
        phase_a_shift=np.array([-80], dtype=np.int64),
        sources=sources,
        time_keys=time_keys,
    )
    with (extra_cache.parent / "extra_conversion_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "status"])
        writer.writerows(stats)
    print(f"wrote extra cache {extra_cache}: {x.shape[0]} valid", flush=True)
    return np.load(extra_cache, allow_pickle=False)


def write_augmented(base_cache, extra_data, output_root, counts):
    base = np.load(base_cache, allow_pickle=False)
    base_x = base["x"].astype(np.float32, copy=False)
    base_mask = base["mask"].astype(np.float32, copy=False)
    base_labels = base["labels"].astype(np.int64, copy=False)
    base_sources = base["sources"].astype(str) if "sources" in base.files else np.array([f"base_{i}" for i in range(base_x.shape[0])])
    base_time_keys = base["time_keys"].astype(np.int64, copy=False) if "time_keys" in base.files else np.array([time_key_from_source(s) for s in base_sources], dtype=np.int64)

    extra_x = extra_data["x"].astype(np.float32, copy=False)
    extra_mask = extra_data["mask"].astype(np.float32, copy=False)
    extra_labels = extra_data["labels"].astype(np.int64, copy=False)
    extra_sources = extra_data["sources"].astype(str)
    extra_time_keys = extra_data["time_keys"].astype(np.int64, copy=False)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for count in counts:
        if count > extra_x.shape[0]:
            raise ValueError(f"Requested {count} extra samples, only {extra_x.shape[0]} valid")
        dataset = f"dataset12_aug0_{count}"
        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "qar_compact_shiftN80.npz"
        x = np.concatenate([base_x, extra_x[:count]], axis=0)
        mask = np.concatenate([base_mask, extra_mask[:count]], axis=0)
        labels = np.concatenate([base_labels, extra_labels[:count]], axis=0)
        sources = np.concatenate([base_sources, extra_sources[:count]])
        time_keys = np.concatenate([base_time_keys, extra_time_keys[:count]])
        order = np.lexsort((sources.astype(str), time_keys))
        # Keep the cache itself in chronological order.  The dataloader will
        # sort again, but this makes inspection easier and deterministic.
        x = x[order]
        mask = mask[order]
        labels = labels[order]
        sources = sources[order]
        time_keys = time_keys[order]
        np.savez_compressed(
            out_path,
            x=x,
            mask=mask,
            labels=labels,
            class_names=np.array(["0", "1"]),
            feature_cols=np.array(FEATURE_COLS),
            phase_a_shift=np.array([-80], dtype=np.int64),
            sources=sources,
            time_keys=time_keys,
        )
        manifest_rows.append({
            "dataset": dataset,
            "extra_count": count,
            "samples": int(labels.shape[0]),
            "class0": int((labels == 0).sum()),
            "class1": int((labels == 1).sum()),
            "first_time_key": int(time_keys[0]),
            "last_time_key": int(time_keys[-1]),
            "cache_file": str(out_path),
        })
        print(f"wrote {out_path}: {labels.shape[0]} samples", flush=True)

    with (output_root / "dataset12_aug0_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache", required=True)
    parser.add_argument("--extra_zip", required=True)
    parser.add_argument("--output_root", default="datasetall_tsfile_compact_dataset12_aug0_chrono_20260712")
    parser.add_argument("--extra_cache", default="")
    parser.add_argument("--counts", nargs="+", type=int, default=[2000, 4000, 6000, 10000, 20000])
    args = parser.parse_args()

    output_root = Path(args.output_root)
    extra_cache = Path(args.extra_cache) if args.extra_cache else output_root / "extra0_valid_20000.npz"
    max_count = max(args.counts)
    extra_data = load_or_build_extra(Path(args.extra_zip), extra_cache, max_count)
    write_augmented(Path(args.base_cache), extra_data, output_root, args.counts)


if __name__ == "__main__":
    main()
