#!/usr/bin/env python
"""Build fixed-length CSV windows from per-flight CSV files using phase anchors.

Input layout:
    <src_root>/0/*.csv
    <src_root>/1/*.csv

Output layout:
    <dst_root>/0/*.csv
    <dst_root>/1/*.csv

Each output file concatenates windows around the first occurrence of each
configured FLIGHT_PHASE transition.  Flights missing any required anchor, or
not long enough for the requested pre/post window, are skipped and recorded in
``skip_log.tsv``.

Presets:
    dataset15_1  - anchors from datasetall_tsfile/build_dataset15_1.py
    320321       - anchors from datasetall_tsfile/320321gongkuang.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ANCHOR_PRESETS = {
    "dataset15_1": [
        (0, 1, 30, 100),
        (1, 2, 30, 80),
        (2, 3, 30, 80),
        (3, 4, 30, 80),
        (4, 5, 100, 500),
        (5, 6, 200, 200),
        (8, 9, 200, 300),
        (9, 10, 200, 300),
        (10, 11, 80, 80),
        (11, 12, 80, 80),
        (12, 13, 80, 60),
    ],
    "320321": [
        (0, 1, 30, 100),
        (1, 2, 30, 80),
        (2, 3, 30, 30),
        (4, 5, 30, 500),
        (5, 6, 200, 200),
        (6, 8, 200, 300),
        (8, 9, 200, 250),
        (9, 11, 200, 80),
        (11, 12, 5, 40),
        (12, 13, 30, 200),
    ],
}


def find_transition(phase: np.ndarray, from_phase: int, to_phase: int) -> int | None:
    hits = np.where((phase[:-1] == from_phase) & (phase[1:] == to_phase))[0]
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def process_file(csv_path: Path, out_path: Path, anchors, phase_col: str):
    df = pd.read_csv(csv_path)
    if phase_col not in df.columns:
        return False, f"missing_phase_col:{phase_col}"

    phase = df[phase_col].to_numpy()
    length = len(df)
    windows = []
    for from_phase, to_phase, pre, post in anchors:
        anchor = find_transition(phase, from_phase, to_phase)
        if anchor is None:
            return False, f"missing_transition:{from_phase}->{to_phase}"
        if anchor < pre:
            return False, f"pre_short:{from_phase}->{to_phase}:anchor={anchor}:pre={pre}"
        if length - anchor < post:
            return False, f"post_short:{from_phase}->{to_phase}:remain={length - anchor}:post={post}"
        windows.append((anchor, pre, post))

    windows.sort(key=lambda item: item[0])
    pieces = [df.iloc[anchor - pre: anchor + post] for anchor, pre, post in windows]
    out_df = pd.concat(pieces, axis=0, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return True, str(len(out_df))


def main():
    parser = argparse.ArgumentParser(description="Build fixed QAR phase-anchor CSV datasets.")
    parser.add_argument("--src_root", required=True, help="Input root with 0/ and 1/ label folders.")
    parser.add_argument("--dst_root", required=True, help="Output root.")
    parser.add_argument("--preset", choices=sorted(ANCHOR_PRESETS), default="dataset15_1")
    parser.add_argument("--phase_col", default="FLIGHT_PHASE")
    parser.add_argument("--labels", nargs="*", default=["0", "1"])
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    anchors = ANCHOR_PRESETS[args.preset]
    expected_len = sum(pre + post for _, _, pre, post in anchors)

    skip_log = dst_root / "skip_log.tsv"
    dst_root.mkdir(parents=True, exist_ok=True)
    with skip_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "file", "ok", "message"])
        for label in args.labels:
            files = sorted(glob.glob(str(src_root / label / "*.csv")))
            print(f"[label {label}] files={len(files)} expected_len={expected_len}")
            ok_count = 0
            for file_name in tqdm(files, desc=f"label {label}"):
                csv_path = Path(file_name)
                out_path = dst_root / label / csv_path.name
                ok, message = process_file(csv_path, out_path, anchors, args.phase_col)
                writer.writerow([label, str(csv_path), int(ok), message])
                ok_count += int(ok)
            print(f"[label {label}] ok={ok_count} skipped={len(files) - ok_count}")


if __name__ == "__main__":
    main()

