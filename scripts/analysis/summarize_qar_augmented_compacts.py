#!/usr/bin/env python
"""Summarize requested and effective normal additions in QAR compact caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_counts(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as cache:
        labels = np.asarray(cache["labels"]).reshape(-1).astype(int)
        counts = np.bincount(labels, minlength=2)
        sources = np.asarray(cache["sources"]).astype(str) if "sources" in cache else np.array([], dtype=str)
        times = np.asarray(cache["time_keys"]).reshape(-1) if "time_keys" in cache else np.array([], dtype=int)
    unique_sources = int(np.unique(sources).size) if sources.size else np.nan
    return {
        "total_samples": int(labels.size),
        "class0_count": int(counts[0]),
        "class1_count": int(counts[1]),
        "unique_sources": unique_sources,
        "duplicate_sources": int(labels.size - unique_sources) if sources.size else np.nan,
        "min_time": int(times.min()) if times.size else "",
        "max_time": int(times.max()) if times.size else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--aug-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Entries formatted as requested_count=dataset_directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_counts(args.base_cache)
    rows: list[dict[str, object]] = []
    for spec in args.variants:
        requested_text, dataset = spec.split("=", 1)
        requested = int(requested_text)
        cache_path = args.aug_root / dataset / "qar_compact_shiftN80.npz"
        current = load_counts(cache_path)
        effective = int(current["class0_count"]) - int(base["class0_count"])
        rows.append(
            {
                "dataset": "dataset12",
                "variant": dataset.removeprefix("dataset12_"),
                "requested_added_normal": requested,
                "effective_added_normal": effective,
                "rejected_or_missing": requested - effective,
                **current,
                "cache_path": str(cache_path),
            }
        )

    output = pd.DataFrame(rows).sort_values("requested_added_normal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
