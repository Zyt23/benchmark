#!/usr/bin/env python
"""Run TabPFN on QAR compact classification caches.

TabPFN is a tabular foundation model, so QAR time-series windows are converted
to fixed-length tabular descriptors before fitting ``TabPFNClassifier``.

Split protocol:
    Same per-class chronological 70/10/20 split as QARFlightDatasetShift and
    run_qar_rocket_classifiers.py.  Only the TRAIN split is used for fitting;
    VAL is reported for traceability but not used for tuning.

Input cache:
    <compact_root>/<dataset>/qar_compact_shiftN80.npz

Expected arrays:
    x:       (N, T, C)
    mask:    (N, T)
    labels:  (N,)
    sources/time_keys optional, used only for chronological sorting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


DATASETS = [
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


def split_bounds_keep_test(n: int) -> tuple[int, int]:
    n = int(n)
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return train_end, val_end


def split_indices_per_class(labels: np.ndarray, keys: np.ndarray, flag: str) -> np.ndarray:
    flag = flag.upper()
    selected: list[int] = []
    labels = labels.astype(int, copy=False)
    for label in sorted(np.unique(labels).astype(int).tolist()):
        group = np.flatnonzero(labels == label)
        group = group[np.argsort(keys[group], kind="mergesort")]
        train_end, val_end = split_bounds_keep_test(len(group))
        if flag == "TRAIN":
            selected.extend(group[:train_end].tolist())
        elif flag in {"VAL", "VALI", "VALID", "VALIDATION"}:
            selected.extend(group[train_end:val_end].tolist())
        elif flag == "TEST":
            selected.extend(group[val_end:].tolist())
        else:
            raise ValueError(f"Unsupported split flag: {flag}")
    selected_arr = np.asarray(selected, dtype=np.int64)
    selected_arr = selected_arr[np.argsort(keys[selected_arr], kind="mergesort")]
    return selected_arr


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"].astype(np.float32, copy=False)
    y = cache["labels"].astype(np.int64, copy=False)
    if "mask" in cache.files:
        mask = cache["mask"].astype(np.float32, copy=False)
    else:
        mask = np.ones(x.shape[:2], dtype=np.float32)
    if "time_keys" in cache.files:
        keys = cache["time_keys"].astype(np.int64, copy=False)
    else:
        keys = np.arange(y.shape[0], dtype=np.int64)
    meta = {
        "shape": list(x.shape),
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "cache_path": str(cache_path),
    }
    return x, mask, y, keys, meta


def _channel_features(values: np.ndarray, time_grid: int) -> np.ndarray:
    """Convert one valid T x C window to a fixed vector."""
    if values.size == 0 or values.shape[0] == 0:
        return np.zeros((values.shape[1] if values.ndim == 2 else 0) * (11 + time_grid), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    first = values[0]
    last = values[-1]
    if values.shape[0] > 1:
        mean_abs_diff = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
        slope = (last - first) / max(values.shape[0] - 1, 1)
    else:
        mean_abs_diff = np.zeros(values.shape[1], dtype=np.float32)
        slope = np.zeros(values.shape[1], dtype=np.float32)
    stats = [
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
        np.median(values, axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
        first,
        last,
        slope,
        mean_abs_diff,
    ]
    if time_grid > 0:
        if values.shape[0] == 1:
            picks = np.zeros(time_grid, dtype=np.int64)
        else:
            picks = np.rint(np.linspace(0, values.shape[0] - 1, time_grid)).astype(np.int64)
        stats.extend([values[p] for p in picks])
    return np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1) for s in stats], axis=0)


def extract_tabular_features(x: np.ndarray, mask: np.ndarray, time_grid: int) -> np.ndarray:
    """Build NaN-free tabular descriptors from masked time-series windows."""
    n, _, c = x.shape
    per_channel_dim = 11 + int(time_grid)
    features = np.zeros((n, c * per_channel_dim + 3), dtype=np.float32)
    for i in range(n):
        valid = mask[i] > 0
        valid_count = int(valid.sum())
        if valid_count > 0:
            values = x[i, valid, :]
            channel_feat = _channel_features(values, time_grid=time_grid)
            features[i, : channel_feat.shape[0]] = channel_feat
        # Three global mask descriptors.  The neural classification code also
        # receives the padding mask, so keeping only coarse mask descriptors is
        # still comparable while avoiding filename/metadata features.
        features[i, c * per_channel_dim + 0] = valid_count / max(mask.shape[1], 1)
        features[i, c * per_channel_dim + 1] = valid_count
        features[i, c * per_channel_dim + 2] = mask.shape[1] - valid_count
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def balanced_subsample(indices: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    unique = sorted(np.unique(labels[indices]).astype(int).tolist())
    base = max_samples // max(len(unique), 1)
    remainder = max_samples - base * len(unique)
    for j, label in enumerate(unique):
        group = indices[labels[indices] == label]
        take = min(len(group), base + (1 if j < remainder else 0))
        if take < len(group):
            group = rng.choice(group, size=take, replace=False)
        parts.append(np.asarray(group, dtype=np.int64))
    out = np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)
    out = out[np.argsort(out, kind="mergesort")]
    if len(out) > max_samples:
        out = np.sort(rng.choice(out, size=max_samples, replace=False))
    return out


def run_one(
    dataset: str,
    compact_root: Path,
    output_dir: Path,
    time_grid: int,
    max_train_samples: int,
    max_test_samples: int,
    n_estimators: int,
    device: str,
    random_state: int,
    model_path: str,
) -> dict:
    from tabpfn import TabPFNClassifier

    cache_path = compact_root / dataset / "qar_compact_shiftN80.npz"
    x, mask, y, keys, meta = load_cache(cache_path)
    train_idx = split_indices_per_class(y, keys, "TRAIN")
    val_idx = split_indices_per_class(y, keys, "VAL")
    test_idx = split_indices_per_class(y, keys, "TEST")
    train_idx = balanced_subsample(train_idx, y, max_train_samples, random_state)
    test_idx = balanced_subsample(test_idx, y, max_test_samples, random_state + 17)

    x_train = extract_tabular_features(x[train_idx], mask[train_idx], time_grid=time_grid)
    x_test = extract_tabular_features(x[test_idx], mask[test_idx], time_grid=time_grid)
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    clf = TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        model_path=model_path,
        ignore_pretraining_limits=True,
        random_state=random_state,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    true_counts = np.bincount(y_test.astype(int), minlength=2).astype(int).tolist()
    pred_counts = np.bincount(pred.astype(int), minlength=2).astype(int).tolist()

    row = {
        "dataset": dataset,
        "model": "TabPFN",
        "acc": float(acc),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "true_counts": json.dumps(true_counts, ensure_ascii=False),
        "pred_counts": json.dumps(pred_counts, ensure_ascii=False),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "feature_dim": int(x_train.shape[1]),
        "time_grid": int(time_grid),
        "n_estimators": int(n_estimators),
        "max_train_samples": int(max_train_samples),
        "max_test_samples": int(max_test_samples),
        "cache_shape": json.dumps(meta["shape"]),
    }

    result_dir = output_dir / "results" / dataset / "TabPFN"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_classification.txt").write_text(
        "\n".join(
            [
                f"dataset:{dataset}",
                "model:TabPFN",
                f"acc:{acc}",
                f"accuracy:{acc}",
                f"macro_f1:{macro_f1}",
                f"weighted_f1:{weighted_f1}",
                f"true counts:{true_counts}",
                f"pred counts:{pred_counts}",
                f"confusion matrix:{[[int(tn), int(fp)], [int(fn), int(tp)]]}",
                f"train_samples:{len(train_idx)}",
                f"val_samples:{len(val_idx)}",
                f"test_samples:{len(test_idx)}",
                f"feature_dim:{x_train.shape[1]}",
                f"time_grid:{time_grid}",
                f"n_estimators:{n_estimators}",
                f"cache_shape:{meta['shape']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_root", type=Path, default=Path("datasetall_tsfile_compact_custom_cls_chrono_20260711"))
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--output_dir", type=Path, default=Path("experiment_artifacts/QAR_benchmark_matrix_20260714/classification_tabpfn"))
    parser.add_argument("--time_grid", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=4096)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_path", default="auto")
    parser.add_argument("--random_state", type=int, default=20260717)
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DATASETS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_path = args.output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("dataset\tmodel\tstatus\tmessage\n")
        for dataset in datasets:
            try:
                print(f"START dataset={dataset} model=TabPFN", flush=True)
                row = run_one(
                    dataset=dataset,
                    compact_root=args.compact_root,
                    output_dir=args.output_dir,
                    time_grid=args.time_grid,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    n_estimators=args.n_estimators,
                    device=args.device,
                    random_state=args.random_state,
                    model_path=args.model_path,
                )
                rows.append(row)
                summary.write(f"{dataset}\tTabPFN\t0\tOK\n")
                summary.flush()
                print(
                    f"DONE dataset={dataset} model=TabPFN acc={row['acc']:.6f} macro_f1={row['macro_f1']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                summary.write(f"{dataset}\tTabPFN\t1\t{type(exc).__name__}: {exc}\n")
                summary.flush()
                print(f"FAIL dataset={dataset} model=TabPFN: {type(exc).__name__}: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# QAR TabPFN Classification\n\n")
        f.write("- Model: `TabPFNClassifier` on tabular features extracted from each QAR time-series window.\n")
        f.write("- Split: per-class chronological 70/10/20; train only is used for fitting.\n")
        f.write(f"- Compact root: `{args.compact_root}`\n")
        f.write(f"- time_grid: `{args.time_grid}`; n_estimators: `{args.n_estimators}`\n")
        f.write(f"- max_train_samples: `{args.max_train_samples}`; max_test_samples: `{args.max_test_samples}`\n\n")
        if not df.empty:
            f.write(df.to_markdown(index=False))
            f.write("\n")


if __name__ == "__main__":
    main()
