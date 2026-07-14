#!/usr/bin/env python
"""Run ROCKET-family classifiers on QAR compact classification caches.

This script is intentionally separate from the neural-network classification
entry point because MiniROCKET/MultiROCKET are sklearn-style feature transforms
rather than torch models.  It reuses the same compact cache format and the same
per-class chronological 70/10/20 split used by ``QARFlightDatasetShift``.

Input cache:
    <compact_root>/<dataset>/qar_compact_shiftN80.npz

Expected arrays:
    x:      (N, T, C)
    labels: (N,)
    sources/time_keys optional, used only for chronological sorting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
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


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"].astype(np.float32, copy=False)
    y = cache["labels"].astype(np.int64, copy=False)
    if "time_keys" in cache.files:
        keys = cache["time_keys"].astype(np.int64, copy=False)
    else:
        keys = np.arange(y.shape[0], dtype=np.int64)
    meta = {
        "shape": list(x.shape),
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "cache_path": str(cache_path),
    }
    return x, y, keys, meta


def make_transform(model: str, n_kernels: int, n_jobs: int, random_state: int):
    from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

    if model == "MiniROCKET":
        return MiniRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    if model == "MultiROCKET":
        # Aeon MultiRocket output dimensionality is roughly n_kernels *
        # n_features_per_kernel, so keep this separately configurable by using
        # the same public n_kernels knob.
        return MultiRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    raise ValueError(f"Unsupported model: {model}")


def run_one(
    dataset: str,
    model: str,
    compact_root: Path,
    output_dir: Path,
    n_kernels: int,
    n_jobs: int,
    random_state: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict:
    cache_path = compact_root / dataset / "qar_compact_shiftN80.npz"
    x, y, keys, meta = load_cache(cache_path)
    train_idx = split_indices_per_class(y, keys, "TRAIN")
    val_idx = split_indices_per_class(y, keys, "VAL")
    test_idx = split_indices_per_class(y, keys, "TEST")

    rng = np.random.default_rng(random_state)
    if max_train_samples and len(train_idx) > max_train_samples:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_samples, replace=False))
    if max_test_samples and len(test_idx) > max_test_samples:
        test_idx = np.sort(rng.choice(test_idx, size=max_test_samples, replace=False))

    # Aeon collection transformers expect (n_cases, n_channels, n_timepoints).
    x_train = np.transpose(x[train_idx], (0, 2, 1))
    x_test = np.transpose(x[test_idx], (0, 2, 1))
    y_train = y[train_idx]
    y_test = y[test_idx]

    transform = make_transform(model, n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    clf = make_pipeline(
        transform,
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=labels).ravel()
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    true_counts = np.bincount(y_test.astype(int), minlength=2).astype(int).tolist()
    pred_counts = np.bincount(pred.astype(int), minlength=2).astype(int).tolist()

    row = {
        "dataset": dataset,
        "model": model,
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
        "n_kernels": int(n_kernels),
        "n_jobs": int(n_jobs),
        "cache_shape": json.dumps(meta["shape"]),
    }

    result_dir = output_dir / "results" / dataset / model
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_classification.txt").write_text(
        "\n".join(
            [
                f"dataset:{dataset}",
                f"model:{model}",
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
                f"n_kernels:{n_kernels}",
                f"cache_shape:{meta['shape']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return row


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact_root", type=Path, default=Path("datasetall_tsfile_compact_custom_cls_chrono_20260711"))
    parser.add_argument("--datasets", default=" ".join(DATASETS))
    parser.add_argument("--models", default="MiniROCKET MultiROCKET")
    parser.add_argument("--output_dir", type=Path, default=Path("experiment_artifacts/QAR_benchmark_matrix_20260714/rocket_classification"))
    parser.add_argument("--n_kernels", type=int, default=10000)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=20260714)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DATASETS)
    models = parse_list(args.models, ["MiniROCKET", "MultiROCKET"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_path = args.output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("dataset\tmodel\tstatus\tmessage\n")
        for dataset in datasets:
            for model in models:
                try:
                    print(f"START dataset={dataset} model={model}", flush=True)
                    row = run_one(
                        dataset=dataset,
                        model=model,
                        compact_root=args.compact_root,
                        output_dir=args.output_dir,
                        n_kernels=args.n_kernels,
                        n_jobs=args.n_jobs,
                        random_state=args.random_state,
                        max_train_samples=args.max_train_samples or None,
                        max_test_samples=args.max_test_samples or None,
                    )
                    rows.append(row)
                    summary.write(f"{dataset}\t{model}\t0\tOK\n")
                    summary.flush()
                    print(f"DONE dataset={dataset} model={model} acc={row['acc']:.6f} macro_f1={row['macro_f1']:.6f}", flush=True)
                except Exception as exc:  # keep sweep running
                    summary.write(f"{dataset}\t{model}\t1\t{type(exc).__name__}: {exc}\n")
                    summary.flush()
                    print(f"FAIL dataset={dataset} model={model}: {type(exc).__name__}: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# QAR ROCKET classification\n\n")
        f.write(f"- Compact root: `{args.compact_root}`\n")
        f.write("- Split: per-class chronological 70/10/20, matching QARFlightDatasetShift.\n")
        f.write(f"- Models: `{', '.join(models)}`\n")
        f.write(f"- n_kernels: `{args.n_kernels}`; n_jobs: `{args.n_jobs}`\n")
        f.write("- Classifier: aeon ROCKET transform + StandardScaler(with_mean=False) + RidgeClassifierCV(class_weight='balanced').\n")


if __name__ == "__main__":
    main()
