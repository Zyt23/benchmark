#!/usr/bin/env python
"""Build QAR tsfile compact caches with custom flight-condition windows.

Output contract:

    <output_root>/<dataset>/qar_compact_shiftN80.npz

The script supports two condition/window definitions:

``anchor``
    Classification-style condition windows.  ``dataset13`` uses anchors from
    ``datasetall_tsfile/build_dataset15_1.py``.  All other datasets use anchors
    from ``datasetall_tsfile/320321gongkuang.py``; datasets 5-12/8-1 use the
    standard 16 QAR features, while dataset14 uses its native 8 features.

``phase_start80``
    Forecasting condition windows.  For each flight, take 80 points from the
    start of every flight phase 0..12, concatenate those 13 snippets, and let
    the forecasting loader run 60->20 prediction inside each 80-point segment.

Optionally, ``--dataset12_aug0_csv_zip`` appends extra class-0 CSV flights to
the original dataset12 cache and writes the combined dataset as
``dataset12_aug0``.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_RE = re.compile(r'^tsfile_datasets/(dataset(?:\d+)(?:-\d+)?)_tsfile/([01])/(.+\.tsfile)$')
ALL_TSFILE_DATASETS = [
    'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset8-1',
    'dataset9', 'dataset10', 'dataset11', 'dataset12', 'dataset13', 'dataset14',
]
STANDARD_FEATURE_NAMES = [
    'N21', 'N22', 'BMPS1', 'BMPS2',
    'PRECOOL_PRESS1', 'PRECOOL_PRESS2',
    'PRV_ENG1_R', 'PRV_ENG2_R',
    'HPV_ENG1_R', 'HPV_ENG2_R',
    'PRECOOL_TEMP1', 'PRECOOL_TEMP2',
    'PACK1_RAM_I_DR', 'PACK1_RAM_O_DR',
    'PACK2_RAM_I_DR', 'PACK2_RAM_O_DR',
]
ANCHORS_320321 = [
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
]
PHASE_START80 = [(phase, phase, 0, 80) for phase in range(13)]


def dataset_sort_key(name):
    match = re.match(r'^dataset(\d+)(?:-(\d+))?(?:_(.+))?$', name)
    if not match:
        return (10 ** 9, 10 ** 9, name)
    suffix = int(match.group(2)) if match.group(2) is not None else -1
    extra = match.group(3) or ''
    return (int(match.group(1)), suffix, extra)


def mode_for_dataset(dataset, mode_set):
    if mode_set == 'anchor':
        if dataset == 'dataset13':
            return 'dataset13_anchors'
        if dataset == 'dataset14':
            return 'dataset14_anchors'
        return 'standard_320321_anchors'

    if mode_set == 'phase_start80':
        if dataset == 'dataset13':
            return 'dataset13_phase_start80'
        if dataset == 'dataset14':
            return 'dataset14_phase_start80'
        return 'standard_phase_start80'

    raise ValueError(f'Unsupported mode_set: {mode_set}')


def anchors_for_csv(mode_set):
    if mode_set == 'anchor':
        return ANCHORS_320321
    if mode_set == 'phase_start80':
        return PHASE_START80
    raise ValueError(f'Unsupported mode_set for CSV: {mode_set}')


def scan_zip(zip_path):
    entries = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace('\\', '/')
            match = DATASET_RE.match(name)
            if not match:
                continue
            dataset, label, _ = match.groups()
            entries.setdefault(dataset, {}).setdefault(label, []).append(name)
    for dataset in entries:
        for label in entries[dataset]:
            entries[dataset][label].sort()
    return entries


def safe_extract_member(zf, member, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, output_file.open('wb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def compile_java(java_src, java_class, class_dir, lib_dir):
    class_dir.mkdir(parents=True, exist_ok=True)
    class_file = class_dir / f'{java_class}.class'
    if class_file.exists() and class_file.stat().st_mtime >= java_src.stat().st_mtime:
        return
    cmd = [
        'javac',
        '-encoding', 'UTF-8',
        '-cp', str(lib_dir / '*'),
        '-d', str(class_dir),
        str(java_src),
    ]
    subprocess.run(cmd, check=True)


def run_java(manifest, raw_out_dir, class_dir, lib_dir, java_class, mode, shift, java_xmx):
    classpath = os.pathsep.join([str(class_dir), str(lib_dir / '*')])
    cmd = [
        'java',
        f'-Xmx{java_xmx}',
        '-cp', classpath,
        java_class,
        '--manifest', str(manifest),
        '--out', str(raw_out_dir),
        '--mode', mode,
        '--shift', str(shift),
        '--skip_errors', 'true',
    ]
    subprocess.run(cmd, check=True)


def save_compact(compact_path, x, mask, labels, feature_cols, shift):
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        compact_path,
        x=x.astype(np.float32, copy=False),
        mask=mask.astype(np.float32, copy=False),
        labels=labels.astype(np.int64, copy=False),
        class_names=np.array(['0', '1']),
        feature_cols=np.asarray(feature_cols).astype(str),
        phase_a_shift=np.array([shift], dtype=np.int64),
    )


def write_npz(raw_out_dir, compact_path, shift):
    meta = json.loads((raw_out_dir / 'meta.json').read_text(encoding='utf-8'))
    n = int(meta['samples'])
    seq_len = int(meta['seq_len'])
    feature_count = int(meta['feature_count'])
    feature_cols = np.array(meta['feature_cols'])

    x = np.fromfile(raw_out_dir / 'x.bin', dtype='>f4').astype(np.float32)
    mask = np.fromfile(raw_out_dir / 'mask.bin', dtype='>f4').astype(np.float32)
    labels = np.fromfile(raw_out_dir / 'labels.bin', dtype='>i4').astype(np.int64)

    expected_x = n * seq_len * feature_count
    expected_mask = n * seq_len
    if x.size != expected_x:
        raise ValueError(f'x size mismatch, expected {expected_x}, got {x.size}')
    if mask.size != expected_mask:
        raise ValueError(f'mask size mismatch, expected {expected_mask}, got {mask.size}')
    if labels.size != n:
        raise ValueError(f'labels size mismatch, expected {n}, got {labels.size}')

    x = x.reshape(n, seq_len, feature_count)
    mask = mask.reshape(n, seq_len)
    save_compact(compact_path, x, mask, labels, feature_cols, shift)
    return meta


def build_one_dataset(zf, entries, dataset, mode, output_root, work_root, class_dir, lib_dir,
                      java_src, java_class, shift, max_per_class, keep_extracted, java_xmx):
    labels = entries.get(dataset)
    if labels is None:
        raise ValueError(f'{dataset} not found in zip')
    missing = [label for label in ('0', '1') if label not in labels]
    if missing:
        raise ValueError(f'{dataset}: missing label dirs {missing}')

    dataset_work = work_root / dataset
    extract_root = dataset_work / 'extract'
    raw_out_dir = dataset_work / 'raw'
    manifest = dataset_work / 'manifest.tsv'
    if dataset_work.exists():
        shutil.rmtree(dataset_work)
    extract_root.mkdir(parents=True)
    raw_out_dir.mkdir(parents=True)

    rows = []
    for label in ('0', '1'):
        members = labels[label]
        if max_per_class:
            members = members[:max_per_class]
        for member in members:
            rel_name = '/'.join(member.split('/')[2:])
            local_path = extract_root / rel_name
            safe_extract_member(zf, member, local_path)
            rows.append((int(label), local_path.resolve(), member))

    with manifest.open('w', encoding='utf-8', newline='') as handle:
        for label, local_path, source in rows:
            handle.write(f'{label}\t{local_path}\t{source}\n')

    compile_java(java_src, java_class, class_dir, lib_dir)
    run_java(manifest, raw_out_dir, class_dir, lib_dir, java_class, mode, shift, java_xmx)

    compact_path = output_root / dataset / f'qar_compact_shiftN{abs(shift)}.npz'
    meta = write_npz(raw_out_dir, compact_path, shift)
    shutil.copy2(raw_out_dir / 'stats.tsv', output_root / dataset / 'tsfile_conversion_stats.tsv')
    shutil.copy2(raw_out_dir / 'meta.json', output_root / dataset / 'tsfile_conversion_meta.json')

    if not keep_extracted:
        shutil.rmtree(dataset_work)
    return compact_path, meta


def column_lookup(columns):
    return {str(column).strip().lower(): column for column in columns}


def linear_fill_single_zero(values):
    arr = values.copy()
    for i in range(1, len(arr) - 1):
        if arr[i] == 0 and arr[i - 1] != 0 and arr[i + 1] != 0:
            arr[i] = (arr[i - 1] + arr[i + 1]) / 2.0
    return arr


def instance_norm(x, mask):
    valid = mask > 0
    if not np.any(valid):
        return x
    mean = x[valid].mean(axis=0)
    std = x[valid].std(axis=0) + 1e-5
    out = x.copy()
    out[valid] = (out[valid] - mean) / std
    out[~valid] = 0.0
    return out


def find_transition(phases, from_phase, to_phase):
    hits = np.flatnonzero((phases[:-1] == from_phase) & (phases[1:] == to_phase))
    if hits.size:
        return int(hits[0] + 1)
    if from_phase == 9 and to_phase == 11:
        hits = np.flatnonzero((phases[:-1] == 10) & (phases[1:] == 11))
        if hits.size:
            return int(hits[0] + 1)
    return -1


def find_phase_start(phases, phase):
    hits = np.flatnonzero(phases == phase)
    return int(hits[0]) if hits.size else -1


def csv_to_window(df, mode_set):
    lookup = column_lookup(df.columns)
    phase_col = lookup.get('flight_phase')
    if phase_col is None:
        return None, None, 'NO_PHASE', 'missing FLIGHT_PHASE'

    phase_float = pd.to_numeric(df[phase_col], errors='coerce').to_numpy(dtype=np.float64)
    phases = np.full(phase_float.shape, -9999, dtype=np.int64)
    ok_phase = np.isfinite(phase_float)
    phases[ok_phase] = np.rint(phase_float[ok_phase]).astype(np.int64)

    feature_arrays = []
    for feature in STANDARD_FEATURE_NAMES:
        col = lookup.get(feature.lower())
        if col is None:
            values = np.zeros(len(df), dtype=np.float32)
        else:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        if feature in ('N21', 'N22'):
            values = linear_fill_single_zero(values)
        feature_arrays.append(values)
    feat = np.stack(feature_arrays, axis=1).astype(np.float32)

    pieces = []
    if mode_set == 'anchor':
        segments = []
        for from_phase, to_phase, pre, post in anchors_for_csv(mode_set):
            idx = find_transition(phases, from_phase, to_phase)
            if idx < 0:
                return None, None, 'MISSING_TRANSITION', f'{from_phase}->{to_phase}'
            if idx < pre:
                return None, None, 'PRE_SHORT', f'{from_phase}->{to_phase} anchor={idx} pre={pre}'
            if len(df) - idx < post:
                return None, None, 'POST_SHORT', f'{from_phase}->{to_phase} after={len(df)-idx} post={post}'
            segments.append((idx, idx - pre, idx + post))
        for _, start, end in sorted(segments):
            pieces.append(feat[start:end])
    elif mode_set == 'phase_start80':
        seq_len = sum(post for _, _, _, post in anchors_for_csv(mode_set))
        x = np.zeros((seq_len, feat.shape[1]), dtype=np.float32)
        mask = np.zeros(seq_len, dtype=np.float32)
        offset = 0
        missing = []
        valid_any = False
        for phase, _, _, post in anchors_for_csv(mode_set):
            idx = find_phase_start(phases, phase)
            if idx < 0:
                missing.append(f'phase={phase}:missing')
                offset += post
                continue
            available = min(post, max(0, len(df) - idx))
            if available <= 0:
                missing.append(f'phase={phase}:empty')
                offset += post
                continue
            x[offset:offset + available] = feat[idx:idx + available]
            mask[offset:offset + available] = 1.0
            if available < post:
                missing.append(f'phase={phase}:short({available}/{post})')
            valid_any = True
            offset += post
        if not valid_any:
            return None, None, 'NO_VALID_PHASE_START', ';'.join(missing)
        x = instance_norm(x, mask)
        status = 'OK' if not missing else 'PARTIAL_PHASE_START'
        return x, mask, status, ';'.join(missing)
    else:
        raise ValueError(f'Unsupported CSV mode_set: {mode_set}')

    x = np.concatenate(pieces, axis=0).astype(np.float32)
    mask = np.ones(x.shape[0], dtype=np.float32)
    x = instance_norm(x, mask)
    return x, mask, 'OK', ''


def build_csv_class0_arrays(csv_zip_path, mode_set, max_csv_files=0):
    x_rows = []
    mask_rows = []
    stats = []
    with zipfile.ZipFile(csv_zip_path) as zf:
        members = sorted([name for name in zf.namelist() if name.lower().endswith('.csv')])
        if max_csv_files:
            members = members[:max_csv_files]
        for idx, member in enumerate(members):
            try:
                with zf.open(member) as handle:
                    df = pd.read_csv(handle)
                x, mask, status, message = csv_to_window(df, mode_set)
            except Exception as exc:
                x, mask, status, message = None, None, 'ERROR', f'{type(exc).__name__}: {exc}'
            if x is not None:
                x_rows.append(x)
                mask_rows.append(mask)
                written_idx = len(x_rows) - 1
            else:
                written_idx = -1
            stats.append({
                'idx': idx,
                'written_idx': written_idx,
                'label': 0,
                'status': status,
                'source': member,
                'message': message,
            })
            if (idx + 1) % 1000 == 0:
                print(f'csv converted {idx + 1} / {len(members)}, written {len(x_rows)}', flush=True)

    if not x_rows:
        raise ValueError(f'No CSV class-0 windows were built from {csv_zip_path}')
    return np.stack(x_rows), np.stack(mask_rows), stats


def write_csv_stats(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['idx', 'written_idx', 'label', 'status', 'source', 'message']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset12_aug0(output_root, mode_set, shift, csv_zip_path, max_csv_files=0):
    base_path = output_root / 'dataset12' / f'qar_compact_shiftN{abs(shift)}.npz'
    if not base_path.exists():
        raise FileNotFoundError(f'Build dataset12 before dataset12_aug0: {base_path}')

    print(f'=== dataset12_aug0 from {csv_zip_path} ===', flush=True)
    with np.load(base_path, allow_pickle=False) as base:
        base_x = base['x']
        base_mask = base['mask']
        base_labels = base['labels']
        feature_cols = base['feature_cols']

    csv_x, csv_mask, stats = build_csv_class0_arrays(csv_zip_path, mode_set, max_csv_files=max_csv_files)
    if csv_x.shape[1:] != base_x.shape[1:]:
        raise ValueError(f'CSV shape {csv_x.shape[1:]} does not match base dataset12 {base_x.shape[1:]}')

    labels = np.concatenate([base_labels, np.zeros(csv_x.shape[0], dtype=np.int64)])
    x = np.concatenate([base_x, csv_x], axis=0)
    mask = np.concatenate([base_mask, csv_mask], axis=0)

    out_dir = output_root / 'dataset12_aug0'
    compact_path = out_dir / f'qar_compact_shiftN{abs(shift)}.npz'
    save_compact(compact_path, x, mask, labels, feature_cols, shift)

    meta_path = output_root / 'dataset12' / 'tsfile_conversion_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    meta.update({
        'samples': int(labels.shape[0]),
        'mode': mode_for_dataset('dataset12', mode_set),
        'augmented_from': 'dataset12',
        'augmented_class0_csv_zip': str(csv_zip_path),
        'augmented_class0_samples': int(csv_x.shape[0]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'tsfile_conversion_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv_stats(out_dir / 'csv_append_stats.csv', stats)
    return compact_path, meta


def write_manifest(output_root, rows, mode_set):
    path = output_root / f'tsfile_{mode_set}_manifest.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = [
            'dataset', 'mode', 'samples', 'class0', 'class1',
            'seq_len', 'feature_count', 'zero_window_count',
            'zero_window_rate', 'cache_file',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_cache(output_root, dataset, compact_path, mode, meta):
    with np.load(compact_path, allow_pickle=False) as cache:
        labels = cache['labels']
        x = cache['x']
        zero = (np.abs(x).sum(axis=(1, 2)) == 0)
    return {
        'dataset': dataset,
        'mode': mode,
        'samples': int(labels.shape[0]),
        'class0': int((labels == 0).sum()),
        'class1': int((labels == 1).sum()),
        'seq_len': int(meta.get('seq_len', x.shape[1])),
        'feature_count': int(meta.get('feature_count', x.shape[2])),
        'zero_window_count': int(zero.sum()),
        'zero_window_rate': float(zero.mean()) if labels.shape[0] else '',
        'cache_file': str(compact_path.relative_to(output_root.parent)),
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare QAR compact caches with custom condition windows.')
    parser.add_argument('--zip_path', default='datasetall_tsfile/tsfile_datasets.zip')
    parser.add_argument('--output_root', default='datasetall_tsfile_compact')
    parser.add_argument('--work_root', default='datasetall_tsfile_work_custom_conditions')
    parser.add_argument('--iotdb_lib', default='.cache/iotdb-2.0.2-lib')
    parser.add_argument('--java_src', default='scripts/tsfile/TsFileWindowDumperAnchors.java')
    parser.add_argument('--java_class', default='TsFileWindowDumperAnchors')
    parser.add_argument('--java_class_dir', default='.cache/tsfile_java_classes')
    parser.add_argument('--datasets', nargs='*', default=ALL_TSFILE_DATASETS)
    parser.add_argument('--mode_set', choices=['anchor', 'phase_start80'], default='anchor')
    parser.add_argument('--dataset12_aug0_csv_zip', default='')
    parser.add_argument('--dataset12_aug_name', default='dataset12_aug0')
    parser.add_argument('--shift', type=int, default=-80)
    parser.add_argument('--max_per_class', type=int, default=0,
                        help='debug: limit tsfile files per class; 0 means all')
    parser.add_argument('--max_csv_files', type=int, default=0,
                        help='debug: limit dataset12 appended CSV files; 0 means all')
    parser.add_argument('--keep_extracted', action='store_true')
    parser.add_argument('--java_xmx', default='4g')
    args = parser.parse_args()

    zip_path = Path(args.zip_path).resolve()
    output_root = Path(args.output_root).resolve()
    work_root = Path(args.work_root).resolve()
    lib_dir = Path(args.iotdb_lib).resolve()
    java_src = Path(args.java_src).resolve()
    class_dir = Path(args.java_class_dir).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if not lib_dir.exists():
        raise FileNotFoundError(f'IoTDB lib dir not found: {lib_dir}')
    if not java_src.exists():
        raise FileNotFoundError(java_src)

    unknown = [dataset for dataset in args.datasets if dataset not in ALL_TSFILE_DATASETS]
    if unknown:
        raise ValueError(f'Unsupported datasets: {unknown}')

    entries = scan_zip(zip_path)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for dataset in sorted(args.datasets, key=dataset_sort_key):
            mode = mode_for_dataset(dataset, args.mode_set)
            print(f'=== {dataset} ({mode}) ===', flush=True)
            compact_path, meta = build_one_dataset(
                zf=zf,
                entries=entries,
                dataset=dataset,
                mode=mode,
                output_root=output_root,
                work_root=work_root,
                class_dir=class_dir,
                lib_dir=lib_dir,
                java_src=java_src,
                java_class=args.java_class,
                shift=args.shift,
                max_per_class=int(args.max_per_class),
                keep_extracted=bool(args.keep_extracted),
                java_xmx=args.java_xmx,
            )
            row = summarize_cache(output_root, dataset, compact_path, mode, meta)
            manifest_rows.append(row)
            print(f'wrote {compact_path} ({row["samples"]} samples)', flush=True)

    if args.dataset12_aug0_csv_zip:
        csv_zip_path = Path(args.dataset12_aug0_csv_zip).resolve()
        if not csv_zip_path.exists():
            raise FileNotFoundError(csv_zip_path)
        compact_path, meta = build_dataset12_aug0(
            output_root=output_root,
            mode_set=args.mode_set,
            shift=args.shift,
            csv_zip_path=csv_zip_path,
            max_csv_files=int(args.max_csv_files),
        )
        row = summarize_cache(output_root, args.dataset12_aug_name, compact_path,
                              mode_for_dataset('dataset12', args.mode_set), meta)
        manifest_rows.append(row)
        print(f'wrote {compact_path} ({row["samples"]} samples)', flush=True)

    write_manifest(output_root, manifest_rows, args.mode_set)
    print(output_root)


if __name__ == '__main__':
    main()
