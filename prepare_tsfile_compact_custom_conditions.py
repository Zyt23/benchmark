#!/usr/bin/env python
"""Build dataset-specific tsfile compact caches for dataset13 and dataset14.

This keeps the standard compact-cache contract used by QARFlightDatasetShift:

    <output_root>/<dataset>/qar_compact_shiftN80.npz

dataset13 uses anchors from ``datasetall_tsfile/build_dataset15_1.py``.
dataset14 uses anchors from ``datasetall_tsfile/320321gongkuang.py`` and
native dataset14 measurements instead of the old 16-feature QAR mapping.
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


DATASET_RE = re.compile(r'^tsfile_datasets/(dataset(?:\d+)(?:-\d+)?)_tsfile/([01])/(.+\.tsfile)$')
DATASET_MODES = {
    'dataset13': 'dataset13_anchors',
    'dataset14': 'dataset14_anchors',
}


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
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        compact_path,
        x=x,
        mask=mask,
        labels=labels,
        class_names=np.array(['0', '1']),
        feature_cols=feature_cols,
        phase_a_shift=np.array([shift], dtype=np.int64),
    )
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


def write_manifest(output_root, rows):
    path = output_root / 'tsfile_custom_condition_manifest.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = [
            'dataset', 'mode', 'samples', 'class0', 'class1',
            'seq_len', 'feature_count', 'zero_window_count',
            'zero_window_rate', 'cache_file',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset13/14 compact QAR caches with custom anchor conditions.')
    parser.add_argument('--zip_path', default='datasetall_tsfile/tsfile_datasets.zip')
    parser.add_argument('--output_root', default='datasetall_tsfile_compact')
    parser.add_argument('--work_root', default='datasetall_tsfile_work_custom_conditions')
    parser.add_argument('--iotdb_lib', default='.cache/iotdb-2.0.2-lib')
    parser.add_argument('--java_src', default='scripts/tsfile/TsFileWindowDumperAnchors.java')
    parser.add_argument('--java_class', default='TsFileWindowDumperAnchors')
    parser.add_argument('--java_class_dir', default='.cache/tsfile_java_classes')
    parser.add_argument('--datasets', nargs='*', default=['dataset13', 'dataset14'])
    parser.add_argument('--shift', type=int, default=-80)
    parser.add_argument('--max_per_class', type=int, default=0,
                        help='debug: limit files per class; 0 means all')
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

    unknown = [dataset for dataset in args.datasets if dataset not in DATASET_MODES]
    if unknown:
        raise ValueError(f'Only dataset13/dataset14 are supported here, got: {unknown}')

    entries = scan_zip(zip_path)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for dataset in args.datasets:
            mode = DATASET_MODES[dataset]
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
            with np.load(compact_path, allow_pickle=False) as cache:
                labels = cache['labels']
                x = cache['x']
                zero = (np.abs(x).sum(axis=(1, 2)) == 0)
            row = {
                'dataset': dataset,
                'mode': mode,
                'samples': int(labels.shape[0]),
                'class0': int((labels == 0).sum()),
                'class1': int((labels == 1).sum()),
                'seq_len': int(meta['seq_len']),
                'feature_count': int(meta['feature_count']),
                'zero_window_count': int(zero.sum()),
                'zero_window_rate': float(zero.mean()) if labels.shape[0] else '',
                'cache_file': str(compact_path.relative_to(output_root.parent)),
            }
            manifest_rows.append(row)
            print(f'wrote {compact_path} ({row["samples"]} samples)', flush=True)

    write_manifest(output_root, manifest_rows)
    print(output_root)


if __name__ == '__main__':
    main()
