#!/usr/bin/env python
"""Build QAR compact classification caches from tsfile dataset zip.

The output cache format matches ``QARFlightDatasetShift``:

    <output_root>/<dataset>/qar_compact_shiftN80.npz

Each cache contains x, mask, labels, class_names, feature_cols and
phase_a_shift. Training can therefore reuse ``--data QAR_shift`` and the
existing datasetall classification sweep script by pointing COMPACT_ROOT to
the output root.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


DATASET_RE = re.compile(r'^tsfile_datasets/(dataset(?:\d+)(?:-\d+)?)_tsfile/([01])/(.+\.tsfile)$')
FEATURE_COLS = np.array([
    'N21', 'N22', 'BMPS1', 'BMPS2',
    'PRECOOL_PRESS1', 'PRECOOL_PRESS2',
    'PRV_ENG1_R', 'PRV_ENG2_R',
    'HPV_ENG1_R', 'HPV_ENG2_R',
    'PRECOOL_TEMP1', 'PRECOOL_TEMP2',
    'PACK1_RAM_I_DR', 'PACK1_RAM_O_DR',
    'PACK2_RAM_I_DR', 'PACK2_RAM_O_DR',
])


def dataset_sort_key(name):
    match = re.match(r'^dataset(\d+)(?:-(\d+))?$', name)
    if not match:
        return (10 ** 9, 10 ** 9, name)
    suffix = int(match.group(2)) if match.group(2) is not None else -1
    return (int(match.group(1)), suffix, name)


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


def compile_java(java_src, class_dir, lib_dir):
    class_dir.mkdir(parents=True, exist_ok=True)
    class_file = class_dir / 'TsFileWindowDumper.class'
    if class_file.exists() and class_file.stat().st_mtime >= java_src.stat().st_mtime:
        return
    classpath = str(lib_dir / '*')
    cmd = ['javac', '-encoding', 'UTF-8', '-cp', classpath, '-d', str(class_dir), str(java_src)]
    subprocess.run(cmd, check=True)


def run_java(manifest, raw_out_dir, class_dir, lib_dir, shift, java_xmx):
    classpath = os.pathsep.join([str(class_dir), str(lib_dir / '*')])
    cmd = [
        'java',
        f'-Xmx{java_xmx}',
        '-cp', classpath,
        'TsFileWindowDumper',
        '--manifest', str(manifest),
        '--out', str(raw_out_dir),
        '--shift', str(shift),
        '--skip_errors', 'true',
    ]
    subprocess.run(cmd, check=True)


def write_npz(raw_out_dir, compact_path, dataset, shift):
    meta = json.loads((raw_out_dir / 'meta.json').read_text(encoding='utf-8'))
    n = int(meta['samples'])
    seq_len = int(meta['seq_len'])
    feature_count = int(meta['feature_count'])
    x = np.fromfile(raw_out_dir / 'x.bin', dtype='>f4').astype(np.float32)
    mask = np.fromfile(raw_out_dir / 'mask.bin', dtype='>f4').astype(np.float32)
    labels = np.fromfile(raw_out_dir / 'labels.bin', dtype='>i4').astype(np.int64)

    expected_x = n * seq_len * feature_count
    expected_mask = n * seq_len
    if x.size != expected_x:
        raise ValueError(f'{dataset}: x size mismatch, expected {expected_x}, got {x.size}')
    if mask.size != expected_mask:
        raise ValueError(f'{dataset}: mask size mismatch, expected {expected_mask}, got {mask.size}')
    if labels.size != n:
        raise ValueError(f'{dataset}: labels size mismatch, expected {n}, got {labels.size}')

    x = x.reshape(n, seq_len, feature_count)
    mask = mask.reshape(n, seq_len)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        compact_path,
        x=x,
        mask=mask,
        labels=labels,
        class_names=np.array(['0', '1']),
        feature_cols=FEATURE_COLS,
        phase_a_shift=np.array([shift], dtype=np.int64),
    )


def build_one_dataset(zf, zip_entries, dataset, output_root, work_root, class_dir, lib_dir,
                      java_src, shift, max_per_class, keep_extracted, java_xmx):
    labels = zip_entries[dataset]
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

    compile_java(java_src, class_dir, lib_dir)
    run_java(manifest, raw_out_dir, class_dir, lib_dir, shift, java_xmx)

    compact_path = output_root / dataset / f'qar_compact_shiftN{abs(shift)}.npz'
    write_npz(raw_out_dir, compact_path, dataset, shift)
    shutil.copy2(raw_out_dir / 'stats.tsv', output_root / dataset / 'tsfile_conversion_stats.tsv')
    with np.load(compact_path, allow_pickle=False) as cache:
        written_labels = cache['labels']
        written_samples = int(written_labels.shape[0])

    if not keep_extracted:
        shutil.rmtree(dataset_work)
    return compact_path, written_samples


def write_manifest(output_root, manifest_rows):
    path = output_root / 'tsfile_compact_manifest.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['dataset', 'samples', 'class0', 'class1', 'cache_file']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def main():
    parser = argparse.ArgumentParser(description='Prepare compact QAR caches from tsfile dataset zip.')
    parser.add_argument('--zip_path', default='datasetall_tsfile/tsfile_datasets.zip')
    parser.add_argument('--output_root', default='datasetall_tsfile_compact')
    parser.add_argument('--work_root', default='datasetall_tsfile_work')
    parser.add_argument('--iotdb_lib', default='.cache/iotdb-2.0.2-lib')
    parser.add_argument('--java_src', default='scripts/tsfile/TsFileWindowDumper.java')
    parser.add_argument('--java_class_dir', default='.cache/tsfile_java_classes')
    parser.add_argument('--datasets', nargs='*', default=None)
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

    entries = scan_zip(zip_path)
    datasets = args.datasets or sorted(entries, key=dataset_sort_key)
    unknown = [dataset for dataset in datasets if dataset not in entries]
    if unknown:
        raise ValueError(f'datasets not found in zip: {unknown}')

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for dataset in datasets:
            print(f'=== {dataset} ===', flush=True)
            compact_path, samples = build_one_dataset(
                zf=zf,
                zip_entries=entries,
                dataset=dataset,
                output_root=output_root,
                work_root=work_root,
                class_dir=class_dir,
                lib_dir=lib_dir,
                java_src=java_src,
                shift=args.shift,
                max_per_class=int(args.max_per_class),
                keep_extracted=bool(args.keep_extracted),
                java_xmx=args.java_xmx,
            )
            with np.load(compact_path, allow_pickle=False) as cache:
                labels = cache['labels']
                class0 = int((labels == 0).sum())
                class1 = int((labels == 1).sum())
            manifest_rows.append({
                'dataset': dataset,
                'samples': samples,
                'class0': class0,
                'class1': class1,
                'cache_file': str(compact_path.relative_to(output_root.parent)),
            })
            print(f'wrote {compact_path} ({samples} samples)', flush=True)

    write_manifest(output_root, manifest_rows)
    print(output_root)


if __name__ == '__main__':
    main()
