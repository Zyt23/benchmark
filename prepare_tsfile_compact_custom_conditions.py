#!/usr/bin/env python
"""Build chronological compact caches for QAR custom-condition experiments.

Outputs keep the compact-cache contract used by the benchmark:

    <output_root>/<dataset>/qar_compact_shiftN80.npz

The cache contains one source flight/window per sample and now stores
``sources`` plus numeric ``time_keys`` so train/val/test can be split strictly
by time in the dataloaders.

Tasks:
  classification
      One multi-anchor window per flight.  dataset5~12 use standard 320/321
      features and 320321gongkuang anchors without 6->8.  dataset13 uses the
      dataset13 anchors.  dataset14 uses dataset14 features with the same
      standard anchors, also without 6->8.
  forecast
      Four independent caches, one for each phase transition:
      2->3, 4->5, 5->6, 8->9.  Each cache saves 80 rows:
      input rows = transition-30 ... transition+29, target rows =
      transition+30 ... transition+49 for seq_len=60/pred_len=20.
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
DEFAULT_DATASETS = [
    'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset8-1',
    'dataset9', 'dataset10', 'dataset11', 'dataset12',
    'dataset13', 'dataset14',
]
FORECAST_ANCHORS = {
    'predict_2_3': (2, 3),
    'predict_4_5': (4, 5),
    'predict_5_6': (5, 6),
    'predict_8_9': (8, 9),
}


def dataset_family(dataset):
    if dataset == 'dataset13':
        return 'dataset13'
    if dataset == 'dataset14':
        return 'dataset14'
    return 'standard'


def classification_mode(dataset):
    family = dataset_family(dataset)
    if family == 'dataset13':
        return 'dataset13_anchors'
    if family == 'dataset14':
        return 'dataset14_anchors'
    return 'standard_anchors'


def forecast_mode(dataset, anchor_name):
    family = dataset_family(dataset)
    return f'{family}_{anchor_name}'


def time_key_from_source(source):
    base = os.path.basename(str(source).replace('\\', '/'))
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

    for m in re.finditer(r'(20\d{2})[-_](\d{2})[-_](\d{2})[ T_](\d{2})[-_:](\d{2})[-_:](\d{2})', base):
        add(*m.groups())
    for m in re.finditer(r'(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})', base):
        add(*m.groups())
    for m in re.finditer(r'(20\d{2})[-_](\d{2})[-_](\d{2})', base):
        add(*m.groups())
    for m in re.finditer(r'(20\d{2})(\d{2})(\d{2})', base):
        add(*m.groups())

    if not candidates:
        return 99999999999999
    y, mo, d, h, mi, s = min(candidates)
    return int(f'{y:04d}{mo:02d}{d:02d}{h:02d}{mi:02d}{s:02d}')


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
            entries[dataset][label].sort(key=lambda src: (time_key_from_source(src), os.path.basename(src)))
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


def load_written_sources(stats_path):
    rows = []
    with stats_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            try:
                written_idx = int(row['written_idx'])
            except Exception:
                continue
            if written_idx < 0:
                continue
            rows.append((written_idx, row.get('source', '')))
    rows.sort(key=lambda x: x[0])
    return [source for _, source in rows]


def write_npz(raw_out_dir, compact_path, shift):
    meta = json.loads((raw_out_dir / 'meta.json').read_text(encoding='utf-8'))
    n = int(meta['samples'])
    seq_len = int(meta['seq_len'])
    feature_count = int(meta['feature_count'])
    feature_cols = np.array(meta['feature_cols'])

    x = np.fromfile(raw_out_dir / 'x.bin', dtype='>f4').astype(np.float32)
    mask = np.fromfile(raw_out_dir / 'mask.bin', dtype='>f4').astype(np.float32)
    labels = np.fromfile(raw_out_dir / 'labels.bin', dtype='>i4').astype(np.int64)
    sources = np.array(load_written_sources(raw_out_dir / 'stats.tsv'))
    time_keys = np.array([time_key_from_source(src) for src in sources], dtype=np.int64)

    expected_x = n * seq_len * feature_count
    expected_mask = n * seq_len
    if x.size != expected_x:
        raise ValueError(f'x size mismatch, expected {expected_x}, got {x.size}')
    if mask.size != expected_mask:
        raise ValueError(f'mask size mismatch, expected {expected_mask}, got {mask.size}')
    if labels.size != n:
        raise ValueError(f'labels size mismatch, expected {n}, got {labels.size}')
    if sources.size != n:
        raise ValueError(f'sources size mismatch, expected {n}, got {sources.size}')

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
        sources=sources,
        time_keys=time_keys,
    )
    return meta, sources, time_keys


def build_one_dataset(zf, entries, dataset, mode, output_root, work_root, class_dir, lib_dir,
                      java_src, java_class, shift, max_per_class, keep_extracted, java_xmx):
    labels = entries.get(dataset)
    if labels is None:
        raise ValueError(f'{dataset} not found in zip')
    missing = [label for label in ('0', '1') if label not in labels]
    if missing:
        raise ValueError(f'{dataset}: missing label dirs {missing}')

    dataset_work = work_root / dataset / mode
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

    rows.sort(key=lambda row: (time_key_from_source(row[2]), row[2]))
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        for label, local_path, source in rows:
            handle.write(f'{label}\t{local_path}\t{source}\n')

    compile_java(java_src, java_class, class_dir, lib_dir)
    run_java(manifest, raw_out_dir, class_dir, lib_dir, java_class, mode, shift, java_xmx)

    compact_path = output_root / dataset / f'qar_compact_shiftN{abs(shift)}.npz'
    meta, sources, time_keys = write_npz(raw_out_dir, compact_path, shift)
    shutil.copy2(raw_out_dir / 'stats.tsv', output_root / dataset / 'tsfile_conversion_stats.tsv')
    shutil.copy2(raw_out_dir / 'meta.json', output_root / dataset / 'tsfile_conversion_meta.json')

    if not keep_extracted:
        shutil.rmtree(dataset_work)
    return compact_path, meta, sources, time_keys


def write_manifest(output_root, rows):
    path = output_root / 'tsfile_custom_condition_manifest.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = [
            'dataset', 'mode', 'samples', 'class0', 'class1',
            'seq_len', 'feature_count', 'zero_window_count',
            'zero_window_rate', 'first_time_key', 'last_time_key',
            'cache_file',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_many(args, task_name, output_root, mode_for_dataset):
    zip_path = Path(args.zip_path).resolve()
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
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for dataset in args.datasets:
            mode = mode_for_dataset(dataset)
            print(f'=== {task_name}: {dataset} ({mode}) ===', flush=True)
            compact_path, meta, sources, time_keys = build_one_dataset(
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
                'first_time_key': int(time_keys.min()) if time_keys.size else '',
                'last_time_key': int(time_keys.max()) if time_keys.size else '',
                'cache_file': str(compact_path.relative_to(output_root.parent)),
            }
            manifest_rows.append(row)
            print(f'wrote {compact_path} ({row["samples"]} samples)', flush=True)

    write_manifest(output_root, manifest_rows)
    print(output_root)


def main():
    parser = argparse.ArgumentParser(description='Prepare chronological QAR compact caches with custom conditions.')
    parser.add_argument('--zip_path', default='datasetall_tsfile/tsfile_datasets.zip')
    parser.add_argument('--classification_output_root', default='datasetall_tsfile_compact_custom_cls_chrono')
    parser.add_argument('--forecast_output_root', default='datasetall_tsfile_compact_custom_forecast_chrono')
    parser.add_argument('--output_root', default='',
                        help='Optional single output root, mainly for one task.')
    parser.add_argument('--work_root', default='datasetall_tsfile_work_custom_conditions_chrono')
    parser.add_argument('--iotdb_lib', default='.cache/iotdb-2.0.2-lib')
    parser.add_argument('--java_src', default='scripts/tsfile/TsFileWindowDumperAnchors.java')
    parser.add_argument('--java_class', default='TsFileWindowDumperAnchors')
    parser.add_argument('--java_class_dir', default='.cache/tsfile_java_classes')
    parser.add_argument('--datasets', nargs='*', default=DEFAULT_DATASETS)
    parser.add_argument('--task', choices=['classification', 'forecast', 'all'], default='all')
    parser.add_argument('--forecast_anchors', nargs='*', default=list(FORECAST_ANCHORS.keys()),
                        choices=list(FORECAST_ANCHORS.keys()))
    parser.add_argument('--shift', type=int, default=-80)
    parser.add_argument('--max_per_class', type=int, default=0,
                        help='debug: limit files per class; 0 means all')
    parser.add_argument('--keep_extracted', action='store_true')
    parser.add_argument('--java_xmx', default='4g')
    args = parser.parse_args()

    if args.task in ('classification', 'all'):
        out = Path(args.output_root or args.classification_output_root).resolve()
        build_many(args, 'classification', out, classification_mode)

    if args.task in ('forecast', 'all'):
        base_out = Path(args.output_root or args.forecast_output_root).resolve()
        for anchor_name in args.forecast_anchors:
            out = base_out / anchor_name
            build_many(args, f'forecast/{anchor_name}', out,
                       lambda dataset, anchor_name=anchor_name: forecast_mode(dataset, anchor_name))


if __name__ == '__main__':
    main()
