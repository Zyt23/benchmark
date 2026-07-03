#!/usr/bin/env python
"""Build compact QAR classification caches directly from dataset*.zip files.

The generated cache has the same layout expected by QARFlightDatasetShift:

    <output_root>/<dataset_name>/qar_compact_shiftN80.npz

It avoids extracting tens of gigabytes of raw CSVs locally or uploading them to
the training server. By default the script refuses to overwrite an existing
cache; pass --force when you intentionally want to rebuild one.
"""

import argparse
import os
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd


DROP_COLS = ('Time', 'FLIGHT_PHASE')
PHASE_COL = 'FLIGHT_PHASE'


def shift_tag(shift):
    return 'N{}'.format(abs(shift)) if shift < 0 else 'P{}'.format(shift)


def find_transition(phase, source, target):
    if phase.shape[0] < 2:
        return None
    hits = np.flatnonzero((phase[:-1] == source) & (phase[1:] == target))
    return None if hits.size == 0 else int(hits[0] + 1)


def extract_window(features, start, end, length):
    output = np.zeros((length, features.shape[1]), dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    source_start = max(start, 0)
    source_end = min(end, features.shape[0])
    if source_end > source_start:
        dest_start = source_start - start
        dest_end = dest_start + (source_end - source_start)
        output[dest_start:dest_end] = features[source_start:source_end]
        mask[dest_start:dest_end] = 1.0
    return output, mask


def instance_norm(values, mask):
    valid = mask > 0
    output = np.zeros_like(values)
    if valid.sum() > 1:
        mean = values[valid].mean(axis=0)
        std = values[valid].std(axis=0) + 1e-5
        output[valid] = (values[valid] - mean) / std
    return output


def linear_fill_single_zero(series):
    values = series.to_numpy(copy=True)
    for idx in range(1, len(values) - 1):
        if values[idx] == 0 and values[idx - 1] != 0 and values[idx + 1] != 0:
            values[idx] = (values[idx - 1] + values[idx + 1]) / 2
    return pd.Series(values, index=series.index, name=series.name)


def process_csv_stream(stream, feature_cols, shift):
    frame = pd.read_csv(stream)
    frame['N21'] = linear_fill_single_zero(frame['N21'])
    frame['N22'] = linear_fill_single_zero(frame['N22'])
    frame = frame.fillna(0.0)
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    phase = frame[PHASE_COL].to_numpy(dtype=np.int64)

    transition = find_transition(phase, 2, 3)
    if transition is None:
        values = np.zeros((100, features.shape[1]), dtype=np.float32)
        mask = np.zeros(100, dtype=np.float32)
    else:
        transition += shift
        values, mask = extract_window(
            features, transition - 30, transition + 70, 100)
    return instance_norm(values, mask), mask


def csv_entries_by_class(zip_file):
    grouped = {}
    for info in zip_file.infolist():
        if info.is_dir() or not info.filename.lower().endswith('.csv'):
            continue
        parts = pathlib.PurePosixPath(info.filename).parts
        if len(parts) < 2:
            continue
        class_name = parts[-2]
        if not class_name.isdigit():
            continue
        grouped.setdefault(class_name, []).append(info)
    return {
        class_name: sorted(infos, key=lambda item: item.filename)
        for class_name, infos in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def dataset_name_for_zip(zip_path):
    """Return the logical dataset name for one zip.

    Most archives are named datasetN.zip and contain both classes. A few are
    split as datasetN-0.zip + datasetN-1.zip, where each zip contains only its
    matching class directory. Those are treated as one logical datasetN.
    Archives like dataset8-1.zip that contain both 0 and 1 remain independent.
    """
    zip_path = pathlib.Path(zip_path)
    stem = zip_path.stem
    match = re.match(r'^(dataset\d+)-([01])$', stem)
    if not match:
        return stem

    with zipfile.ZipFile(zip_path) as zip_file:
        class_names = set(csv_entries_by_class(zip_file).keys())
    if class_names == {match.group(2)}:
        return match.group(1)
    return stem


def read_feature_columns(zip_file, first_info):
    with zip_file.open(first_info) as stream:
        header = pd.read_csv(stream, nrows=0).columns.tolist()
    return [column for column in header if column not in DROP_COLS]


def build_cache(zip_paths, output_root, shift, force=False, dataset_name=None):
    if isinstance(zip_paths, (str, pathlib.Path)):
        zip_paths = [zip_paths]
    zip_paths = [pathlib.Path(path) for path in zip_paths]
    dataset_name = dataset_name or dataset_name_for_zip(zip_paths[0])
    output_dir = pathlib.Path(output_root) / dataset_name
    output_path = output_dir / 'qar_compact_shift{}.npz'.format(shift_tag(shift))
    if output_path.exists() and not force:
        print('skip existing {}'.format(output_path), flush=True)
        return output_path

    all_files_per_class = {}
    feature_cols = None
    records = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zip_file:
            files_per_class = csv_entries_by_class(zip_file)
            if not files_per_class:
                raise ValueError('No class CSV files found in {}'.format(zip_path))

            class_names = list(files_per_class.keys())
            first_info = files_per_class[class_names[0]][0]
            current_feature_cols = read_feature_columns(zip_file, first_info)
            if feature_cols is None:
                feature_cols = current_feature_cols
            elif feature_cols != current_feature_cols:
                raise ValueError(
                    'Feature columns mismatch in {}: expected {}, got {}'.format(
                        zip_path, feature_cols, current_feature_cols))

            for class_name, infos in files_per_class.items():
                all_files_per_class.setdefault(class_name, 0)
                all_files_per_class[class_name] += len(infos)
                for info in infos:
                    with zip_file.open(info) as stream:
                        x, mask = process_csv_stream(stream, feature_cols, shift)
                    records.append((info.filename, int(class_name), x, mask))
                    if len(records) % 500 == 0:
                        print('{} processed {}'.format(dataset_name, len(records)), flush=True)

    class_names = sorted(all_files_per_class.keys(), key=lambda value: int(value))
    if not records:
        raise ValueError('No records produced for {}'.format(dataset_name))

    # Keep class-wise deterministic order so QARFlightDatasetShift's per-class
    # sorted 80/20 split remains stable even when a logical dataset spans zips.
    records.sort(key=lambda item: (item[1], item[0]))

    x = np.stack([item[2] for item in records]).astype(np.float32, copy=False)
    mask = np.stack([item[3] for item in records]).astype(np.float32, copy=False)
    labels = np.asarray([item[1] for item in records], dtype=np.int64)
    filenames = np.asarray([pathlib.PurePosixPath(item[0]).name for item in records])

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + '.tmp')
    with open(tmp_path, 'wb') as output_stream:
        np.savez(
            output_stream,
            x=x,
            mask=mask,
            labels=labels,
            filenames=filenames,
            feature_cols=np.asarray(feature_cols),
            class_names=np.asarray(class_names),
            phase_a_shift=np.asarray([shift], dtype=np.int64),
            source_zip=np.asarray([str(path) for path in zip_paths]),
        )
    if output_path.exists():
        output_path.unlink()
    os.replace(tmp_path, output_path)
    print('saved {} samples to {}'.format(len(records), output_path), flush=True)
    print('x={} mask={} size={:.1f} MiB'.format(
        x.shape, mask.shape, output_path.stat().st_size / (1024 ** 2)), flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Build compact QAR caches directly from dataset*.zip files.')
    parser.add_argument('--zip_dir', default='./datasetall',
                        help='Directory containing dataset*.zip files')
    parser.add_argument('--output_root', default='./datasetall_compact',
                        help='Output directory for compact cache folders')
    parser.add_argument('--phase_a_shift', type=int, default=-80)
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Optional dataset zip stems to include, e.g. dataset5 dataset7')
    parser.add_argument('--no_auto_merge_parts', action='store_true',
                        help='Do not auto-merge datasetN-0.zip + datasetN-1.zip style archives')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing compact caches')
    args = parser.parse_args()

    zip_dir = pathlib.Path(args.zip_dir)
    if args.datasets:
        zip_paths = []
        for name in args.datasets:
            stem = name[:-4] if name.endswith('.zip') else name
            zip_paths.append(zip_dir / '{}.zip'.format(stem))
    else:
        zip_paths = sorted(zip_dir.glob('dataset*.zip'))

    if not zip_paths:
        raise FileNotFoundError('No dataset*.zip files found under {}'.format(zip_dir))

    for zip_path in zip_paths:
        if not zip_path.is_file():
            raise FileNotFoundError(zip_path)

    groups = {}
    for zip_path in zip_paths:
        dataset_name = zip_path.stem if args.no_auto_merge_parts else dataset_name_for_zip(zip_path)
        groups.setdefault(dataset_name, []).append(zip_path)

    for dataset_name in sorted(groups):
        build_cache(
            sorted(groups[dataset_name]),
            args.output_root,
            args.phase_a_shift,
            force=args.force,
            dataset_name=dataset_name)


if __name__ == '__main__':
    main()
