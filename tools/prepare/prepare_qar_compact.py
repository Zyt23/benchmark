#!/usr/bin/env python
import argparse
import glob
import os

import numpy as np
import pandas as pd


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


def process_file(filepath, feature_cols, shift):
    frame = pd.read_csv(filepath)
    frame['N21'] = linear_fill_single_zero(frame['N21'])
    frame['N22'] = linear_fill_single_zero(frame['N22'])
    frame = frame.fillna(0.0)
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    phase = frame['FLIGHT_PHASE'].to_numpy(dtype=np.int64)

    transition = find_transition(phase, 2, 3)
    if transition is None:
        values = np.zeros((100, features.shape[1]), dtype=np.float32)
        mask = np.zeros(100, dtype=np.float32)
    else:
        transition += shift
        values, mask = extract_window(
            features, transition - 30, transition + 70, 100)
    return instance_norm(values, mask), mask


def main():
    parser = argparse.ArgumentParser(
        description='Build a compact, numerically equivalent QAR cache for one fixed phase-A shift.')
    parser.add_argument('--root_path', default='./dataset6/')
    parser.add_argument('--phase_a_shift', type=int, default=-80)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    output = args.output or os.path.join(
        args.root_path, 'qar_compact_shift{}.npz'.format(shift_tag(args.phase_a_shift)))
    if os.path.exists(output):
        raise FileExistsError('Refusing to overwrite existing cache: {}'.format(output))

    class_names = sorted(
        [name for name in os.listdir(args.root_path)
         if os.path.isdir(os.path.join(args.root_path, name))],
        key=lambda value: int(value))
    files_per_class = {
        class_name: sorted(glob.glob(os.path.join(args.root_path, class_name, '*.csv')))
        for class_name in class_names
    }
    first_file = files_per_class[class_names[0]][0]
    header = pd.read_csv(first_file, nrows=0).columns.tolist()
    feature_cols = [column for column in header if column not in ('Time', 'FLIGHT_PHASE')]

    records = []
    for class_name in class_names:
        for filepath in files_per_class[class_name]:
            x, mask = process_file(filepath, feature_cols, args.phase_a_shift)
            records.append((filepath, int(class_name), x, mask))
            if len(records) % 500 == 0:
                print('processed {}'.format(len(records)), flush=True)

    x = np.stack([item[2] for item in records]).astype(np.float32, copy=False)
    mask = np.stack([item[3] for item in records]).astype(np.float32, copy=False)
    labels = np.asarray([item[1] for item in records], dtype=np.int64)
    filenames = np.asarray([os.path.basename(item[0]) for item in records])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez(
        output,
        x=x,
        mask=mask,
        labels=labels,
        filenames=filenames,
        feature_cols=np.asarray(feature_cols),
        class_names=np.asarray(class_names),
        phase_a_shift=np.asarray([args.phase_a_shift], dtype=np.int64),
    )
    print('saved {} samples to {}'.format(len(records), output))
    print('x={} mask={} size={:.1f} MiB'.format(
        x.shape, mask.shape, os.path.getsize(output) / (1024 ** 2)))


if __name__ == '__main__':
    main()
