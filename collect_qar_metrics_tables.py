#!/usr/bin/env python
"""Collect QAR classification metrics into per-dataset tables.

Example:
    python collect_qar_metrics_tables.py \
      --run_tags datasetall_shiftN80_formal_20260624_120059 datasetall_extra_shiftN80_20260625_233801 \
      --output_dir experiment_artifacts/QAR_all_datasets_shiftN80_20260625_233801
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np


MODEL_ORDER = ['Transformer', 'TimesNet', 'PatchTST', 'DLinear', 'iTransformer']


def dataset_sort_key(name):
    match = re.match(r'^dataset(\d+)(?:-(\d+))?$', name)
    if not match:
        return (10 ** 9, 10 ** 9, name)
    suffix = int(match.group(2)) if match.group(2) is not None else -1
    return (int(match.group(1)), suffix, name)


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return float('nan')


def read_summary_rows(root, run_tags):
    rows = []
    for run_tag in run_tags:
        summary_dir = root / 'logs' / 'datasetall' / run_tag
        summary_files = sorted(summary_dir.glob('summary_*.tsv'))
        plain_summary = summary_dir / 'summary.tsv'
        if plain_summary.exists():
            summary_files.insert(0, plain_summary)
        if not summary_files:
            raise FileNotFoundError('No summary TSV files found in {}'.format(summary_dir))
        for summary_file in summary_files:
            with summary_file.open(newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                for row in reader:
                    row['run_tag'] = run_tag
                    row['summary_file'] = str(summary_file)
                    rows.append(row)
    return rows


def grab_metric(text, pattern, default=''):
    # result_classification.txt is append-only.  A resumed/final evaluation may
    # therefore follow an older TEST record in the same setting directory.
    # Always collect the most recent complete metric rather than the stale one.
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[-1].strip() if matches else default


def discover_result_dir(root, run_tag, dataset, model):
    """Recover result paths omitted when the server result root is a symlink."""
    results_root = root / 'results'
    if not results_root.exists():
        return None
    suffix = '_{}_{}_{}_0'.format(run_tag, dataset, model)
    matches = sorted(
        (path for path in results_root.iterdir()
         if path.is_dir() and path.name.endswith(suffix)),
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    return matches[0] if matches else None


def support_by_class(text):
    supports = {}
    for line in text.splitlines():
        match = re.match(r'\s*(\d+)\s+\S+\s+\S+\s+\S+\s+(\d+)\s*$', line)
        if match:
            supports[match.group(1)] = match.group(2)
    return supports


def parse_confusion_counts(text):
    values = {}
    for key in ('TN', 'FP', 'FN', 'TP'):
        values[key] = grab_metric(text, r'^{}:([^\n]+)'.format(key))
    return values


def epoch_info(log_text):
    best_epoch = ''
    best_val_acc = -1.0
    test_acc_at_best = ''
    epochs_run = 0
    for line in log_text.splitlines():
        # Current leakage-safe training logs contain TRAIN/VAL only. Keep the
        # optional TEST parse for backward compatibility with historical runs.
        match = re.search(
            r'Epoch:\s*(\d+),\s*Steps:.*?Vali Acc:\s*([0-9.]+)',
            line)
        if not match:
            continue
        epoch = int(match.group(1))
        val_acc = float(match.group(2))
        test_match = re.search(r'Test Acc:\s*([0-9.]+)', line)
        test_acc = float(test_match.group(1)) if test_match else ''
        epochs_run = max(epochs_run, epoch)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc_at_best = test_acc
    return epochs_run, best_epoch, '' if best_val_acc < 0 else best_val_acc, test_acc_at_best


def collect_metrics(root, run_tags):
    rows = read_summary_rows(root, run_tags)
    metrics = []
    for row in rows:
        dataset = row['dataset']
        model = row['model']
        status = int(row['status'])
        log_file = root / row['log'].lstrip('./')
        result_dir_text = row.get('result_dir', '').strip()
        result_dir = root / result_dir_text.lstrip('./') if result_dir_text else None
        if status == 0 and result_dir is None:
            result_dir = discover_result_dir(root, row['run_tag'], dataset, model)
        if result_dir is None:
            result_dir = root / '__missing_result_dir__'
        result_file = result_dir / 'result_classification.txt'

        result_text = result_file.read_text(errors='replace') if result_file.exists() else ''
        log_text = log_file.read_text(errors='replace') if log_file.exists() else ''
        supports = support_by_class(result_text)
        epochs_run, best_epoch, best_val_acc, test_acc_at_best = epoch_info(log_text)

        accuracy = grab_metric(result_text, r'^accuracy:([^\n]+)')
        macro_f1 = grab_metric(result_text, r'^macro F1:([^\n]+)')
        weighted_f1 = grab_metric(result_text, r'^weighted F1:([^\n]+)')
        true_counts = grab_metric(result_text, r'^true counts:([^\n]+)')
        pred_counts = grab_metric(result_text, r'^pred counts:([^\n]+)')
        confusion_counts = parse_confusion_counts(result_text)
        metrics.append({
            'dataset': dataset,
            'model': model,
            'status': status,
            'acc': accuracy,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'true_counts': true_counts,
            'pred_counts': pred_counts,
            'TN': confusion_counts['TN'],
            'FP': confusion_counts['FP'],
            'FN': confusion_counts['FN'],
            'TP': confusion_counts['TP'],
            'epochs_run': epochs_run,
            'best_epoch_by_val_acc': best_epoch,
            'best_val_acc': best_val_acc,
            'test_acc_at_best_val': test_acc_at_best,
            'test_support_class0': supports.get('0', ''),
            'test_support_class1': supports.get('1', ''),
            'run_tag': row['run_tag'],
            'log_file': str(log_file.relative_to(root)) if log_file.exists() else str(log_file),
            'result_file': str(result_file.relative_to(root)) if result_file.exists() else str(result_file),
        })
    metrics.sort(key=lambda item: (
        dataset_sort_key(item['dataset']),
        MODEL_ORDER.index(item['model']) if item['model'] in MODEL_ORDER else 999))
    return metrics


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cache_manifest(root, datasets, compact_root='datasetall_compact'):
    rows = []
    for dataset in sorted(datasets, key=dataset_sort_key):
        cache = root / compact_root / dataset / 'qar_compact_shiftN80.npz'
        if not cache.exists():
            rows.append({
                'dataset': dataset,
                'samples': '',
                'train_samples': '',
                'val_samples': '',
                'test_samples': '',
                'class0': '',
                'class1': '',
                'cache_file': str(cache.relative_to(root)),
                'sha256': '',
            })
            continue
        data = np.load(cache, allow_pickle=False)
        labels = data['labels']
        class0 = int((labels == 0).sum())
        class1 = int((labels == 1).sum())
        def split_bounds(n):
            if n <= 0:
                return 0, 0
            if n == 1:
                return 0, 0
            if n == 2:
                return 1, 1
            train_end = max(1, min(int(n * 0.7), n - 2))
            val_end = max(train_end + 1, min(int(n * 0.8), n - 1))
            return train_end, val_end

        train0, val_end0 = split_bounds(class0)
        train1, val_end1 = split_bounds(class1)
        train = train0 + train1
        val = (val_end0 - train0) + (val_end1 - train1)
        test = int(labels.shape[0]) - train - val
        rows.append({
            'dataset': dataset,
            'samples': int(labels.shape[0]),
            'train_samples': train,
            'val_samples': val,
            'test_samples': test,
            'class0': class0,
            'class1': class1,
            'cache_file': str(cache.relative_to(root)),
            'sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
        })
    return rows


def markdown_table(rows, columns):
    lines = []
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(column, '')) for column in columns) + ' |')
    return '\n'.join(lines)


def format_metric(value):
    number = parse_float(value)
    if number != number:
        return value
    return '{:.6f}'.format(number)


def copy_artifacts(root, output_dir, metrics, copy_code_snapshot=False):
    for row in metrics:
        log_file = root / row['log_file']
        if log_file.exists():
            dst = output_dir / 'logs' / row['run_tag'] / log_file.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_file, dst)

        result_file = root / row['result_file']
        if result_file.exists():
            dst = output_dir / 'results' / row['dataset'] / row['model'] / 'result_classification.txt'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_file, dst)

    if not copy_code_snapshot:
        return

    code_files = [
        '.gitattributes', '.gitignore', 'prepare_qar_compact.py',
        'prepare_qar_compact_from_zips.py', 'prepare_tsfile_compact_from_zip.py',
        'collect_qar_metrics_tables.py', 'run.py',
        'data_provider/data_loader.py', 'data_provider/m4.py', 'data_provider/data_factory.py',
        'layers/SelfAttention_Family.py', 'exp/exp_classification.py',
        'scripts/tsfile/TsFileWindowDumper.java',
        'scripts/classification/TimesNet_QAR_shiftN80.sh',
        'scripts/classification/run_QAR_datasetall_shiftN80.sh',
        'scripts/classification/orchestrate_QAR_datasetall_shiftN80.sh',
    ]
    for rel in code_files:
        src = root / rel
        if src.exists():
            dst = output_dir / 'code' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_readme(output_dir, metrics, cache_rows, run_tags, remote_project, compact_root='datasetall_compact'):
    datasets = sorted({row['dataset'] for row in metrics}, key=dataset_sort_key)
    lines = []
    lines.append('# QAR shiftN80 all-dataset classification tables')
    lines.append('')
    lines.append('- Run tags: `{}`'.format('`, `'.join(run_tags)))
    if remote_project:
        lines.append('- Remote project: `{}`'.format(remote_project))
    lines.append('- Models: {}'.format(', '.join(MODEL_ORDER)))
    lines.append('- Metrics columns: `acc` is kept as an alias of `accuracy`, followed by `macro_f1`, `weighted_f1`, and binary `TN/FP/FN/TP`.')
    lines.append('- Compact cache root: `{}`.'.format(compact_root))
    lines.append('- Training setup: `phase_a_shift=-80`, up to `train_epochs=50`, model-specific batch sizes; QAR classification uses deterministic per-class `70/10/20` TRAIN/VAL/TEST split, class weighting, and early stopping on validation `macro_f1`.')
    lines.append('- TEST is evaluated once only after loading the best validation checkpoint; it is not read or logged during training.')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    best_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in metrics if row['dataset'] == dataset]
        best = sorted(
            dataset_rows,
            key=lambda row: (parse_float(row['macro_f1']), parse_float(row['accuracy'])),
            reverse=True)[0]
        best_rows.append({
            'dataset': dataset,
            'best_model': best['model'],
            'accuracy': format_metric(best['accuracy']),
            'macro_f1': format_metric(best['macro_f1']),
            'weighted_f1': format_metric(best['weighted_f1']),
        })
    lines.append(markdown_table(best_rows, ['dataset', 'best_model', 'accuracy', 'macro_f1', 'weighted_f1']))
    lines.append('')
    lines.append('## Per-dataset metric tables')
    lines.append('')
    table_columns = ['model', 'acc', 'accuracy', 'macro_f1', 'weighted_f1', 'true_counts', 'pred_counts', 'TN', 'FP', 'FN', 'TP']
    for dataset in datasets:
        lines.append('### {}'.format(dataset))
        lines.append('')
        dataset_rows = []
        for row in [item for item in metrics if item['dataset'] == dataset]:
            dataset_rows.append({
                'model': row['model'],
                'acc': format_metric(row['acc']),
                'accuracy': format_metric(row['accuracy']),
                'macro_f1': format_metric(row['macro_f1']),
                'weighted_f1': format_metric(row['weighted_f1']),
                'true_counts': row['true_counts'],
                'pred_counts': row['pred_counts'],
                'TN': row['TN'],
                'FP': row['FP'],
                'FN': row['FN'],
                'TP': row['TP'],
            })
        lines.append(markdown_table(dataset_rows, table_columns))
        lines.append('')
    lines.append('## Cache manifest')
    lines.append('')
    lines.append(markdown_table(cache_rows, [
        'dataset', 'samples', 'train_samples', 'test_samples', 'class0', 'class1']))
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- Full CSV: `all_metrics.csv`; per-dataset CSV tables: `tables/<dataset>_metrics.csv`.')
    lines.append('- `logs/` and `results/` contain the copied raw training logs and classification reports; `all_metrics.csv` also records `true_counts`, `pred_counts`, and `TN/FP/FN/TP` to spot majority-class collapse.')
    lines.append('- Some datasets (notably dataset9 and dataset12) may still reach very high scores; before using those as scientific conclusions, inspect possible source/metadata distribution leakage and consider group split.')
    (output_dir / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Collect QAR metrics into per-dataset tables.')
    parser.add_argument('--root', default='.')
    parser.add_argument('--run_tags', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--remote_project', default='')
    parser.add_argument('--compact_root', default='datasetall_compact')
    parser.add_argument('--copy_code_snapshot', action='store_true',
                        help='Opt in to copying a code/ snapshot into the result artifact.')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError('Refusing to overwrite {}'.format(output_dir))
        artifact_root = (root / 'experiment_artifacts').resolve()
        if artifact_root not in output_dir.parents:
            raise ValueError('Refusing to remove non-artifact output {}'.format(output_dir))
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    metrics = collect_metrics(root, args.run_tags)
    if not metrics:
        raise ValueError('No metrics collected')

    # A sharded rerun can leave a non-zero summary row when an obsolete worker
    # is stopped after a successful replacement shard has already completed.
    # Keep one row per dataset/model, preferring a successful row and otherwise
    # the later run tag supplied on the command line.
    deduplicated = {}
    for row in metrics:
        key = (row.get('dataset', ''), row.get('model', ''))
        previous = deduplicated.get(key)
        current_ok = str(row.get('status', '')) in ('0', '0.0')
        previous_ok = previous is not None and str(previous.get('status', '')) in ('0', '0.0')
        if previous is None or current_ok or not previous_ok:
            deduplicated[key] = row
    metrics = list(deduplicated.values())

    all_fields = [
        'dataset', 'model', 'status', 'acc', 'accuracy', 'macro_f1', 'weighted_f1',
        'true_counts', 'pred_counts', 'TN', 'FP', 'FN', 'TP',
        'epochs_run', 'best_epoch_by_val_acc', 'best_val_acc', 'test_acc_at_best_val',
        'test_support_class0', 'test_support_class1', 'run_tag', 'log_file', 'result_file',
    ]
    write_csv(output_dir / 'all_metrics.csv', metrics, all_fields)

    table_fields = ['model', 'acc', 'accuracy', 'macro_f1', 'weighted_f1',
                    'true_counts', 'pred_counts', 'TN', 'FP', 'FN', 'TP']
    for dataset in sorted({row['dataset'] for row in metrics}, key=dataset_sort_key):
        rows = [
            {field: row[field] for field in table_fields}
            for row in metrics
            if row['dataset'] == dataset
        ]
        write_csv(output_dir / 'tables' / '{}_metrics.csv'.format(dataset), rows, table_fields)

    cache_rows = build_cache_manifest(root, {row['dataset'] for row in metrics}, compact_root=args.compact_root)
    write_csv(output_dir / 'cache_manifest.csv', cache_rows, list(cache_rows[0].keys()))
    copy_artifacts(root, output_dir, metrics, copy_code_snapshot=args.copy_code_snapshot)
    write_readme(output_dir, metrics, cache_rows, args.run_tags, args.remote_project, compact_root=args.compact_root)

    print(output_dir)
    print('metrics:', len(metrics))


if __name__ == '__main__':
    main()
