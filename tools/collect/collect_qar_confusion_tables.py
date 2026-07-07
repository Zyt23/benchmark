#!/usr/bin/env python
"""Collect QAR binary classification confusion tables.

Example:
    python collect_qar_confusion_tables.py \
      --run_tags tsfile_custom_20260702_120000 \
      --output_dir experiment_artifacts/QAR_tsfile_custom_20260702_120000
"""

import argparse
import ast
import csv
import re
from pathlib import Path


MODEL_ORDER = ['Transformer', 'TimesNet', 'PatchTST', 'DLinear', 'iTransformer']


def dataset_sort_key(name):
    match = re.match(r'^dataset(\d+)(?:-(\d+))?(?:_(.+))?$', name)
    if not match:
        return (10 ** 9, 10 ** 9, name)
    suffix = int(match.group(2)) if match.group(2) is not None else -1
    extra = match.group(3) or ''
    return (int(match.group(1)), suffix, extra)


def model_sort_key(name):
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER)


def read_summary_rows(root, run_tags):
    rows = []
    for run_tag in run_tags:
        summary_dir = root / 'logs' / 'datasetall' / run_tag
        summary_files = []
        plain = summary_dir / 'summary.tsv'
        if plain.exists():
            summary_files.append(plain)
        summary_files.extend(sorted(summary_dir.glob('summary_*.tsv')))
        summary_all = summary_dir / 'summary_all.tsv'
        if summary_all.exists():
            summary_files.append(summary_all)
        if not summary_files:
            raise FileNotFoundError(f'No summary TSV files found in {summary_dir}')
        seen = set()
        for summary_file in summary_files:
            with summary_file.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                for row in reader:
                    key = (row.get('dataset'), row.get('model'), row.get('result_dir'))
                    if key in seen:
                        continue
                    seen.add(key)
                    row['run_tag'] = run_tag
                    row['summary_file'] = str(summary_file)
                    rows.append(row)
    return rows


def grab(text, pattern, default=''):
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_int(text, name):
    value = grab(text, rf'^{name}:(-?\d+)')
    return int(value) if value != '' else ''


def parse_confusion(text):
    tn = parse_int(text, 'TN')
    fp = parse_int(text, 'FP')
    fn = parse_int(text, 'FN')
    tp = parse_int(text, 'TP')
    if all(v != '' for v in (tn, fp, fn, tp)):
        return tn, fp, fn, tp

    raw = grab(text, r'^confusion matrix:(.+)$')
    if raw:
        try:
            matrix = ast.literal_eval(raw)
            if len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2:
                return int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0]), int(matrix[1][1])
        except Exception:
            pass
    return '', '', '', ''


def collect(root, run_tags):
    rows = []
    for summary in read_summary_rows(root, run_tags):
        result_dir = summary.get('result_dir', '').lstrip('./')
        result_file = root / result_dir / 'result_classification.txt'
        text = result_file.read_text(errors='replace', encoding='utf-8') if result_file.exists() else ''
        tn, fp, fn, tp = parse_confusion(text)
        total = ''
        if all(v != '' for v in (tn, fp, fn, tp)):
            total = tn + fp + fn + tp
        accuracy = grab(text, r'^accuracy:([^\n]+)')
        if accuracy == '' and total:
            accuracy = (tn + tp) / total
        rows.append({
            'dataset': summary.get('dataset', ''),
            'model': summary.get('model', ''),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'total': total,
            'accuracy': format_accuracy(accuracy),
            'status': summary.get('status', ''),
            'run_tag': summary.get('run_tag', ''),
            'result_file': str(result_file.relative_to(root)) if result_file.exists() else str(result_file),
            'log': summary.get('log', ''),
        })
    rows.sort(key=lambda row: (dataset_sort_key(row['dataset']), model_sort_key(row['model'])))
    return rows


def format_accuracy(value):
    if value == '':
        return ''
    try:
        return f'{float(value):.6f}'
    except Exception:
        return str(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['dataset', 'model', 'TN', 'FP', 'FN', 'TP', 'total', 'accuracy',
              'status', 'run_tag', 'result_file', 'log']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    columns = ['model', 'TN', 'FP', 'FN', 'TP', 'total', 'accuracy']
    lines = ['| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |')
    return '\n'.join(lines)


def write_markdown(path, rows, run_tags):
    datasets = sorted({row['dataset'] for row in rows}, key=dataset_sort_key)
    lines = ['# QAR confusion tables', '']
    lines.append('- Run tags: `{}`'.format('`, `'.join(run_tags)))
    lines.append('- Columns use binary layout: class 0 is negative, class 1 is positive.')
    lines.append('')
    for dataset in datasets:
        lines.append(f'## {dataset}')
        lines.append('')
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        lines.append(markdown_table(dataset_rows))
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Collect binary confusion tables from QAR classification runs.')
    parser.add_argument('--root', default='.')
    parser.add_argument('--run_tags', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(root, args.run_tags)
    if not rows:
        raise SystemExit('No rows collected')
    write_csv(output_dir / 'confusion_tables.csv', rows)
    write_markdown(output_dir / 'confusion_tables.md', rows, args.run_tags)
    print(output_dir / 'confusion_tables.md')


if __name__ == '__main__':
    main()
