"""Compare traffic-light IDs and count labeled boxes without running inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import socket
import sys

import yaml

from merge_yolo_ids import SPLITS, images_for, label_for, load_dataset


LIGHT_NAMES = (r'traffic[ _-]*(?:light|signal)|signal[ _-]*light|'
               r'(?:^|[^a-z])(?:red|green|yellow|amber)(?:$|[^a-z])|'
               r'信号灯|红灯|绿灯|黄灯|[红绿黄].*(?:直行|左转|右转|箭头)')


def checkpoint_names(path):
    path = path.expanduser().resolve()
    if path.suffix.lower() != '.pt' or not path.is_file():
        raise ValueError(f'Expected an existing local .pt checkpoint: {path}')
    try:
        import torch

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        raise ValueError(f'Could not read {path}; run in the environment used to train it: {exc}') from exc
    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get('ema')
        if model is None:
            model = checkpoint.get('model')
    names = getattr(model, 'names', None)
    if names is None and isinstance(checkpoint, dict):
        names = checkpoint.get('names')
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError(f'Checkpoint does not contain class names: {path}')
    parsed = {}
    for key, name in names.items():
        if not re.fullmatch(r'[0-9]+', str(key)) or not isinstance(name, str) or not name.strip():
            raise ValueError(f'Invalid checkpoint class entry: {key!r}: {name!r}')
        cid = int(key)
        if cid in parsed:
            raise ValueError(f'Duplicate checkpoint class ID: {cid}')
        parsed[cid] = name.strip()
    return parsed


def scan_dataset(dataset):
    stats = {}
    for split in SPLITS:
        if not dataset['config'].get(split):
            stats[split] = None
            continue
        print(f'扫描 {dataset["yaml"]} [{split}]', file=sys.stderr, flush=True)
        counts = Counter()
        issues = Counter()
        examples = []
        images = images_for(dataset, split)
        for image in images:
            label = label_for(image)
            try:
                rows = label.read_text(encoding='utf-8-sig').splitlines()
            except (OSError, UnicodeError) as exc:
                issue = 'missing_labels' if isinstance(exc, FileNotFoundError) else 'unreadable_labels'
                issues[issue] += 1
                if len(examples) < 10:
                    examples.append(f'{issue}: {label}')
                continue
            for number, row in enumerate(rows, 1):
                fields = row.split()
                if not fields:
                    continue
                if len(fields) != 5 or not fields[0].isdecimal():
                    issue = 'invalid_label_rows'
                else:
                    cid = int(fields[0])
                    if cid in dataset['names']:
                        counts[cid] += 1
                        continue
                    issue = 'unknown_id_rows'
                issues[issue] += 1
                if len(examples) < 10:
                    examples.append(f'{issue}: {label}:{number}: {fields[0]}')
        stats[split] = {'images': len(images), 'boxes_by_id': dict(sorted(counts.items())),
                        'issues': dict(issues), 'examples': examples}
    return {'yaml': str(dataset['yaml']), 'names': dataset['names'], 'splits': stats}


def read_merge_report(path, merged_names):
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    reported_names = {int(cid): name for cid, name in report['names'].items()}
    padding = set(report.get('output_padding_ids', []))
    if reported_names != {cid: name for cid, name in merged_names.items() if cid not in padding}:
        raise ValueError(f'Merge report names do not match the merged data.yaml: {path}')
    return ({Path(source).resolve() for source in report['sources']},
            {int(old): int(new) for old, new in report['id_mapping'].items()},
            {int(cid): name for cid, name in report['original_names'].items()})


def class_ids(names, name):
    return sorted(cid for cid, value in names.items() if value == name)


def compare_ids(left, right):
    if not left or not right:
        return 'missing'
    if len(left) != 1 or len(right) != 1:
        return 'ambiguous'
    return 'match' if left == right else 'mismatch'


def describe_class(name, sources, merged, weights, merge_report):
    merged_ids = class_ids(merged['names'], name)
    weight_ids = class_ids(weights, name)
    source_rows = []
    for source in sources:
        ids = class_ids(source['names'], name)
        status = compare_ids(ids, merged_ids)
        expected = None
        if merge_report:
            input_paths, mapping, original_names = merge_report
            if Path(source['yaml']) in input_paths and ids:
                expected = [mapping[cid] for cid in ids if cid in mapping and original_names.get(cid) == name]
                if not expected:
                    status = 'not_in_merge_mapping'
                elif len(expected) == len(ids) == 1 and expected == merged_ids:
                    status = 'mapped' if ids != expected else 'match'
                elif len(ids) > 1 or len(merged_ids) > 1:
                    status = 'ambiguous'
                else:
                    status = 'mapping_mismatch'
        source_rows.append({'ids': ids, 'expected_merged_ids': expected, 'status': status})
    return {'name': name, 'merged_ids': merged_ids, 'checkpoint_ids': weight_ids,
            'checkpoint_vs_merged': compare_ids(weight_ids, merged_ids), 'sources': source_rows}


def counts_text(dataset, ids):
    values = []
    for split in SPLITS:
        data = dataset['splits'][split]
        values.append('/'.join(str(data['boxes_by_id'].get(cid, 0)) for cid in ids)
                      if ids and data is not None else '—')
    return ' | '.join(values)


def print_results(report):
    sources = report['sources']
    merged = report['merged']
    for number, dataset in enumerate(sources, 1):
        print(f'源{number}  {dataset["yaml"]}')
    print(f'合并  {merged["yaml"]}')
    print(f'权重  {report["checkpoint"]["path"]}')
    print()
    print('类别 | 数据 | ID | train 框 | val 框 | test 框 | 对比合并数据')
    print('--- | --- | --- | ---: | ---: | ---: | ---')
    for entry in report['classes']:
        name = entry['name']
        for number, (source, row) in enumerate(zip(sources, entry['sources']), 1):
            ids = row['ids']
            if ids:
                print(f'{name} | 源{number} | {ids} | {counts_text(source, ids)} | {row["status"]}')
        ids = entry['merged_ids']
        print(f'{name} | 合并 | {ids} | {counts_text(merged, ids)} | —')
        print(f'{name} | best.pt | {entry["checkpoint_ids"]} | — | — | — | {entry["checkpoint_vs_merged"]}')
    for dataset in [*sources, merged]:
        for split, stats in dataset['splits'].items():
            if stats and stats['issues']:
                print(f'检查项 {dataset["yaml"]} [{split}] {stats["issues"]}，示例见 JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='extend', nargs='+', required=True,
                        help='Source dataset folders or data.yaml files')
    parser.add_argument('--merged', type=Path, required=True, help='Merged dataset folder or data.yaml')
    parser.add_argument('--weights', type=Path, required=True, help='Trusted local best.pt checkpoint')
    parser.add_argument('--merge-report', type=Path, help='Defaults to report.json beside merged data.yaml if present')
    parser.add_argument('--classes', action='extend', nargs='+',
                        help='Exact class names to check instead of automatic detection')
    parser.add_argument('--class-regex', default=LIGHT_NAMES, help='Case-insensitive class discovery pattern')
    parser.add_argument('--output', type=Path, default=Path('traffic_light_audit.json'))
    args = parser.parse_args()
    try:
        sources = [load_dataset(path) for path in args.source]
        merged = load_dataset(args.merged)
        weights = checkpoint_names(args.weights)
        report_path = args.merge_report or merged['yaml'].with_name('report.json')
        merge_report = None
        if args.merge_report or report_path.is_file():
            merge_report = read_merge_report(report_path, merged['names'])
        if args.classes:
            names = sorted(set(args.classes))
        else:
            pattern = re.compile(args.class_regex, re.IGNORECASE)
            names = sorted({name for table in [*(source['names'] for source in sources), merged['names'], weights]
                            for name in table.values() if pattern.search(name)})
        if not names:
            raise ValueError('No traffic-light classes matched; specify the names with --classes')
        sources = [scan_dataset(source) for source in sources]
        merged = scan_dataset(merged)
        report = {'machine': socket.gethostname(), 'sources': sources, 'merged': merged,
                  'checkpoint': {'path': str(args.weights.resolve()), 'names': weights},
                  'merge_report': str(report_path.resolve()) if merge_report else None,
                  'classes': [describe_class(name, sources, merged, weights, merge_report) for name in names]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print_results(report)
        print(f'JSON  {args.output.resolve()}')
    except (ValueError, OSError, TypeError, KeyError, yaml.YAMLError, re.error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
