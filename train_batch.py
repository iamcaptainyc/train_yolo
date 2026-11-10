"""Train the same YOLO checkpoint independently on multiple datasets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from search_worker import write_json


TRAIN_SCRIPT = Path(__file__).resolve().with_name('train.py')
DATASETS_DIR = TRAIN_SCRIPT.parent/'yolo_datasets'
PARAMETERS = {
    'epochs': int, 'imgsz': int, 'batch': int, 'eval_batch': int,
    'workers': int, 'seed': int, 'patience': int, 'device': (str, int),
    'cache': (str, bool), 'amp': bool, 'fliplr': (int, float),
}
SPLITS = ('val', 'test')
SCORES = ('precision', 'recall', 'map50', 'map50_95')


def training_flags(params, context):
    if not isinstance(params, dict):
        raise ValueError(f'{context} must be a mapping')
    unknown = set(params) - PARAMETERS.keys()
    if unknown:
        raise ValueError(f'{context} has unsupported parameters: {", ".join(sorted(map(str, unknown)))}')
    flags = []
    for name, value in params.items():
        types = PARAMETERS[name]
        if type(value) not in (types if isinstance(types, tuple) else (types,)):
            raise ValueError(f'Invalid {context}.{name}: {value!r}')
        if (name in {'epochs', 'imgsz', 'eval_batch'} and value <= 0
                or name in {'workers', 'seed', 'patience'} and value < 0
                or name == 'batch' and value != -1 and value < 1
                or name == 'fliplr' and not 0 <= value <= 1
                or name == 'device' and not str(value).strip()
                or name == 'cache' and value not in ('false', 'ram', 'disk', False)):
            raise ValueError(f'Invalid {context}.{name}: {value!r}')
        flag = '--' + name.replace('_', '-')
        if name == 'amp':
            flags.append(flag if value else '--no-amp')
        else:
            flags.extend((flag, 'false' if name == 'cache' and value is False else str(value)))
    return flags


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.keys()-{'model', 'output', 'defaults', 'datasets'}:
        raise ValueError('Expected YAML keys model, datasets, and optional output and defaults')

    def resolve(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} must be a path')
        return (path.parent/Path(value).expanduser()).resolve()

    model = resolve(config.get('model'), 'model')
    if model.suffix.lower() != '.pt' or not model.is_file():
        raise ValueError(f'Expected an existing local .pt checkpoint: {model}')
    output = resolve(config.get('output', '../runs/batch_train'), 'output')
    defaults = config.get('defaults', {})
    training_flags(defaults, 'defaults')
    entries = config.get('datasets')
    if not isinstance(entries, list) or not entries:
        raise ValueError('datasets must be a non-empty list')
    experiments, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.keys()-{'name', 'data', 'params'}:
            raise ValueError('Each dataset entry supports name, data and optional params')
        value = entry.get('data')
        data = resolve(value, 'datasets.data')
        if (value not in ('.', '..') and not any(char in value for char in '/\\:')
                and Path(value).suffix.lower() not in ('.yaml', '.yml')):
            candidate = DATASETS_DIR/value
            if candidate.is_dir() or not data.exists():
                data = candidate/'data.yaml'
        if data.is_dir():
            data = data/'data.yaml'
        data = data.resolve()
        if data.suffix.lower() not in ('.yaml', '.yml') or not data.is_file():
            raise ValueError(f'Dataset YAML does not exist: {data}')
        name = entry.get('name', data.parent.name if data.stem == 'data' else data.stem)
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name)
                or name.casefold() in names):
            raise ValueError(f'Experiment names must be unique and use letters, digits, underscores or hyphens: {name!r}')
        names.add(name.casefold())
        overrides = entry.get('params', {})
        training_flags(overrides, f'{name}.params')
        experiments.append({'name':name, 'data':str(data), 'params':{**defaults, **overrides}})
    return model, output, experiments


def read_evaluation(directory, split):
    path = directory/f'{split}_result.json'
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'status':'missing'}
    except (OSError, ValueError) as exc:
        return {'status':'invalid', 'error':str(exc)}
    if (not isinstance(result, dict) or result.get('split') != split
            or result.get('status') not in ('ok', 'skipped', 'failed')):
        return {'status':'invalid', 'error':f'Unexpected evaluation result in {path}'}
    metrics = result.get('metrics')
    if result.get('status') == 'ok' and (not isinstance(metrics, dict) or any(
            type(metrics.get(key)) not in (float, int) or not 0 <= metrics[key] <= 1 for key in SCORES)):
        return {'status':'invalid', 'error':f'Invalid evaluation metrics in {path}'}
    return result


def write_summary(directory, model, config, records):
    write_json(directory/'summary.json', {'model':str(model), 'config':str(config), 'experiments':records})
    headers = ['experiment', 'status', 'data', 'best_checkpoint', 'elapsed_seconds']
    headers += [f'{split}_{key}' for split in SPLITS for key in ('status', *SCORES)]
    rows = []
    for record in records:
        row = {'experiment':record['name'], **{key:record.get(key) for key in headers[1:5]}}
        for split, result in record.get('evaluation', {}).items():
            row[f'{split}_status'] = result.get('status')
            for key in SCORES:
                row[f'{split}_{key}'] = result['metrics'][key] if result.get('status') == 'ok' else None
        rows.append(row)
    with (directory/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    def cell(value):
        return '—' if value is None else f'{value:.2%}'

    lines = ['# 批量训练实验对比', '', f'初始权重 `{model}`', '',
             '每行指标来自该实验自己的验证集和测试集。', '',
             '| 实验 | 状态 | Val P | Val R | Val mAP50 | Val mAP50–95 | Test P | Test R | Test mAP50 | Test mAP50–95 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    states = {'pending':'待运行', 'running':'训练中', 'complete':'完成', 'failed':'失败', 'interrupted':'中断'}
    for row in rows:
        values = [row['experiment'], states[row['status']]]
        values.extend(cell(row.get(f'{split}_{key}')) for split in SPLITS for key in SCORES)
        lines.append('| '+' | '.join(values)+' |')
    lines += ['', '完整参数、评估状态、失败原因和权重路径见 `summary.json`。']
    (directory/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_batch(config_path):
    config_path = Path(config_path).resolve()
    model, output, experiments = load_config(config_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = output/f'{config_path.stem}_{stamp}'
    directory.mkdir(parents=True)
    records = [{**exp, 'status':'pending'} for exp in experiments]
    write_summary(directory, model, config_path, records)
    print(f'Batch output: {directory}', flush=True)
    for record in records:
        project = directory/record['name']
        train_dir = project/'training'
        command = [sys.executable, str(TRAIN_SCRIPT), '--mode', 'train', '--model', str(model),
                   '--data', record['data'], '--project', str(project), '--name', 'training',
                   *training_flags(record['params'], record['name'])]
        record.update(status='running', train_dir=str(train_dir), command=command)
        write_summary(directory, model, config_path, records)
        print(f'\n[{record["name"]}] {record["data"]}', flush=True)
        started = time.monotonic()
        child = None
        try:
            child = subprocess.Popen(command)
            code = child.wait()
            if code in (-2, 130, 0xC000013A, -1073741510):
                raise KeyboardInterrupt
            record['exit_code'] = code
            record['status'] = 'complete' if code == 0 else 'failed'
            if code:
                record['error'] = f'train.py exited with code {code}'
        except OSError as exc:
            record.update(status='failed', error=str(exc))
        except KeyboardInterrupt:
            record['status'] = 'interrupted'
            if child is not None and child.poll() is None:
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait()
        finally:
            record['elapsed_seconds'] = round(time.monotonic()-started, 2)
            checkpoint = train_dir/'weights/best.pt'
            record['best_checkpoint'] = str(checkpoint) if checkpoint.is_file() else None
            record['evaluation'] = {split:read_evaluation(train_dir, split) for split in SPLITS}
            if record['status'] == 'complete' and (not checkpoint.is_file()
                    or record['evaluation']['val']['status'] != 'ok'
                    or record['evaluation']['test']['status'] not in ('ok', 'skipped')):
                record.update(status='failed', error='Training outputs or evaluation results are incomplete')
            write_summary(directory, model, config_path, records)
        if record['status'] == 'interrupted':
            print(f'Interrupted. Partial comparison: {directory/"SUMMARY.md"}')
            return 130
    print(f'Comparison: {directory/"SUMMARY.md"}')
    return int(any(record['status'] == 'failed' for record in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Batch training YAML')
    args = parser.parse_args()
    try:
        return run_batch(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
