"""Merge YOLO detection datasets preserving IDs; reserved IDs require explicit declaration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import yaml
from tqdm import tqdm

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLITS = ('train', 'val', 'test')


def load_dataset(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path/'data.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError(f'Invalid YAML: {path}')
    names = config.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError(f'Missing names: {path}')
    parsed = {}
    for key, name in names.items():
        if isinstance(key, bool) or not re.fullmatch(r'\d+', str(key)) or not isinstance(name, str) or not name.strip():
            raise ValueError(f'Invalid class entry {key!r}: {name!r} in {path}')
        cid = int(key)
        if cid in parsed:
            raise ValueError(f'Duplicate class ID {cid}: {path}')
        parsed[cid] = name.strip()
    root = Path(config.get('path') or path.parent).expanduser()
    if not root.is_absolute():
        root = path.parent/root
    return {'yaml': path, 'config': config, 'root': root.resolve(),
            'names': parsed, 'reserved': set()}


def parse_reserved_ids(specs, source_count):
    result = defaultdict(set)
    for spec in specs or []:
        if not re.fullmatch(r'[1-9]\d*:\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', spec):
            raise ValueError(f'Invalid --reserved-ids {spec!r}; use source:IDs, e.g. 2:0-26 or 3:0,2-5')
        source, values = spec.split(':')
        source = int(source)
        if source > source_count:
            raise ValueError(f'Reserved source index {source} exceeds dataset count {source_count}')
        for item in values.split(','):
            start, _, end = item.partition('-')
            start, end = int(start), int(end or start)
            if end < start or end - start > 100000:
                raise ValueError(f'Invalid/oversized reserved range: {item}')
            result[source].update(range(start, end+1))
    return dict(result)


def images_for(dataset, split):
    entry = dataset['config'].get(split)
    if entry is None or entry == '':
        return []
    entries = entry if isinstance(entry, list) else [entry]
    result = []
    for raw in entries:
        path = Path(raw)
        if not path.is_absolute():
            path = dataset['root']/path
        if path.is_dir():
            result.extend(p.absolute() for p in sorted(path.rglob('*')) if p.suffix.lower() in SUFFIXES and p.is_file())
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                if not image.is_absolute():
                    image = path.parent/image
                result.append(image.absolute())
        else:
            raise ValueError(f'Missing split directory/image list: {path}')
    return result


def label_for(image):
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts) if part == 'images']
    if not positions:
        raise ValueError(f'Expected an images/ directory in path: {image}')
    parts[positions[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def validate_label(path, dataset):
    raw = path.read_bytes()  # Missing labels are errors, not implicit negatives.
    counts = Counter()
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdecimal():
            raise ValueError(f'Expected detection rows: class x y w h; {path}:{number}')
        cid = int(fields[0])
        if cid not in dataset['names'] or cid in dataset['reserved']:
            raise ValueError(f'Label uses unknown/reserved ID {cid}: {path}:{number}')
        x, y, w, h = map(float, fields[1:])
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 < w <= 1 and 0 < h <= 1)
                or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
            raise ValueError(f'Invalid normalized box: {path}:{number}')
        counts[cid] += 1
    return raw, counts


def merge(datasets, output, copy_images=False, reserved_ids=None):
    sources = [load_dataset(value) for value in datasets]
    requested_reserved = {}
    absent_reserved = {}
    for index, ids in (reserved_ids or {}).items():
        if not isinstance(index, int) or not 1 <= index <= len(sources):
            raise ValueError(f'Invalid reserved source index: {index}')
        source = sources[index-1]
        ids = set(ids)
        requested_reserved[f'd{index}'] = sorted(ids)
        absent_reserved[f'd{index}'] = sorted(ids - source['names'].keys())
        source['reserved'] = ids & source['names'].keys()
    output = Path(output).expanduser().resolve()
    for source in sources:
        for root in (source['yaml'].parent, source['root']):
            if output == root or output in root.parents or root in output.parents:
                raise ValueError('Output must be separate from source dataset directories')
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')
    names = {}
    ignored = {}
    for index, source in enumerate(sources, 1):
        ignored[f'd{index}'] = sorted(source['reserved'])
        for cid, name in source['names'].items():
            if cid in source['reserved']:
                continue
            if cid in names and names[cid] != name:
                raise ValueError(f'Conflicting ID {cid}: {names[cid]!r} versus {name!r}')
            names[cid] = name
    if not names or set(names) != set(range(max(names)+1)):
        raise ValueError('Merged real class IDs must be continuous from 0; reserved names cannot fill gaps')
    records = []
    inode_splits = {}
    seen_paths = set()
    split_stats = {split: {'images': 0, 'negative_images': 0} for split in SPLITS}
    class_images = {split: Counter() for split in SPLITS}
    boxes = {split: Counter() for split in SPLITS}
    for index, source in enumerate(sources, 1):
        for split in SPLITS:
            for image in tqdm(images_for(source, split), desc=f'Check d{index}/{split}', unit='img'):
                stat = image.stat()
                if not image.is_file() or stat.st_size == 0:
                    raise ValueError(f'Missing/empty image: {image}')
                identity = (stat.st_dev, stat.st_ino)
                if stat.st_ino and identity in inode_splits and inode_splits[identity] != split:
                    raise ValueError(f'Same image/hardlink in different splits: {image}')
                inode_splits[identity] = split
                key = (index, split, str(image))
                if key in seen_paths:
                    raise ValueError(f'Duplicate image entry: {image}')
                seen_paths.add(key)
                label = label_for(image)
                raw, counts = validate_label(label, source)
                token = hashlib.sha256(str(image).encode('utf-8')).hexdigest()
                filename = f'd{index}_{token}'
                records.append((split, image, raw, filename))
                split_stats[split]['images'] += 1
                split_stats[split]['negative_images'] += not counts
                class_images[split].update(counts.keys())
                boxes[split].update(counts)
    if not split_stats['train']['images'] or not split_stats['val']['images']:
        raise ValueError('Merged dataset must have non-empty train and val splits')
    output.mkdir(parents=True)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Merge incomplete. Do not train until data.yaml and MERGE_REPORT.md are generated.\n')
    for split in SPLITS:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    for split, image, raw, filename in tqdm(records, desc='Copy' if copy_images else 'Hardlink', unit='img'):
        target = output/'images'/split/(filename+image.suffix.lower())
        if copy_images:
            shutil.copy2(image, target)
        else:
            try:
                os.link(image, target)
            except OSError as exc:
                raise ValueError(f'Hardlink failed: {image}. Sources/output must share a filesystem; '
                                 'use --copy-images to explicitly copy instead. Use a new output for retry.') from exc
        (output/'labels'/split/(filename+'.txt')).write_bytes(raw)
    names = dict(sorted(names.items()))
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'nc': len(names), 'names': names}
    if split_stats['test']['images']:
        config['test'] = 'images/test'
    report = {'sources': [str(s['yaml']) for s in sources], 'class_ids_preserved': True,
              'ignored_reserved_ids': ignored, 'names': names, 'splits': split_stats,
              'requested_reserved_ids': requested_reserved, 'absent_reserved_ids': absent_reserved,
              'reserved_policy': 'explicit source/ID declarations only; no class-name matching',
              'ignored_reserved_names': {f'd{i}': {cid:s['names'][cid] for cid in sorted(s['reserved'])}
                                         for i, s in enumerate(sources, 1)},
              'class_image_counts': class_images, 'class_box_counts': boxes,
              'image_mode': 'copy' if copy_images else 'hardlink',
              'duplicate_check': 'same inode across splits only; no full content hash or visual/semantic audit'}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# YOLO 合并报告', '', f'共 {len(records):,} 张图片，{len(names)} 类。类别 ID 与坐标保持原值；原 train/val/test 划分保留。',
             '', f'图片保存方式：{report["image_mode"]}；图片按来源前缀平铺，标签单独写入。',
             '', '仅检查已知相同文件/硬链接跨划分重复，不校验所有图片内容或判断框的语义正确性。',
             '', '## 显式指定的占位类别', '', '默认不按类别名称忽略任何 ID；以下仅为用户指定且确认未被标签使用的占位项。',
             '', json.dumps(report['ignored_reserved_names'], ensure_ascii=False),
             '', '请求范围中未在对应 YAML 声明的 ID（无需处理）：' + json.dumps(absent_reserved, ensure_ascii=False),
             '', '## 各划分图片数', '', '| 划分 | 图片数 | 负样本 |', '| --- | ---: | ---: |']
    lines += [f'| {s} | {split_stats[s]["images"]} | {split_stats[s]["negative_images"]} |' for s in SPLITS]
    lines += ['', '## 类别统计', '', '| ID | 名称 | train 图片 | val 图片 | test 图片 | 框总数 |', '| ---: | --- | ---: | ---: | ---: | ---: |']
    lines += [f'| {cid} | {name.replace(chr(124), chr(92)+chr(124))} | {class_images["train"][cid]} | {class_images["val"][cid]} | {class_images["test"][cid]} | {sum(boxes[s][cid] for s in SPLITS)} |' for cid, name in names.items()]
    (output/'MERGE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    marker.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs='+', required=True, help='Dataset directories with data.yaml, or YAML paths')
    parser.add_argument('--output', type=Path, required=True, help='New directory, outside source datasets')
    parser.add_argument('--copy-images', action='store_true', help='Copy images instead of hardlinking across filesystems')
    parser.add_argument('--reserved-ids', action='append', default=[],
                        help='Unused padding ID range per source (1-based), e.g. 2:0-26. Only declared IDs are ignored; absent IDs are no-ops. Repeatable.')
    args = parser.parse_args()
    if len(args.datasets) < 2:
        parser.error('Provide at least two datasets')
    try:
        reserved = parse_reserved_ids(args.reserved_ids, len(args.datasets))
        report = merge(args.datasets, args.output, args.copy_images, reserved_ids=reserved)
        print(f'Done: {len(report["names"])} classes; YAML: {args.output / "data.yaml"}')
    except KeyboardInterrupt:
        parser.exit(130, '\nStopped. Source datasets unchanged. Use a new output directory for retry.\n')
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
