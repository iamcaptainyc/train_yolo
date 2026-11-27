"""Merge YOLO detection datasets preserving IDs; reserved IDs require explicit declaration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Hashable
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import yaml
from tqdm import tqdm

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLITS = ('train', 'val', 'test')
DATASETS_DIR = Path(__file__).resolve().parent/'yolo_datasets'


class UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.checked_mappings = set()

    def flatten_mapping(self, node):
        if node not in self.checked_mappings:
            self.checked_mappings.add(node)
            keys = set()
            for key_node, _ in node.value:
                if key_node.tag == 'tag:yaml.org,2002:merge':
                    continue  # Preserve YAML anchor defaults and explicit overrides.
                if key_node.tag == 'tag:yaml.org,2002:value':
                    key_node.tag = 'tag:yaml.org,2002:str'
                key = self.construct_object(key_node)
                if not isinstance(key, Hashable):
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping', node.start_mark,
                        'found unhashable key', key_node.start_mark,
                    )
                if key in keys:
                    raise yaml.constructor.ConstructorError(
                        'while constructing a mapping', node.start_mark,
                        f'duplicate YAML key {key!r}', key_node.start_mark,
                    )
                keys.add(key)
        super().flatten_mapping(node)


def load_dataset(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path/'data.yaml'
    with path.open(encoding='utf-8-sig') as stream:
        config = yaml.load(stream, Loader=UniqueKeyLoader)
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


def parse_reserved_ids(specs, source_count, option='--reserved-ids'):
    result = defaultdict(set)
    for spec in specs or []:
        if not re.fullmatch(r'[1-9]\d*:\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', spec):
            raise ValueError(f'Invalid {option} {spec!r}; use source:IDs, e.g. 2:0-26 or 3:0,2-5')
        source, values = spec.split(':')
        source = int(source)
        if source > source_count:
            raise ValueError(f'Source index {source} exceeds dataset count {source_count} for {option}')
        for item in values.split(','):
            start, _, end = item.partition('-')
            start, end = int(start), int(end or start)
            if end < start or end - start > 100000:
                raise ValueError(f'Invalid/oversized ID range for {option}: {item}')
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


def validate_label(path, dataset, *, check_class_ids=True):
    raw = path.read_bytes()  # Missing labels are errors, not implicit negatives.
    counts = Counter()
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdecimal():
            raise ValueError(f'Expected detection rows: class x y w h; {path}:{number}')
        cid = int(fields[0])
        if (check_class_ids and cid not in dataset['names']) or cid in dataset['reserved']:
            raise ValueError(f'Label uses unknown/reserved ID {cid}: {path}:{number}')
        x, y, w, h = map(float, fields[1:])
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 < w <= 1 and 0 < h <= 1)
                or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
            raise ValueError(f'Invalid normalized box: {path}:{number}')
        counts[cid] += 1
    return raw, counts


def selected_label(raw, selected, id_mapping):
    """Filter rows and rewrite IDs while retaining coordinate strings verbatim."""
    lines, counts = [], Counter()
    for line in raw.decode('utf-8-sig').splitlines(keepends=True):
        if not line.strip():
            continue
        cid = int(line.split()[0])
        if selected is not None and cid not in selected:
            continue
        new_id = id_mapping[cid]
        counts[new_id] += 1
        lines.append(re.sub(r'^(\s*)\d+', lambda m: m[1]+str(new_id), line))
    return ''.join(lines).encode('utf-8'), counts


def parse_id_map(specs):
    mapping = {}
    for spec in specs or ():
        match = re.fullmatch(r'([0-9]+):([0-9]+)', spec)
        if not match:
            raise ValueError(f'Invalid --id-map entry {spec!r}; use OLD:NEW with non-negative integers')
        old, new = map(int, match.groups())
        if old in mapping:
            raise ValueError(f'Duplicate source ID {old} in --id-map')
        mapping[old] = new
    return mapping


def atomic_text(path, content):
    """Replace a small metadata file only after its complete content is written."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_merge_report(report):
    """Render both in-memory statistics and JSON statistics without reading images."""
    if not isinstance(report, dict):
        raise ValueError('Invalid merge report: expected a JSON object')

    def by_id(values, label):
        if not isinstance(values, dict):
            raise ValueError(f'Invalid merge report: missing {label}')
        result = {}
        for key, value in values.items():
            if isinstance(key, bool) or not re.fullmatch(r'\d+', str(key)) or int(key) in result:
                raise ValueError(f'Invalid/duplicate class ID in {label}: {key!r}')
            result[int(key)] = value
        return result

    def count(value, label):
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid merge report count: {label}')
        return value

    names = by_id(report.get('names'), 'names')
    if not names or any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError('Invalid merge report names')
    splits = report.get('splits', {})
    images, boxes, split_counts = {}, {}, {}
    for split in SPLITS:
        if not isinstance(splits.get(split), dict):
            raise ValueError(f'Invalid merge report: missing splits/{split}')
        split_counts[split] = {key: count(splits[split].get(key), f'{split}/{key}')
                               for key in ('images', 'negative_images')}
        if split_counts[split]['negative_images'] > split_counts[split]['images']:
            raise ValueError(f'Negative images exceed total images in {split}')
        for target, key in ((images, 'class_image_counts'), (boxes, 'class_box_counts')):
            values = by_id(report.get(key, {}).get(split), f'{key}/{split}')
            if values.keys() - names.keys():
                raise ValueError(f'Unknown class IDs in {key}/{split}')
            target[split] = Counter({cid: count(value, f'{key}/{split}/{cid}') for cid, value in values.items()})
        for cid in names:
            if images[split][cid] > split_counts[split]['images'] or boxes[split][cid] < images[split][cid]:
                raise ValueError(f'Inconsistent image/box counts for {split}/{cid}')

    def cell(value):
        return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')

    total_images = sum(row['images'] for row in split_counts.values())
    total_boxes = sum(sum(row.values()) for row in boxes.values())
    active = sum(any(boxes[split][cid] for split in SPLITS) for cid in names)
    id_note = '类别 ID 保持原值'
    if report.get('compact_ids'):
        id_note = '类别 ID 已按原 ID 升序重排为连续编号'
    elif report.get('id_offset'):
        id_note = f'类别 ID 统一偏移 {report["id_offset"]:+d}'
    elif not report.get('class_ids_preserved', True):
        id_note = '已应用显式类别 ID 映射'
    lines = ['# YOLO 合并报告', '',
             f'共 **{total_images:,} 张图片、{total_boxes:,} 个框**。标签表 **{len(names)} 类**，实际有样本 **{active} 类**。',
             '', f'{id_note}；保留原 train/val/test 划分及保留框的坐标。',
             '', '统计依据为合并时保存的 report.json，不重新扫描当前图片或标签。',
             '', '## 各划分统计', '',
             '| 划分 | 图片数 | 比例 | 负样本 | 框数 | 跳过的跨划分重复图片 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    filtering = report.get('filtering', {})
    skipped_total = 0
    for split in SPLITS:
        skipped = sum(source.get(split, {}).get('excluded_cross_split_duplicates', 0) for source in filtering.values())
        skipped_total += skipped
        fraction = 100 * split_counts[split]['images'] / total_images if total_images else 0
        lines.append(f'| {split} | {split_counts[split]["images"]:,} | {fraction:.2f}% | '
                     f'{split_counts[split]["negative_images"]:,} | {sum(boxes[split].values()):,} | {skipped:,} |')
    lines += [f'| 合计 | {total_images:,} | {"100.00" if total_images else "0.00"}% | '
              f'{sum(row["negative_images"] for row in split_counts.values()):,} | {total_boxes:,} | {skipped_total:,} |',
              '', '## 类别统计', '',
              '“图片合计”为包含该类的图片数；多类别同图会分别计入各类，不能将各类图片数相加作为数据集总数。',
              '', '| ID | 名称 | train 图片 | val 图片 | test 图片 | 图片合计 | 框总数 |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for cid, name in sorted(names.items()):
        counts = [images[split][cid] for split in SPLITS]
        values = ' | '.join(f'{value:,}' for value in (*counts, sum(counts), sum(boxes[s][cid] for s in SPLITS)))
        lines.append(f'| {cid} | {cell(name)} | {values} |')
    empty = [f'{cid}: {name}' for cid, name in sorted(names.items()) if not any(boxes[s][cid] for s in SPLITS)]
    if empty:
        lines += ['', '没有样本的类别：' + '、'.join(cell(name) for name in empty) + '。']

    lines += ['', '## 来源统计', '', '| 来源 | train 图片 | val 图片 | test 图片 | 合计 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for index, source in enumerate(report.get('sources', []), 1):
        tag = f'd{index}'
        filename = str(source).replace('\\', '/').rstrip('/').rsplit('/', 2)
        label = filename[-2] if len(filename) > 1 else filename[-1]
        stats = filtering.get(tag)
        if stats is None:
            lines.append(f'| {tag}: {cell(label)} | — | — | — | — |')
        else:
            counts = [stats.get(split, {}).get('kept_images', 0) for split in SPLITS]
            lines.append(f'| {tag}: {cell(label)} | ' + ' | '.join(f'{value:,}' for value in (*counts, sum(counts))) + ' |')

    lines += ['', '## 附录：合并规则', '',
              f'图片保存方式：{cell(report.get("image_mode", "未记录"))}；图片按来源前缀平铺，标签单独写入。',
              '', '同一文件或硬链接出现在多个划分时，按 test → val → train 保留，跳过低优先级图片和标签。',
              '', '只检查文件身份，不对独立复制的图片做内容去重或语义审查。',
              '', '### 显式指定的占位类别', '',
              '默认不按名称忽略 ID；以下仅为显式指定且未被标签使用的占位项。',
              '', json.dumps(report.get('ignored_reserved_names', {}), ensure_ascii=False),
              '', '范围中未声明、无需处理的 ID：' + json.dumps(report.get('absent_reserved_ids', {}), ensure_ascii=False)]
    padding = report.get('output_padding_ids', [])
    if padding:
        lines += ['', f'输出 data.yaml 含 {len(padding)} 个未使用占位 ID，详见 report.json 的 output_padding_ids。']
    duplicate_names = report.get('same_name_different_ids', {})
    if duplicate_names:
        lines += ['', '以下同名类别分属多个 ID，各 ID 分开统计：', '', json.dumps(duplicate_names, ensure_ascii=False)]
    included = report.get('included_ids', {})
    has_includes = any(value is not None for value in included.values())
    if has_includes:
        lines += ['', '### 类别筛选与编号映射', '',
                  '指定来源只保留所选类别框，混合类别图片保留；未选类别框不属于本次检测任务。',
                  '不含所选类别的正样本不输出；是否保留原始负样本：' + str(report.get('keep_original_negatives_when_filtering', False)),
                  '', '各来源选择（null 为全部）：' + json.dumps(included, ensure_ascii=False)]
    if has_includes or report.get('compact_ids') or report.get('id_offset') or not report.get('class_ids_preserved', True):
        mapping = by_id(report.get('id_mapping', {}), 'id_mapping')
        original_names = by_id(report.get('original_names', {}), 'original_names')
        lines += ['', '| 原 ID | 输出 ID | 名称 |', '| ---: | ---: | --- |']
        lines += [f'| {cid} | {new} | {cell(original_names.get(cid, "未记录"))} |' for cid, new in sorted(mapping.items())]
    lines += ['', '### 各来源完整路径', '']
    lines += [f'- {cell(source)}' for source in report.get('sources', [])]
    return '\n'.join(lines) + '\n'


def regenerate_report(dataset):
    """Recover Markdown from the existing JSON; never touch dataset payload or status."""
    dataset = Path(dataset).expanduser().resolve()
    path = dataset/'report.json'
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    target = dataset/'MERGE_REPORT.md'
    atomic_text(target, render_merge_report(report))
    return target


def merge(datasets, output, copy_images=False, reserved_ids=None, include_ids=None,
          compact_ids=False, keep_negatives=False, id_offset=0, id_map=None):
    if type(id_offset) is not int:
        raise ValueError('ID offset must be an integer')
    if id_offset and compact_ids:
        raise ValueError('--id-offset cannot be combined with --compact-ids')
    id_map = id_map or {}
    if id_map and (compact_ids or id_offset):
        raise ValueError('--id-map cannot be combined with --compact-ids or a nonzero --id-offset')
    if any(type(old) is not int or type(new) is not int or old < 0 or new < 0
           for old, new in id_map.items()):
        raise ValueError('--id-map requires non-negative integer IDs')
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
    for source in sources:
        source['include'] = None
    for index, ids in (include_ids or {}).items():
        if not isinstance(index, int) or not 1 <= index <= len(sources):
            raise ValueError(f'Invalid include source index: {index}')
        source, ids = sources[index-1], set(ids)
        if not ids or ids - source['names'].keys():
            raise ValueError(f'Included IDs must exist in source {index}: {sorted(ids - source["names"].keys())}')
        if ids & source['reserved']:
            raise ValueError(f'Included IDs overlap reserved IDs in source {index}: {sorted(ids & source["reserved"])}')
        source['include'] = ids
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
            if cid in source['reserved'] or (source['include'] is not None and cid not in source['include']):
                continue
            if cid in names and names[cid] != name:
                raise ValueError(f'Conflicting ID {cid}: {names[cid]!r} versus {name!r}')
            names[cid] = name
    if not names:
        raise ValueError('No real class IDs remain')
    first_id = min(names) if id_offset else 0
    if not (compact_ids or id_map) and max(names) - first_id + 1 != len(names):
        if id_offset:
            raise ValueError('Class IDs must be consecutive for --id-offset; it preserves the distance between IDs')
        raise ValueError('Merged real class IDs must be continuous from 0; reserved names cannot fill gaps. '
                         'Use --compact-ids to explicitly renumber a selected subset.')
    if first_id + id_offset < 0:
        raise ValueError('--id-offset would produce negative class IDs')
    unknown = id_map.keys() - names.keys()
    if unknown:
        raise ValueError(f'--id-map source IDs must be retained after filtering: {sorted(unknown)}')
    id_mapping = {cid: new if compact_ids else id_map.get(cid, cid + id_offset)
                  for new, cid in enumerate(sorted(names))}
    original_names = dict(names)
    by_target = {}
    for old, new in id_mapping.items():
        if new in by_target:
            raise ValueError(f'Output ID {new} conflicts: input IDs {by_target[new]} and {old}')
        by_target[new] = old
    names = {new: original_names[by_target[new]] for new in sorted(by_target)}
    ids_by_name = defaultdict(list)
    for cid, name in names.items():
        ids_by_name[name].append(cid)
    duplicate_names = {name: ids for name, ids in ids_by_name.items() if len(ids) > 1}
    padding = {cid: f'reserved_{cid}' for cid in range(max(names)) if cid not in names}
    records = []
    inode_splits = {}
    seen_paths = set()
    split_stats = {split: {'images': 0, 'negative_images': 0} for split in SPLITS}
    class_images = {split: Counter() for split in SPLITS}
    boxes = {split: Counter() for split in SPLITS}
    filtering = {f'd{i}': {split: {'scanned_images':0, 'kept_images':0,
                                  'excluded_without_selected_class':0, 'excluded_negatives':0,
                                  'excluded_cross_split_duplicates':0,
                                  'removed_boxes':0} for split in SPLITS} for i in range(1,len(sources)+1)}
    # Evaluate all higher-priority splits before considering any lower-priority source.
    split_priority = ('test', 'val', 'train')
    for split in split_priority:
        for index, source in enumerate(sources, 1):
            for image in tqdm(images_for(source, split), desc=f'Check d{index}/{split}', unit='img'):
                stat = image.stat()
                if not image.is_file() or stat.st_size == 0:
                    raise ValueError(f'Missing/empty image: {image}')
                key = (index, split, str(image))
                if key in seen_paths:
                    raise ValueError(f'Duplicate image entry: {image}')
                seen_paths.add(key)
                stats = filtering[f'd{index}'][split]
                stats['scanned_images'] += 1
                identity = (stat.st_dev, stat.st_ino)
                preferred_split = inode_splits.get(identity)
                if preferred_split is not None and preferred_split != split:
                    stats['excluded_cross_split_duplicates'] += 1
                    continue
                raw, counts = validate_label(label_for(image), source)
                selected = source['include']
                if selected is not None and not (selected & counts.keys()):
                    if counts or not keep_negatives:
                        reason = 'excluded_without_selected_class' if counts else 'excluded_negatives'
                        stats[reason] += 1
                        stats['removed_boxes'] += sum(counts.values())
                        continue
                if selected is not None or compact_ids or id_offset or id_map:
                    total_before = sum(counts.values())
                    raw, counts = selected_label(raw, selected, id_mapping)
                    stats['removed_boxes'] += total_before-sum(counts.values())
                stats['kept_images'] += 1
                if stat.st_ino:
                    inode_splits[identity] = split
                token = hashlib.sha256(str(image).encode('utf-8')).hexdigest()
                filename = f'd{index}_{token}'
                records.append((split, image, raw, filename))
                split_stats[split]['images'] += 1
                split_stats[split]['negative_images'] += not counts
                class_images[split].update(counts.keys())
                boxes[split].update(counts)
    if not split_stats['train']['images'] or not split_stats['val']['images']:
        raise ValueError('Merged dataset must have non-empty train and val splits after filtering and duplicate removal')
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
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'nc': len(padding) + len(names), 'names': dict(sorted((padding | names).items()))}
    if split_stats['test']['images']:
        config['test'] = 'images/test'
    report = {'sources': [str(s['yaml']) for s in sources],
              'class_ids_preserved': all(cid==new for cid,new in id_mapping.items()),
              'compact_ids': compact_ids, 'id_offset': id_offset, 'id_mapping': id_mapping,
              'original_names': original_names, 'output_padding_ids': list(padding),
              'same_name_different_ids': duplicate_names,
              'included_ids': {f'd{i}':sorted(s['include']) if s['include'] is not None else None
                               for i,s in enumerate(sources,1)},
              'filtering': filtering, 'keep_original_negatives_when_filtering':keep_negatives,
              'mixed_image_policy': 'keep image and selected boxes; omit other classes from the detection task',
              'ignored_reserved_ids': ignored, 'names': names, 'splits': split_stats,
              'requested_reserved_ids': requested_reserved, 'absent_reserved_ids': absent_reserved,
              'reserved_policy': 'explicit source/ID declarations only; no class-name matching',
              'ignored_reserved_names': {f'd{i}': {cid:s['names'][cid] for cid in sorted(s['reserved'])}
                                         for i, s in enumerate(sources, 1)},
              'class_image_counts': class_images, 'class_box_counts': boxes,
              'image_mode': 'copy' if copy_images else 'hardlink',
              'duplicate_check': 'same inode across splits only; no full content hash or visual/semantic audit',
              'cross_split_priority': split_priority}
    # Render before saving the snapshot, so formatting failures cannot leave a
    # newly written report.json followed by a predictable rendering exception.
    markdown = render_merge_report(report)
    atomic_text(output/'report.json', json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    atomic_text(output/'data.yaml', yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    try:
        atomic_text(output/'MERGE_REPORT.md', markdown)
    except OSError as exc:
        raise ValueError('Images, labels and report.json were saved, but writing MERGE_REPORT.md failed. '
                         f'Fix the write error ({exc}), then run: python merge_yolo_ids.py '
                         f'--report-only "{output}". INCOMPLETE.txt is retained.') from exc
    marker.unlink()
    return report


def read_merge_config(path):
    path = path.expanduser().resolve()
    with path.open(encoding='utf-8-sig') as stream:
        config = yaml.load(stream, Loader=UniqueKeyLoader)
    list_keys = ('datasets', 'include_ids', 'reserved_ids', 'id_map')
    flag_keys = ('copy_images', 'compact_ids', 'keep_negatives')
    if not isinstance(config, dict) or config.keys() - {*list_keys, *flag_keys, 'output', 'id_offset'}:
        raise ValueError('Unexpected config fields; see configs/merge_yolo_ids.yaml')
    if 'id_offset' in config and type(config['id_offset']) is not int:
        raise ValueError('Config id_offset must be an integer')
    for key in list_keys:
        if key in config and (not isinstance(config[key], list)
                or any(not isinstance(value, str) or not value.strip() for value in config[key])):
            raise ValueError(f'Config {key} must be a list of non-empty strings')
    for key in flag_keys:
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'Config {key} must be true or false')
    if 'output' in config:
        if not isinstance(config['output'], str) or not config['output'].strip():
            raise ValueError('Config output must be a path')
        config['output'] = path.parent/Path(config['output']).expanduser()
    if 'datasets' in config:
        datasets = []
        for value in config['datasets']:
            dataset_path = Path(value).expanduser()
            resolved = path.parent/dataset_path
            if (value not in ('.', '..') and not any(char in value for char in '/\\:')
                    and dataset_path.suffix.lower() not in ('.yaml', '.yml')):
                candidate = DATASETS_DIR/value
                if candidate.is_dir() or not resolved.exists():
                    resolved = candidate
            datasets.append(resolved)
        config['datasets'] = datasets
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', type=Path, metavar='DATASET',
                        help='Regenerate MERGE_REPORT.md from an existing report.json; no scans or merge')
    parser.add_argument('--config', type=Path, help='Merge YAML; explicit CLI options override its settings')
    parser.add_argument('--datasets', type=Path, nargs='+', help='Dataset directories with data.yaml, or YAML paths')
    parser.add_argument('--output', type=Path, help='New directory, outside source datasets')
    parser.add_argument('--copy-images', action=argparse.BooleanOptionalAction, help='Copy images across filesystems')
    parser.add_argument('--reserved-ids', action='extend', nargs='+',
                        help='One or more unused padding ID ranges per source (1-based), e.g. 1:27-82 2:0-26. Only declared IDs are ignored; absent IDs are no-ops. Repeatable.')
    parser.add_argument('--include-ids', action='extend', nargs='+',
                        help='One or more source class selections, e.g. 1:0,25,26 2:27-35; other sources remain unfiltered. Repeatable.')
    parser.add_argument('--compact-ids', action=argparse.BooleanOptionalAction, help='Renumber selected global IDs to 0..N-1')
    parser.add_argument('--id-offset', type=int, help='Add the same signed integer to all selected class IDs')
    parser.add_argument('--id-map', action='extend', nargs='+', metavar='OLD:NEW',
                        help='Set individual global class IDs, e.g. 3:10 5:2. Unspecified IDs stay the same. Repeatable.')
    parser.add_argument('--keep-negatives', action=argparse.BooleanOptionalAction, help='Keep original empty-label images from class-filtered sources')
    args = parser.parse_args()
    try:
        if args.report_only is not None:
            if any(getattr(args, key) is not None for key in
                   ('config', 'datasets', 'output', 'copy_images', 'reserved_ids', 'include_ids',
                    'compact_ids', 'id_offset', 'id_map', 'keep_negatives')):
                parser.error('--report-only cannot be combined with merge options')
            target = regenerate_report(args.report_only)
            print(f'Report regenerated: {target}\nImages, labels, YAML and completion markers unchanged.')
            return
        if args.config:
            for key, value in read_merge_config(args.config).items():
                if getattr(args, key) is None:
                    setattr(args, key, value)
        if not args.datasets or args.output is None:
            parser.error('datasets and output are required in the config or on the command line')
        reserved = parse_reserved_ids(args.reserved_ids, len(args.datasets))
        included = parse_reserved_ids(args.include_ids, len(args.datasets), option='--include-ids')
        report = merge(args.datasets, args.output, args.copy_images, reserved_ids=reserved,
                       include_ids=included, compact_ids=args.compact_ids, keep_negatives=args.keep_negatives,
                       id_offset=0 if args.id_offset is None else args.id_offset,
                       id_map=parse_id_map(args.id_map))
        print(f'Done: {len(report["names"])} classes; YAML: {args.output / "data.yaml"}')
    except KeyboardInterrupt:
        parser.exit(130, '\nStopped. Source datasets unchanged. Use a new output directory for retry.\n')
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
