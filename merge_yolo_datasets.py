"""Merge camping LILA and Animals Detection YOLO datasets, preserving splits and hardlinking images."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import CancelledError
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

from filelock import FileLock
import yaml
from tqdm import tqdm

from lila_camping import load_policy
from lila_parallel import bounded_map
from lila_state import atomic_text, fingerprint, same_content_attributes, signature
from prepare_dataset import IMAGE_SUFFIXES

SPLITS = ('train', 'val', 'test')
# English application names, not scientific species guesses.
ALIASES = {
    'bull': 'cattle', 'cow': 'cattle', 'domestic_cattle': 'cattle',
    'domestic_dog': 'dog', 'domestic_cat': 'cat',
    'mountain_lion': 'cougar', 'puma': 'cougar',
    'wild_pig': 'wild_boar', 'eurasian_wild_pig': 'wild_boar', 'bushpig': 'wild_boar',
    'moths_and_butterflies': 'insect', 'human': 'person', 'car': 'vehicle',
}
EXCLUDED = {'animal', 'mammal', 'small_mammal', 'unidentified_animal',
            'unidentified_mammal', 'unidentified_small_mammal', 'unidentified_small_animal', 'other_animal'}
ANIMALS_NAMES = '''bear,brown bear,bull,butterfly,camel,canary,caterpillar,cattle,centipede,cheetah,chicken,crab,crocodile,deer,duck,eagle,elephant,fish,fox,frog,giraffe,goat,goldfish,goose,hamster,harbor seal,hedgehog,hippopotamus,horse,jaguar,jellyfish,kangaroo,koala,ladybug,leopard,lion,lizard,lynx,magpie,monkey,moths and butterflies,mouse,mule,ostrich,otter,owl,panda,parrot,penguin,pig,polar bear,rabbit,raccoon,raven,red panda,rhinoceros,scorpion,sea lion,sea turtle,seahorse,shark,sheep,shrimp,snail,snake,sparrow,spider,squid,squirrel,starfish,swan,tick,tiger,tortoise,turkey,turtle,whale,woodpecker,worm,zebra'''.split(',')


def normalized(name):
    value = re.sub(r'[\s_-]+', '_', str(name).strip().casefold()).replace('grey', 'gray')
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*', value):
        raise ValueError(f'Expected reviewed English common names, got {name!r}; convert old scientific LILA classes with --remap-from first')
    return ALIASES.get(value, value)


def class_metadata():
    result = dict(load_policy()['classes'])
    for raw in ANIMALS_NAMES:
        name = normalized(raw)
        result.setdefault(name, {'chinese': '', 'group': 'animal', 'parent': None})
    groups = {
        'bear': 'brown_bear polar_bear panda',
        'cat_family': 'cheetah jaguar lion tiger',
        'ungulate': 'camel giraffe hippopotamus mule rhinoceros sheep',
        'other_mammal': 'hamster hedgehog kangaroo koala red_panda',
        'bird': 'canary duck goose ostrich parrot penguin raven sparrow swan',
        'reptile': 'crocodile sea_turtle tortoise frog',
        'aquatic': 'fish goldfish harbor_seal sea_lion seahorse shark whale',
        'invertebrate': 'butterfly caterpillar centipede crab jellyfish ladybug scorpion shrimp snail spider squid starfish tick worm',
    }
    for group, names in groups.items():
        for name in names.split():
            result[name]['group'] = group
    for name in ('brown_bear', 'polar_bear', 'panda'):
        result[name]['parent'] = 'bear'
    for name in ('canary duck goose ostrich parrot penguin raven sparrow swan').split():
        result[name]['parent'] = 'bird'
    for name in ('goldfish', 'shark', 'seahorse'):
        result[name]['parent'] = 'fish'
    for name in ('sea_turtle', 'tortoise'):
        result[name]['parent'] = 'turtle'
    for name in ('butterfly', 'caterpillar', 'ladybug'):
        result[name]['parent'] = 'insect'
    result['hamster'].update(group='rodent', parent='rodent')
    result['frog']['group'] = 'amphibian'
    translations = '''熊,棕熊,公牛,蝴蝶,骆驼,金丝雀,毛毛虫,家牛,蜈蚣,猎豹,家鸡,蟹,鳄鱼,鹿,鸭,鹰,象,鱼,狐,蛙,长颈鹿,山羊,金鱼,鹅,仓鼠,港海豹,刺猬,河马,马,美洲豹,水母,袋鼠,考拉,瓢虫,豹,狮,蜥蜴,猞猁,鹊类,猴,蛾与蝴蝶,鼠,骡,鸵鸟,水獭,鸮类,大熊猫,鹦鹉,企鹅,猪,北极熊,兔,浣熊,渡鸦,小熊猫,犀牛,蝎子,海狮,海龟,海马,鲨,绵羊,虾,蜗牛,蛇,麻雀,蜘蛛,鱿鱼,松鼠,海星,天鹅,蜱,虎,陆龟,火鸡,龟,鲸,啄木鸟,蠕虫,斑马'''.split(',')
    for raw, zh in zip(ANIMALS_NAMES, translations, strict=True):
        info = result[normalized(raw)]
        if not info['chinese']:
            info['chinese'] = zh
    return result


def load_dataset(value, alias, known):
    path = Path(value).resolve()
    if path.is_dir():
        path = path / 'data.yaml'
    if not path.is_file():
        raise ValueError(f'Missing dataset YAML: {path}')
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError(f'Dataset YAML must be an object: {path}')
    names = config.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError(f'Missing names in {path}')
    try:
        original = {int(k): v for k, v in names.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid class IDs in {path}') from exc
    if len(original) != len(names) or set(original) != set(range(len(original))):
        raise ValueError(f'Class IDs must be contiguous from zero: {path}')
    if any(not isinstance(name, str) or not name.strip() for name in original.values()):
        raise ValueError(f'Class names must be non-empty strings: {path}')
    if 'nc' in config and config['nc'] != len(original):
        raise ValueError(f'nc does not match names: {path}')
    mapping = {}
    for old_id, name in original.items():
        target = normalized(name)
        if target not in known and target not in EXCLUDED:
            raise ValueError(f'Unreviewed class {alias}:{old_id} {name!r}; add an explicit common-name mapping before merging')
        mapping[old_id] = None if target in EXCLUDED else target
    root = Path(config.get('path') or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    # Converted datasets store absolute server paths. Support moving the entire
    # folder while keeping its images/labels and relative split settings together.
    if not root.is_dir() and all(isinstance(config.get(s), str) and (path.parent / config[s]).is_dir() for s in SPLITS):
        root = path.parent
    directories = {}
    for split in SPLITS:
        entry = config.get(split)
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f'{path}: {split} must be an image directory (lists/URL/txt splits are not supported)')
        directory = Path(entry)
        directory = (directory if directory.is_absolute() else root / directory).resolve()
        if not directory.is_dir():
            raise ValueError(f'Missing {alias}/{split} image directory: {directory}')
        directories[split] = directory
    return {'alias': alias, 'yaml': path, 'root': root, 'directories': directories,
            'original_names': original, 'mapping': mapping, 'yaml_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def label_for(image):
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts[:-1]) if part == 'images']
    if not positions:
        raise ValueError(f'Expected a YOLO images/ directory in {image}')
    parts[positions[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def parse_label(path, mapping):
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5 or not re.fullmatch(r'\d+', tokens[0]):
            raise ValueError(f'{path}:{number}: expected class_id cx cy width height')
        old = int(tokens[0])
        if old not in mapping:
            raise ValueError(f'{path}:{number}: unknown class ID {old}')
        values = [float(v) for v in tokens[1:]]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or min(values[2:]) <= 0:
            raise ValueError(f'{path}:{number}: invalid normalized YOLO coordinates')
        x, y, w, h = values
        if min(x - w / 2, y - h / 2) < -1.1e-8 or max(x + w / 2, y + h / 2) > 1 + 1.1e-8:
            raise ValueError(f'{path}:{number}: box lies outside normalized image bounds')
        if mapping[old] is None:
            raise ValueError(f'{path}:{number}: excluded uncertain animal category; exclude whole image')
        # Keep the exact coordinate tokens. Only the class ID will change.
        rows.append((mapping[old], tuple(tokens[1:])))
    return rows


def unique_rows(rows):
    """Aliases can produce the same class/box twice; keep its first unchanged tokens."""
    result, seen = [], set()
    for name, coords in rows:
        identity = (name, tuple(float(x) for x in coords))
        if identity not in seen:
            seen.add(identity)
            result.append((name, coords))
    return result


def scan_one(task):
    dataset, split, image = task
    relative = image.relative_to(dataset['directories'][split]).as_posix()
    record = {'dataset': dataset['alias'], 'split': split, 'relative': relative, 'image': image}
    try:
        record['label'] = label_for(image)
        if not record['label'].is_file():
            raise ValueError('missing label file; never assume a missing positive label is a negative')
        record['image_stamp'] = fingerprint(image)
        if record['image_stamp']['size'] <= 0:
            raise ValueError('empty image file')
        record['label_stamp'] = fingerprint(record['label'])
        rows = parse_label(record['label'], dataset['mapping'])
        record['rows'] = unique_rows(rows)
        record['duplicate_boxes_removed'] = len(rows) - len(record['rows'])
        if not same_content_attributes(record['label_stamp'], fingerprint(record['label'])):
            raise ValueError('label changed while scanning')
        # Namespace by dataset/split and full relative path, including extension.
        stem = hashlib.sha256(f"{dataset['alias']}/{split}/{relative}".encode()).hexdigest()
        record['image_rel'] = Path('images') / split / dataset['alias'] / (stem + image.suffix.lower())
        record['label_rel'] = Path('labels') / split / dataset['alias'] / (stem + '.txt')
    except (OSError, ValueError) as exc:
        record['error'] = str(exc)
    return record


def exclusions_for_duplicates(records, content_hashes=False):
    """Return indices to exclude. Hashes, when requested, exist in memory only."""
    by_identity, by_size = defaultdict(list), defaultdict(list)
    for i, r in enumerate(records):
        info = r['image_stamp']
        if info['inode']:
            by_identity[(info['device'], info['inode'])].append(i)
        by_size[info['size']].append(i)
    groups = list(by_identity.values())
    hashed = 0
    if content_hashes:
        by_digest = defaultdict(list)
        # Exact copies have equal sizes; unique-size files cannot be exact duplicates.
        for indices in tqdm(by_size.values(), desc='Check duplicate content (equal-size candidates)', unit='size'):
            if len(indices) < 2:
                continue
            seen_identity = {}
            for i in indices:
                r, info = records[i], records[i]['image_stamp']
                identity = (info['device'], info['inode']) if info['inode'] else ('path', str(r['image']))
                if identity not in seen_identity:
                    with r['image'].open('rb') as stream:
                        seen_identity[identity] = hashlib.file_digest(stream, 'sha256').hexdigest()
                    hashed += 1
                    if not same_content_attributes(info, fingerprint(r['image'])):
                        raise ValueError(f'Image changed during duplicate check: {r["image"]}')
                by_digest[seen_identity[identity]].append(i)
        groups += list(by_digest.values())
    rejected = {}
    for indices in groups:
        if len(indices) < 2:
            continue
        if len({records[i]['split'] for i in indices}) > 1:
            for i in indices:
                rejected[i] = 'same image occurs across original splits; exclude all affected records without moving splits'
        else:
            labels = {tuple(sorted((name, tuple(float(x) for x in coords)) for name, coords in records[i]['rows'])) for i in indices}
            if len(labels) > 1:
                for i in indices:
                    rejected.setdefault(i, 'same image has conflicting unified labels; exclude all affected records')
    return rejected, hashed


def assert_unchanged(record):
    for name in ('image', 'label'):
        if not same_content_attributes(record[name + '_stamp'], fingerprint(record[name])):
            raise ValueError(f'Source changed after label scan: {record[name]}')


def output_text(record, ids):
    return ''.join(f'{ids[name]} ' + ' '.join(coords) + '\n' for name, coords in record['rows'])


def prepare_output_directories(records, output):
    """Create each directory once; clean atomic label leftovers with one scan.

    Never glob the growing flat label directory per image: with N labels that
    would enumerate roughly N**2 entries, even when no temp file exists.
    """
    image_dirs = {output / r['image_rel'].parent for r in records}
    label_dirs = {output / r['label_rel'].parent for r in records}
    for directory in sorted(image_dirs | label_dirs):
        directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for directory in sorted(label_dirs):
        with os.scandir(directory) as entries:
            for entry in entries:
                # Only this merger's generated atomic label files are owned.
                if re.fullmatch(r'\.[0-9a-f]{64}\.txt\.[0-9a-f]{32}\.tmp', entry.name):
                    if entry.is_file(follow_symlinks=False):
                        (directory / entry.name).unlink()
                        removed += 1
    return removed


def write_one(record, output, ids):
    assert_unchanged(record)
    image, label = output / record['image_rel'], output / record['label_rel']
    text = output_text(record, ids)
    temporary = image.with_name(image.name + '.merge-part')
    temporary_label = label.with_name(label.name + '.merge-part')
    temporary.unlink(missing_ok=True)
    temporary_label.unlink(missing_ok=True)
    image_hit = image.exists() and os.path.samefile(image, record['image'])
    label_hit = label.is_file() and label.read_text(encoding='utf-8') == text
    if not image_hit:
        try:
            os.link(record['image'], temporary)  # No copy fallback, including across disks.
            os.replace(temporary, image)
        finally:
            temporary.unlink(missing_ok=True)
    if not label_hit:
        # Parent directories already exist. A stable per-record temp path avoids
        # repeated mkdir calls and per-image wildcard cleanup on resume.
        try:
            temporary_label.write_text(text, encoding='utf-8')
            os.replace(temporary_label, label)
        finally:
            temporary_label.unlink(missing_ok=True)
    assert_unchanged(record)
    return image_hit and label_hit


def audit_one(record, output, ids, source_mapping):
    assert_unchanged(record)
    if not os.path.samefile(record['image'], output / record['image_rel']):
        raise ValueError(f'Output is not a source hardlink: {record["image_rel"]}')
    expected = unique_rows(parse_label(record['label'], source_mapping))
    actual = parse_label(output / record['label_rel'], {i: name for name, i in ids.items()})
    if expected != actual:
        raise ValueError(f'Written class/coordinate tokens do not match source: {record["label_rel"]}')
    assert_unchanged(record)


def make_report(datasets, records, excluded, ids, metadata, reused, hashes, content_check):
    stats = {s: {'images': 0, 'negative_images': 0, 'boxes': Counter(), 'class_images': Counter()} for s in SPLITS}
    sources = {d['alias']: {s: {'images': 0, 'negative_images': 0, 'boxes': 0} for s in SPLITS} for d in datasets}
    for r in records:
        s = stats[r['split']]
        s['images'] += 1
        s['negative_images'] += not r['rows']
        s['boxes'].update(name for name, _ in r['rows'])
        s['class_images'].update({name for name, _ in r['rows']})
        d = sources[r['dataset']][r['split']]
        d['images'] += 1
        d['negative_images'] += not r['rows']
        d['boxes'] += len(r['rows'])
    return {
        'schema': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'classes': list(ids), 'stats': stats, 'dataset_split_stats': sources,
        'source_yaml_files': {d['alias']: str(d['yaml']) for d in datasets},
        'source_class_mapping': {d['alias']: [
            {'source_id': i, 'source_name': name, 'target': d['mapping'][i], 'target_id': ids.get(d['mapping'][i])}
            for i, name in sorted(d['original_names'].items())
        ] for d in datasets},
        'hierarchy': {name: metadata[name] for name in ids},
        'excluded_images': len(excluded),
        'excluded_by_dataset': dict(Counter(r['dataset'] for r in excluded)),
        'excluded_by_reason': dict(Counter(r['error'] for r in excluded)),
        'missing_classes': {s: sorted(set(ids) - stats[s]['boxes'].keys()) for s in SPLITS},
        'audit': {'images_verified': len(records), 'boxes_verified': sum(len(r['rows']) for r in records),
                  'duplicate_box_records_removed': sum(r['duplicate_boxes_removed'] for r in records),
                  'image_storage': 'hardlink', 'resumed_output_files': reused, 'source_splits_preserved': True,
                  'coordinate_tokens_preserved': True, 'image_decodes': 0, 'image_content_hashes': hashes,
                  'duplicate_check': 'sha256_equal_size_candidates_and_file_identity' if content_check else 'file_identity_only',
                  'independent_copied_image_duplicates_checked': content_check,
                  'same_split_identical_duplicates': 'retained as independent source records'},
    }


def markdown(report):
    lines = ['# YOLO 数据集合并报告', '',
             '状态：合并检查通过。LILA 与 Animals Detection 保留各自原 train/val/test；图片为硬链接，统一类别 ID 并去除合并后同类同框的重复记录，保留框的原坐标文本逐项核对。', '',
             '没有重新解码图片，不重新认证原始物种、框位置或图片完整性。两个输入应先完成各自的转换检查。', '',
             '## 划分统计', '', '| 来源 | 划分 | 图片 | 负样本 | 框 |', '| --- | --- | ---: | ---: | ---: |']
    for ds, splits in report['dataset_split_stats'].items():
        for s, value in splits.items():
            lines.append(f"| {ds} | {s} | {value['images']} | {value['negative_images']} | {value['boxes']} |")
    for s, v in report['stats'].items():
        lines.append(f"| 合计 | {s} | {v['images']} | {v['negative_images']} | {sum(v['boxes'].values())} |")
    lines += ['', '## 统一英文类别', '', '| 新 ID | 英文类别 | 中文含义 | 总框 | train | val | test |', '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for i, name in enumerate(report['classes']):
        counts = [report['stats'][s]['boxes'].get(name, 0) for s in SPLITS]
        lines.append(f"| {i} | {name} | {report['hierarchy'][name].get('chinese', '')} | {sum(counts)} | " + ' | '.join(map(str, counts)) + ' |')
    lines += ['', '## 原 ID → 新 ID', '', '| 来源 | 原 ID | 原名称 | 统一名称 | 新 ID |', '| --- | --- | --- | --- | --- |']
    for ds, rows in report['source_class_mapping'].items():
        for r in rows:
            name = r['source_name'].replace('|', '\\|')
            lines.append(f"| {ds} | {r['source_id']} | {name} | {r['target'] or '排除'} | {r['target_id'] if r['target_id'] is not None else '无保留样本'} |")
    lines += ['', '## 检查范围与排除', '',
              '`pig` 保持家野未明确的猪；`wild_boar` 为约定的野生猪类，二者分开。`bull/cow/cattle` 合为 `cattle`；`moths and butterflies` 归 `insect`；不猜更细物种。', '',
              '原始空标签保留为负样本；缺失标签、非法标签、排除类别及已发现的跨划分重复/标签冲突均整图排除，完整清单见 excluded_images.jsonl。', '',
              '没有移动图片到其他划分。相同划分内同图同标签的重复记录保留；没有训练样本的类别整图排除并记录。', '',
              '内容重复检查：' + ('已对等大小候选图片计算 SHA-256，检查字节相同的跨划分重复；不能发现不同编码的近似图。' if report['audit']['independent_copied_image_duplicates_checked'] else '仅检查相同设备/inode 的文件身份，未检查独立复制文件的字节重复；不能声称不存在全部跨数据集内容泄漏。需要时使用 --check-content-hashes。'), '',
              f"排除图片：{report['excluded_images']}。", '', '```json', json.dumps(report['audit'], ensure_ascii=False, indent=2), '```', '',
              '源目录删除说明：完成且 data.yaml、report.json、MERGE_REPORT.md 已生成后，合并后的硬链接不依赖输入图片路径继续存在；源图删除不会删除合并图的内容。重跑合并仍依赖输入目录，因此删除后不能再从这些输入续跑。硬链接目录删除通常不会释放图片主体空间。', '']
    return '\n'.join(lines)


def merge(lila, animals, output, *, workers=8, check_content_hashes=False):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('--workers must be a positive integer')
    output = Path(output).resolve()
    metadata = class_metadata()
    datasets = [load_dataset(lila, 'lila', metadata), load_dataset(animals, 'animals', metadata)]
    for d in datasets:
        for source in (d['root'], d['yaml'].parent, *d['directories'].values()):
            if output == source or output in source.parents or source in output.parents:
                raise ValueError('Inputs and output must be separate, non-nested directories')
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / ('.' + output.name + '.merge.lock')), timeout=0):
        state_path = output / '.merge-state/run.json'
        if output.exists() and any(output.iterdir()) and not state_path.is_file():
            raise ValueError('Non-empty output has no merge checkpoint; choose a new output directory')
        previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        output.mkdir(parents=True, exist_ok=True)
        config = {'inputs': {d['alias']: {'yaml': str(d['yaml']), 'sha256': d['yaml_sha256']} for d in datasets},
                  'content_hash_check': check_content_hashes, 'policy': signature([metadata, ALIASES, sorted(EXCLUDED)])}
        if previous and previous['config'] != config:
            raise ValueError('Merge inputs or policy changed; use a new output directory')
        # Never expose a stale certificate while resuming or failing.
        for name in ('data.yaml', 'MERGE_REPORT.md'):
            (output / name).unlink(missing_ok=True)
        def status(value, error=None):
            atomic_text(state_path, json.dumps({'schema': 1, 'config': config, 'status': value, 'error': error,
                                               **({'inventory': previous['inventory']} if previous and previous.get('inventory') else {})}, ensure_ascii=False, indent=2))
        status('RUNNING')
        records, excluded, input_counts = [], [], defaultdict(Counter)
        try:
            for dataset in datasets:
                for split, directory in dataset['directories'].items():
                    images = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
                    if not images:
                        raise ValueError(f'No images in {dataset["alias"]}/{split}: {directory}')
                    input_counts[dataset['alias']][split] = len(images)
                    tasks = ((dataset, split, p) for p in images)
                    with tqdm(total=len(images), desc=f'Scan labels {dataset["alias"]}/{split}', unit='img') as progress, closing(bounded_map(scan_one, tasks, workers, progress)) as results:
                        for r in results:
                            (excluded if r.get('error') else records).append(r)
            duplicates, hashes = exclusions_for_duplicates(records, check_content_hashes)
            for index, reason in duplicates.items():
                records[index]['error'] = reason
                excluded.append(records[index])
            records = [r for i, r in enumerate(records) if i not in duplicates]
            trained = {name for r in records if r['split'] == 'train' for name, _ in r['rows']}
            for r in records:
                missing = {name for name, _ in r['rows']} - trained
                if missing:
                    r['error'] = 'no training examples for unified classes: ' + ', '.join(sorted(missing))
                    excluded.append(r)
            records = [r for r in records if not r.get('error')]
            with (output / 'excluded_images.jsonl').open('w', encoding='utf-8') as stream:
                for r in excluded:
                    stream.write(json.dumps({'dataset': r['dataset'], 'split': r['split'], 'image': str(r['image']), 'reason': r['error']}, ensure_ascii=False) + '\n')
            names = sorted({name for r in records for name, _ in r['rows']})
            if not names or any(not any(r['split'] == s and r['rows'] for r in records) for s in SPLITS):
                raise ValueError('Every original split must retain positive boxes after exclusions')
            ids = {name: i for i, name in enumerate(names)}
            inventory = signature([[r['dataset'], r['split'], r['relative'], {k: v for k, v in r['image_stamp'].items() if k != 'ctime_ns'}, {k: v for k, v in r['label_stamp'].items() if k != 'ctime_ns'}, r['rows']] for r in records])
            old_inventory = previous.get('inventory') if previous else None
            if old_inventory and old_inventory != inventory:
                raise ValueError('Source files/labels or exclusions changed; use a new output directory')
            state_value = {'schema': 1, 'config': config, 'status': 'RUNNING', 'inventory': inventory}
            atomic_text(state_path, json.dumps(state_value, ensure_ascii=False, indent=2))
            print('Prepare output directories (single-pass temporary cleanup)...', flush=True)
            temporary_labels_removed = prepare_output_directories(records, output)
            reused = 0
            with tqdm(total=len(records), desc='Hardlink images and remap IDs', unit='img') as progress, closing(bounded_map(lambda r: write_one(r, output, ids), records, workers, progress)) as results:
                for hit in results:
                    reused += hit
            by_alias = {d['alias']: d for d in datasets}
            with tqdm(total=len(records), desc='Audit links and label coordinates', unit='img') as progress, closing(bounded_map(lambda r: audit_one(r, output, ids, by_alias[r['dataset']]['mapping']), records, workers, progress)) as results:
                for _ in results:
                    pass
            for kind, field in (('images', 'image_rel'), ('labels', 'label_rel')):
                expected = {str(r[field]) for r in records}
                actual = {str(p.relative_to(output)) for p in (output / kind).rglob('*') if p.is_file() and p.suffix != '.cache'}
                if actual != expected:
                    raise ValueError(f'Unexpected or missing files in output {kind}; no training YAML issued')
            report = make_report(datasets, records, excluded, ids, metadata, reused, hashes, check_content_hashes)
            report['audit']['temporary_labels_removed'] = temporary_labels_removed
            report['input_images'] = {ds: dict(counts) for ds, counts in input_counts.items()}
            if len(records) + len(excluded) != sum(sum(c.values()) for c in input_counts.values()):
                raise ValueError('Input/retained/excluded image accounting mismatch')
            atomic_text(output / 'report.json', json.dumps(report, ensure_ascii=False, indent=2))
            atomic_text(output / 'class_mapping.json', json.dumps(report['source_class_mapping'], ensure_ascii=False, indent=2))
            atomic_text(output / 'class_hierarchy.json', json.dumps({'classes': report['hierarchy']}, ensure_ascii=False, indent=2))
            atomic_text(output / 'MERGE_REPORT.md', markdown(report))
            atomic_text(output / 'data.yaml', yaml.safe_dump({'path': output.as_posix(), **{s: 'images/' + s for s in SPLITS}, 'nc': len(names), 'names': dict(enumerate(names))}, sort_keys=False, allow_unicode=True))
            state_value['status'] = 'COMPLETE'
            atomic_text(state_path, json.dumps(state_value, ensure_ascii=False, indent=2))
            (output / 'PROCESSING_REPORT.md').unlink(missing_ok=True)
            print(f'Merged {len(records):,} images; {len(names)} classes; excluded {len(excluded):,}. YAML: {output / "data.yaml"}', flush=True)
            return report
        except BaseException as exc:
            (output / 'data.yaml').unlink(missing_ok=True)
            (output / 'MERGE_REPORT.md').unlink(missing_ok=True)
            current = json.loads(state_path.read_text(encoding='utf-8'))
            current.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED', error=str(exc))
            atomic_text(state_path, json.dumps(current, ensure_ascii=False, indent=2))
            atomic_text(output / 'PROCESSING_REPORT.md', '# Merge did not complete\n\n' + str(exc) + '\n\nSources were not modified. Rerun the same command to resume unchanged inputs. No training YAML was issued.\n')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lila', type=Path, default=Path('yolo_datasets/lila_camping'), help='Processed common-name LILA directory or data.yaml')
    parser.add_argument('--animals', type=Path, default=Path('yolo_datasets/animals-detection-images-dataset'), help='Converted Animals Detection directory or data.yaml')
    parser.add_argument('--output', type=Path, default=Path('yolo_datasets/camping_merged'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-content-hashes', action='store_true', help='Also hash equal-size candidate images for exact-copy leakage/conflict detection; never decode images or persist hashes')
    args = parser.parse_args()
    try:
        merge(args.lila, args.animals, args.output, workers=args.workers, check_content_hashes=args.check_content_hashes)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Rerun the same command to resume.\n')
    except (ValueError, OSError, KeyError, TypeError, CancelledError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\nSee PROCESSING_REPORT.md and excluded_images.jsonl when available.\n')


if __name__ == '__main__':
    main()
