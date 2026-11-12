"""Download Open Images V7 images or add official detection boxes to existing images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
import urllib.request

from tqdm import tqdm

BASE = 'https://storage.googleapis.com/openimages/v7/'
DEFAULT_CLASSES = ['Wolf', 'Red wolf', 'Coyote', 'Jackal', 'Cat', 'Lion', 'Crocodile']
WOLVES = {'Wolf', 'Red wolf', 'Coyote', 'Jackal'}
BOX_CLASSES = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'
BOX_URLS = {
    'train': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
    'validation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
    'test': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
}
YOLO_SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}


def output_class(name):
    return 'wolf' if name in WOLVES else name.lower().replace(' ', '_')


def download(url, path, retries=5, progress=False):
    """Publish only complete files; retain a resumable .part after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return
    partial = path.with_name(path.name+'.part')
    for attempt in range(retries+1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                    if not match or int(match[1]) != offset:
                        raise ValueError('Server returned an invalid resume range')
                    total = int(match[3])
                else:
                    offset = 0
                    total = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
                with partial.open('ab' if offset else 'wb') as stream, tqdm(
                        total=total, initial=offset, unit='B', unit_scale=True,
                        desc=path.name, disable=not progress) as bar:
                    while chunk := response.read(1024*1024):
                        stream.write(chunk)
                        bar.update(len(chunk))
                if total is not None and partial.stat().st_size != total:
                    raise OSError('Incomplete HTTP download')
            if not partial.stat().st_size:
                raise OSError('Empty download')
            partial.replace(path)
            return
        except (OSError, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 15))


def select_images(files, class_ids, maximum, seed):
    # Reservoir sampling keeps memory bounded and scans the original labels once.
    rng = random.Random(seed)
    pools, available = defaultdict(list), Counter()
    for split, path in files:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            for row in tqdm(csv.DictReader(stream), desc=f'Scan {split} image labels', unit='rows'):
                name = class_ids.get(row['LabelName'])
                if name is None or float(row['Confidence']) != 1:
                    continue
                image_id = row['ImageID']
                if not re.fullmatch(r'[0-9a-fA-F]{16}', image_id):
                    raise ValueError(f'Invalid Open Images image ID: {image_id}')
                available[name] += 1
                item = (split, image_id)
                if len(pools[name]) < maximum:
                    pools[name].append(item)
                else:
                    index = rng.randrange(available[name])
                    if index < maximum:
                        pools[name][index] = item
    images = {}
    for name, pool in pools.items():
        group = output_class(name)
        for split, image_id in pool:
            record = images.setdefault((split,image_id), {'split':split, 'id':image_id,
                                                         'groups':set(), 'original_classes':set()})
            record['groups'].add(group)
            record['original_classes'].add(name)
    # Different target species in one image cannot be safely folder-labeled.
    selected, conflicts = [], []
    for key in sorted(images):
        record = images[key]
        record['groups'] = sorted(record['groups'])
        record['original_classes'] = sorted(record['original_classes'])
        if len(record['groups']) > 1:
            conflicts.append(record)
        else:
            record['group'] = record.pop('groups')[0]
            selected.append(record)
    # Balance fine classes within each output class; cap unique images after merging.
    groups = defaultdict(lambda: defaultdict(list))
    for record in selected:
        for name in record['original_classes']:
            groups[record['group']][name].append(record)
    selected = []
    for group in sorted(groups):
        buckets = groups[group]
        for name in sorted(buckets):
            rng.shuffle(buckets[name])
        used = set()
        while len(used) < maximum:
            added = False
            for name in sorted(buckets):
                bucket = buckets[name]
                while bucket and (bucket[-1]['split'], bucket[-1]['id']) in used:
                    bucket.pop()
                if bucket:
                    record = bucket.pop()
                    used.add((record['split'], record['id']))
                    selected.append(record)
                    added = True
                    if len(used) == maximum:
                        break
            if not added:
                break
    return selected, dict(available), conflicts


def match_boxes(path, image_ids, groups_by_mid):
    matched = defaultdict(set)
    coordinates = ('XMin', 'YMin', 'XMax', 'YMax')
    required = {'ImageID', 'LabelName', *coordinates}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Expected an official bounding-box CSV, with coordinates: {path}')
        for row in tqdm(reader, desc=f'Scan {path.name}', unit='rows'):
            image_id = row['ImageID']
            if image_id not in image_ids:
                continue
            group = groups_by_mid.get(row['LabelName'])
            if group is None or row.get('IsGroupOf') == '1' or row.get('IsDepiction') == '1':
                continue
            x1, y1, x2, y2 = (float(row[key]) for key in coordinates)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f'Invalid box coordinates: {path}:{reader.line_num}')
            matched[image_id].add((group, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))
    return matched


def export_annotations(args):
    root = args.output.resolve()
    destination = (args.yolo_output or root/'yolo_bbox').resolve()
    if destination.exists() or destination in root.parents:
        raise ValueError(f'Use a new YOLO output directory: {destination}')
    meta = root/'_metadata'
    selection = meta/'download_config.json'
    original_classes = json.loads(selection.read_text(encoding='utf-8'))['classes'] if selection.is_file() else None
    requested = args.classes or original_classes or DEFAULT_CLASSES
    groups = {output_class(name) for name in requested}
    images = defaultdict(dict)
    for group in sorted(groups):
        for image in sorted((root/group).glob('*.jpg')):
            match = re.fullmatch(r'(train|validation|test)_([0-9a-f]{16})', image.stem, re.IGNORECASE)
            if match and match[1].lower() in args.splits and image.is_file() and image.stat().st_size:
                split, image_id = (part.lower() for part in match.groups())
                images[split].setdefault(image_id, []).append((group, image))
    if not images:
        raise ValueError(f'No previously downloaded images found under {root}; check --output and --classes')

    classes_path = meta/'boxable_classes.csv'
    download(BOX_CLASSES, classes_path, args.retries, True)
    with classes_path.open(newline='', encoding='utf-8-sig') as stream:
        boxable = {name: mid for mid, name in csv.reader(stream)}
    groups_by_mid = {boxable[name]: output_class(name) for name in requested if name in boxable}
    unboxable = sorted(set(requested)-boxable.keys())
    covered_groups = set(groups_by_mid.values())
    counts, exported, skipped = Counter(), Counter(), Counter()
    splits, rows = Counter(), []
    used_groups = set()
    for split, candidates in images.items():
        image_ids = {image_id for image_id, paths in candidates.items()
                     if any(group in covered_groups for group, _ in paths)}
        matched = {}
        if image_ids:
            path = meta/f'{split}_boxes.csv'
            download(BOX_URLS[split], path, args.retries, True)
            matched = match_boxes(path, image_ids, groups_by_mid)
        for image_id, paths in candidates.items():
            boxes = matched.get(image_id, ())
            box_groups = {box[0] for box in boxes}
            kept = None
            for group, image in paths:
                counts[group] += 1
                if group not in box_groups:
                    skipped[group] += 1
                elif kept is None:
                    kept = image
                    exported[group] += 1
                else:
                    skipped[group] += 1  # Same official split and image ID.
            if kept is not None:
                yolo_split = YOLO_SPLITS[split]
                splits[yolo_split] += 1
                used_groups.update(box_groups)
                rows.append((yolo_split, kept, sorted(boxes)))

    ids = {name: cid for cid, name in enumerate(sorted(used_groups))}
    box_counts = {split: Counter() for split in YOLO_SPLITS.values()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.openimages-bbox-', dir=destination.parent) as temp:
        staging = Path(temp)/'dataset'
        for split in splits:
            (staging/'images'/split).mkdir(parents=True)
            (staging/'labels'/split).mkdir(parents=True)
        for split, image, boxes in tqdm(rows, desc='Export official YOLO boxes', unit='img'):
            try:
                os.link(image, staging/'images'/split/image.name)
            except OSError as exc:
                raise OSError('Could not hardlink images; keep --yolo-output on the same filesystem') from exc
            labels = ''.join(f'{ids[group]} ' + ' '.join(f'{value:.10f}' for value in box) + '\n'
                             for group, *box in boxes)
            (staging/'labels'/split/image.with_suffix('.txt').name).write_text(labels, encoding='utf-8')
            box_counts[split].update(box[0] for box in boxes)
        staging.mkdir(exist_ok=True)
        if rows:
            # JSON is valid YAML and avoids adding a dependency to the downloader.
            config = {'path': destination.as_posix(), **{split: f'images/{split}' for split in splits},
                      'nc': len(ids), 'names': list(ids)}
            (staging/'data.yaml').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
        report = {'source': str(root), 'classes_without_official_boxes': unboxable,
                  'images_found_by_folder': dict(counts), 'images_exported_by_folder': dict(exported),
                  'images_skipped_by_folder': dict(skipped), 'images_by_split': dict(splits),
                  'boxes_by_split_and_class': box_counts, 'class_ids': ids}
        (staging/'bbox_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        staging.rename(destination)
    for group, count in sorted(counts.items()):
        print(f'{group}: exported {exported[group]}/{count}, skipped {skipped[group]}')
    if unboxable:
        print(f'No official detection class: {", ".join(unboxable)}')
    if not (splits['train'] and splits['val']):
        print('Source splits were preserved. A nonempty train and val are needed for training.')
    print(f'YOLO output: {destination}\nReport: {destination / "bbox_report.json"}')
    return 0


def run(args):
    output = args.output.resolve()
    meta = output/'_metadata'
    meta.mkdir(parents=True, exist_ok=True)
    config = {'classes':args.classes, 'splits':args.splits, 'seed':args.seed,
              'max_per_class':args.max_per_class, 'cap_scope':'merged_output_class'}
    config_path = meta/'download_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Selection settings changed; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    classes_path = meta/'classes.csv'
    download(BASE+'oidv7-class-descriptions.csv', classes_path, args.retries, True)
    with classes_path.open(newline='', encoding='utf-8-sig') as stream:
        names = {name:mid for mid,name in csv.reader(stream)}
    unknown = set(args.classes)-names.keys()
    if unknown:
        raise ValueError(f'Unknown image-level classes: {sorted(unknown)}')
    class_ids = {names[name]:name for name in args.classes}
    files = []
    for split in args.splits:
        token = 'val' if split == 'validation' else split
        path = meta/f'{split}_human_labels.csv'
        download(BASE+f'oidv7-{token}-annotations-human-imagelabels.csv', path, args.retries, True)
        files.append((split,path))
    records, available, conflicts = select_images(files, class_ids, args.max_per_class, args.seed)
    (meta/'selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    (meta/'conflicts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in conflicts), encoding='utf-8')
    counts, original, failures = Counter(), Counter(), []
    def fetch(record):
        target = output/record['group']/(record['split']+'_'+record['id']+'.jpg')
        url = f'https://open-images-dataset.s3.amazonaws.com/{record["split"]}/{record["id"]}.jpg'
        try:
            download(url, target, args.retries)
            return record, None
        except (OSError, ValueError) as exc:
            return record, str(exc)
    # Process bounded batches so interrupt does not queue thousands of downloads.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(records), desc='Download images', unit='img') as bar:
        for offset in range(0,len(records),2*args.workers):
            for record, error in pool.map(fetch, records[offset:offset+2*args.workers]):
                bar.update()
                if error:
                    failures.append({**record, 'error':error})
                else:
                    counts[record['group']] += 1
                    original.update(record['original_classes'])
    report = {'available_positive_labels':available, 'downloaded_original_class_memberships':dict(original),
        'downloaded_images':dict(counts), 'total_images':sum(counts.values()), 'max_per_output_class':args.max_per_class,
        'conflicting_selected_images_skipped':len(conflicts), 'failures':failures, 'seed':args.seed,
        'splits':args.splits, 'classes':args.classes}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# Open Images 图片下载统计', '', '仅筛选人工确认的正图像标签。未生成检测框。',
             'Wolf、Red wolf、Coyote、Jackal 按业务规则统一为 wolf。',
             f'每个输出类别最多 {args.max_per_class} 张，wolf 的四个来源合计也最多 {args.max_per_class} 张。',
             'wolf 内部按原始类别轮流抽取，尽量均衡；不足的来源由其他来源补齐。',
             '同一图片的多个 wolf 原始标签去重；已选择图片中不同输出类别冲突的图片跳过。',
             '同图仍可能存在其他未筛选动物，后续按文件夹生成伪标注时需要复核。',
             '', '| 输出类别 | 已下载图片 |', '| --- | ---: |']
    lines.extend(f'| {name} | {count} |' for name,count in sorted(counts.items()))
    lines.append(f'| 合计 | {sum(counts.values())} |')
    lines.extend(['', f'失败 {len(failures)} 张，重复执行同一命令可重试，完整文件复用，部分下载尝试续传。',
                  '元数据保存在 _metadata，训练标注脚本会跳过其中的非图片文件。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Images: {output}\nTotal downloaded: {sum(counts.values())}; failed: {len(failures)}')
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classes',nargs='+')
    p.add_argument('--annotations-only', action='store_true',
                   help='Download official boxes for existing local images and export YOLO')
    p.add_argument('--yolo-output', type=Path, help='New export directory; defaults to OUTPUT/yolo_bbox')
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation','test'])
    p.add_argument('--max-per-class',type=int,default=5000,help='Cap each MERGED OUTPUT class across all requested splits')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--retries',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args()
    if args.max_per_class<1 or args.workers<1 or args.retries<0:
        p.error('Require positive max-per-class/workers and nonnegative retries')
    args.splits = list(dict.fromkeys(args.splits))
    if args.yolo_output and not args.annotations_only:
        p.error('--yolo-output requires --annotations-only')
    if args.classes:
        args.classes = list(dict.fromkeys(args.classes))
    try:
        if args.annotations_only:
            return export_annotations(args)
        args.classes = args.classes or DEFAULT_CLASSES
        return run(args)
    except KeyboardInterrupt:
        p.exit(130,'\nInterrupted; downloaded files retained. Rerun the same command.\n')
    except (OSError,ValueError,KeyError) as exc:
        p.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
