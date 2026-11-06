"""Download human-confirmed Open Images V7 animal images by class, without boxes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import re
import time
import urllib.request

from tqdm import tqdm

BASE = 'https://storage.googleapis.com/openimages/v7/'
DEFAULT_CLASSES = ['Wolf', 'Red wolf', 'Coyote', 'Jackal', 'Cat', 'Lion', 'Crocodile']
WOLVES = {'Wolf', 'Red wolf', 'Coyote', 'Jackal'}


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
        group = 'wolf' if name in WOLVES else name.lower().replace(' ', '_')
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
    return selected, dict(available), conflicts


def run(args):
    output = args.output.resolve()
    meta = output/'_metadata'
    meta.mkdir(parents=True, exist_ok=True)
    config = {'classes':args.classes, 'splits':args.splits, 'seed':args.seed, 'max_per_class':args.max_per_class}
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
        'downloaded_images':dict(counts), 'total_images':sum(counts.values()), 'max_per_original_class':args.max_per_class,
        'conflicting_selected_images_skipped':len(conflicts), 'failures':failures, 'seed':args.seed,
        'splits':args.splits, 'classes':args.classes}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# Open Images 图片下载统计', '', '仅筛选人工确认的正图像标签。未生成检测框。',
             'Wolf、Red wolf、Coyote、Jackal 按业务规则统一为 wolf。',
             f'每个原始类别最多 {args.max_per_class} 张，合并后的 wolf 最多 {4*args.max_per_class} 张。',
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
    p.add_argument('--classes',nargs='+',default=DEFAULT_CLASSES)
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation','test'])
    p.add_argument('--max-per-class',type=int,default=5000,help='Cap each ORIGINAL class across all requested splits')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--retries',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args()
    if args.max_per_class<1 or args.workers<1 or args.retries<0:
        p.error('Require positive max-per-class/workers and nonnegative retries')
    args.splits = list(dict.fromkeys(args.splits))
    args.classes = list(dict.fromkeys(args.classes))
    try:
        return run(args)
    except KeyboardInterrupt:
        p.exit(130,'\nInterrupted; downloaded files retained. Rerun the same command.\n')
    except (OSError,ValueError,KeyError) as exc:
        p.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
