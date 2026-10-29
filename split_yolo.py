"""Batch reshuffle YOLO datasets into new train/val/test layouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import re
import shutil

import yaml
from tqdm import tqdm

from merge_yolo_ids import SUFFIXES, SPLITS, label_for, load_dataset, validate_label


def dataset_file(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    for name in ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml'):
        if (path/name).is_file():
            return path/name
    if path.is_dir() and ((path/'images').is_dir() or any(
            (path/split/'images').is_dir() for split in ('train','val','valid','validation','test'))):
        # Internal metadata target only; never written without real class names.
        return path/'data.yaml'
    raise ValueError(f'No dataset YAML or recognizable images directory: {path}')


def load_split_source(path):
    """A split-only dataset may have numeric labels without a class-name table."""
    path = Path(path).resolve()
    if path.is_file():
        source = load_dataset(path)
        source['has_names'] = True
        return source
    root = path.parent
    if dataset_file(root) != path:
        raise ValueError(f'Dataset metadata not found: {path}')
    return {'yaml':path, 'config':{}, 'root':root, 'names':{}, 'reserved':set(), 'has_names':False}


def collect_images(source):
    """Pool every local images tree, including old train/images layouts.

    YAML split paths add support for external directories and txt lists. Missing
    split paths are recorded rather than preventing recovery of local images.
    """
    roots = list(dict.fromkeys((source['yaml'].parent, source['root'])))
    candidates, ignored = [], []

    def add_path(path):
        if path.is_dir():
            candidates.extend(p.absolute() for p in sorted(path.rglob('*'))
                              if p.is_file() and p.suffix.lower() in SUFFIXES)
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip()).expanduser()
                if not image.is_absolute():
                    image = path.parent/image
                if not image.is_file():
                    raise ValueError(f'Missing image referenced by {path}: {image}')
                if image.suffix.lower() not in SUFFIXES:
                    raise ValueError(f'Unsupported image in {path}: {image}')
                candidates.append(image.absolute())
        else:
            ignored.append(str(path))

    for root in roots:
        if (root/'images').is_dir():
            add_path(root/'images')
        for name in ('train', 'val', 'valid', 'validation', 'test'):
            if (root/name/'images').is_dir():
                add_path(root/name/'images')
    for split in SPLITS:
        entries = source['config'].get(split)
        for raw in entries if isinstance(entries, list) else ([entries] if entries else []):
            if not isinstance(raw, str):
                raise ValueError(f'Invalid {split} entry in {source["yaml"]}')
            path = Path(raw).expanduser()
            if path.is_absolute():
                add_path(path)
            else:
                matches = [root/path for root in roots if (root/path).exists()]
                if matches:
                    for match in matches:
                        add_path(match)
                else:
                    ignored.append(str(source['root']/path))
    # The same path may be discovered both recursively and through old split lists.
    unique = {}
    for path in candidates:
        unique.setdefault(str(path.resolve()), path)
    images = sorted(unique.values(), key=lambda p: str(p))
    if not images:
        raise ValueError(f'No images found for {source["yaml"]}')
    return images, sorted(set(ignored))


def bounded_map(executor, function, values, workers):
    iterator = iter(values)
    while batch := list(islice(iterator, max(1, workers)*32)):
        yield from executor.map(function, batch)


def target_counts(total, ratios):
    if len(ratios) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratios) or not math.isfinite(sum(ratios)):
        raise ValueError('Provide three finite positive ratios, e.g. 8 1 1')
    if total < 3:
        raise ValueError('At least 3 images are required for non-empty train/val/test')
    weights = [v/sum(ratios) for v in ratios]
    exact = [total*v for v in weights]
    counts = [math.floor(v) for v in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i]-counts[i]), i))
    for i in order[:total-sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[i] = 1
    return dict(zip(SPLITS, counts))


def assign_splits(records, ratios, seed):
    targets = target_counts(len(records), ratios)
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record['identity']].append(i)
    if len(groups) < 3:
        raise ValueError('Need at least 3 independent image/file groups for train/val/test')
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    counts = Counter()
    for position, indices in enumerate(shuffled):
        empty = [split for split in SPLITS if not counts[split]]
        choices = empty if len(shuffled)-position == len(empty) else SPLITS
        split = max(choices, key=lambda s: targets[s]-counts[s])
        counts[split] += len(indices)
        for i in indices:
            records[i]['split'] = split
    return targets, len(groups)


def split_dataset(source, output, ratios=(8,1,1), seed=42, workers=16, copy_images=False, collected=None):
    images, ignored = collected if collected is not None else collect_images(source)
    has_names = source.get('has_names',True)

    def inspect(image):
        stamp = image.stat()
        if stamp.st_size == 0:
            raise ValueError(f'Empty image: {image}')
        label = label_for(image)
        raw, counts = validate_label(label, source, check_class_ids=has_names)
        identity = (stamp.st_dev, stamp.st_ino) if stamp.st_ino else ('path', str(image.resolve()))
        return {'image':image, 'label':label, 'raw':raw, 'counts':counts, 'identity':identity}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(bounded_map(pool, inspect, images, workers), total=len(images),
                            desc=f'Check {source["yaml"].parent.name}', unit='img'))
    targets, group_count = assign_splits(records, ratios, seed)
    output.mkdir(parents=True, exist_ok=False)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Split incomplete. Do not use until the split report is generated.\n', encoding='utf-8')
    for split in SPLITS:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)

    def write(record):
        image, split = record['image'], record['split']
        token = hashlib.sha256(str(image).encode('utf-8')).hexdigest()
        target = output/'images'/split/(token+image.suffix.lower())
        if copy_images:
            shutil.copy2(image, target)
        else:
            try:
                os.link(image, target)
            except OSError as exc:
                raise ValueError(f'Hardlink failed for {image}; use --copy-images across filesystems') from exc
        (output/'labels'/split/(token+'.txt')).write_bytes(record['raw'])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(bounded_map(pool, write, records, workers), total=len(records),
                      desc='Copy' if copy_images else 'Hardlink', unit='img'):
            pass
    stats = {split: {'images':0, 'negative_images':0} for split in SPLITS}
    class_images, boxes = {s:Counter() for s in SPLITS}, {s:Counter() for s in SPLITS}
    for record in records:
        split, counts = record['split'], record['counts']
        stats[split]['images'] += 1
        stats[split]['negative_images'] += not counts
        class_images[split].update(counts.keys())
        boxes[split].update(counts)
    observed_ids = sorted(set().union(*(set(boxes[s]) for s in SPLITS)))
    report = {'source':str(source['yaml'] if has_names else source['root']), 'output':str(output), 'source_splits_preserved':False,
              'seed':seed, 'ratios':list(ratios), 'target_images':targets, 'total_images':len(records),
              'independent_file_groups':group_count, 'splits':stats, 'names':source['names'],
              'class_names_available':has_names, 'observed_class_ids':observed_ids, 'data_yaml_generated':has_names,
              'class_image_counts':class_images, 'class_box_counts':boxes,
              'image_mode':'copy' if copy_images else 'hardlink', 'ignored_missing_split_paths':ignored,
              'class_ids_preserved':True, 'label_bytes_preserved':True,
              'grouping':'same physical file/hardlinks stay in one split; all distinct image paths retained',
              'limitation':'random image/file-group split, not class stratification or camera/sequence grouping; no content hash or decode'}
    config = {'path':output.as_posix(), 'train':'images/train', 'val':'images/val', 'test':'images/test',
              'nc':len(source['names']), 'names':source['names']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    class_note = f'{len(source["names"])} 个类别定义' if has_names else f'{len(observed_ids)} 个实际出现的类别 ID（未提供名称）'
    lines = ['# YOLO 数据集重新划分报告', '', f'来源：`{report["source"]}`', '',
             f'共 {len(records):,} 张，{class_note}；原划分已全部打散。', '',
             f'随机种子：{seed}；比例：{list(ratios)}；图片：{report["image_mode"]}。类别 ID 和标签原文保留。', '',
             '按图片随机分配；同一文件/硬链接整体分配，实际比例可能略有偏差。未按物种、相机或序列分层，不识别独立复制的重复图。', '',
             '| 划分 | 目标图片数 | 实际图片数 | 负样本 |', '| --- | ---: | ---: | ---: |']
    lines += [f'| {s} | {targets[s]} | {stats[s]["images"]} | {stats[s]["negative_images"]} |' for s in SPLITS]
    lines += ['', '## 各类别统计', '', '| ID | 类别 | train 图片 | val 图片 | test 图片 | 框总数 |',
              '| ---: | --- | ---: | ---: | ---: | ---: |']
    for cid in sorted(source['names']) if has_names else observed_ids:
        name = source['names'].get(cid,'—（未提供名称）')
        name = name.replace('|','\\|').replace('\n',' ')
        lines.append(f'| {cid} | {name} | {class_images["train"][cid]} | {class_images["val"][cid]} | {class_images["test"][cid]} | {sum(boxes[s][cid] for s in SPLITS)} |')
    if ignored:
        lines += ['', '## 忽略的失效划分路径', '', '已从本地 images 目录及有效 YAML 路径收集图片；以下路径不存在：', '']
        lines += [f'- `{path}`' for path in ignored]
    if not has_names:
        lines += ['', '## 未提供类别表', '',
                  '未找到数据集 YAML。已保留标签中的全部原始 ID、框与负样本，并按 ID 统计；不猜测名称、不生成 data.yaml。',
                  '直接训练或按名称合并前，需要补充真实的 ID→类别名称表。']
    (output/'SPLIT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if has_names:
        (output/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    marker.unlink()
    return report


def run_batch(datasets, output, ratios=(8,1,1), seed=42, workers=16, copy_images=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    target_counts(10, ratios)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')
    paths = [dataset_file(value) for value in datasets]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError('Provide distinct dataset YAMLs')
    # Protect every input before creating any output, including later batch members.
    prepared, errors = {}, {}
    for path in paths:
        try:
            prepared[path] = load_split_source(path)
        except (ValueError,OSError,UnicodeError,yaml.YAMLError) as exc:
            errors[path] = str(exc)
        roots = (path.parent, prepared[path]['root']) if path in prepared else (path.parent,)
        for root in roots:
            if output == root or output in root.parents or root in output.parents:
                raise ValueError(f'Output must be separate from every input: {root}')
    output.mkdir(parents=True)
    (output/'INCOMPLETE.txt').write_text('Batch in progress. Read BATCH_REPORT.md for each dataset status.\n',encoding='utf-8')
    results = []

    def save_batch():
        (output/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        lines = ['# YOLO 批量重新划分报告', '', '所有数据集分别重新打乱，不合并类别，不保留原划分。', '',
                 '| 来源 | 状态 | 图片数 | train / val / test | 输出 / 错误 |', '| --- | --- | ---: | --- | --- |']
        for result in results:
            if result['status'] == 'ok':
                r = result['report']
                counts = ' / '.join(str(r['splits'][s]['images']) for s in SPLITS)
                detail = f'[{Path(r["output"]).name}]({Path(r["output"]).name}/SPLIT_REPORT.md)'
                lines.append(f'| {result["source"].replace("|","_")} | 完成 | {r["total_images"]} | {counts} | {detail} |')
            else:
                error = result['error'].replace('|','\\|').replace('\n',' ')
                lines.append(f'| {result["source"].replace("|","_")} | 失败 | — | — | {error} |')
        (output/'BATCH_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    try:
        for path in paths:
            slug = re.sub(r'[^\w.-]+','_',path.parent.name).strip('.') or 'dataset'
            name = slug+'-'+hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:8]
            try:
                if path in errors:
                    raise ValueError(errors[path])
                source = prepared[path]
                report = split_dataset(source,output/name,ratios,seed,workers,copy_images)
                results.append({'source':report['source'],'status':'ok','report':report})
            except (ValueError,OSError,UnicodeError,yaml.YAMLError) as exc:
                results.append({'source':str(path),'status':'failed','error':str(exc)})
                print(f'Failed: {path}: {exc}',flush=True)
            save_batch()
    finally:
        save_batch()
    if all(r['status']=='ok' for r in results) and len(results)==len(paths):
        (output/'INCOMPLETE.txt').unlink()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs='+',required=True,help='Dataset roots or YAML files')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output',type=Path,help='New batch output directory')
    destination.add_argument('--in-place',action='store_true',help='Resplit directly in each original directory with rollback')
    parser.add_argument('--ratios',type=float,nargs=3,default=[8,1,1],metavar=('TRAIN','VAL','TEST'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--copy-images',action='store_true',help='Copy instead of hardlinking across filesystems')
    args = parser.parse_args()
    if args.in_place and args.copy_images:
        parser.error('--in-place uses temporary hardlinks; do not combine with --copy-images')
    try:
        if args.in_place:
            from split_yolo_inplace import run_in_place_batch
            results = run_in_place_batch(args.datasets,args.ratios,args.seed,args.workers)
        else:
            results = run_batch(args.datasets,args.output,args.ratios,args.seed,args.workers,args.copy_images)
        failed = sum(r['status']!='ok' for r in results)
        report_note = 'SPLIT_REPORT.md in each original dataset' if args.in_place else str(args.output / 'BATCH_REPORT.md')
        print(f'Finished: {len(results)-failed} OK, {failed} failed. Report: {report_note}')
        if failed:
            parser.exit(1)
    except KeyboardInterrupt:
        message = 'Rerun the same command; interrupted in-place transactions recover first.' if args.in_place else 'Source data unchanged; use a new output directory to rerun.'
        parser.exit(130,f'\nStopped. {message}\n')
    except ImportError as exc:
        parser.exit(1,f'Missing dependency: {exc}; install requirements.txt\n')
    except (ValueError,OSError,UnicodeError,yaml.YAMLError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    main()
