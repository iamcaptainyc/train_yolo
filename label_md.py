"""Generate folder-class YOLO pseudo-labels with an Ultralytics-compatible MegaDetector."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil

import yaml
from tqdm import tqdm

from merge_yolo_ids import SUFFIXES


def discover(root):
    records, seen = [], {}
    splits = {'train', 'training', 'val', 'valid', 'validation', 'test'}
    for image in sorted(root.rglob('*')):
        if image.suffix.lower() not in SUFFIXES or not image.is_file():
            continue
        rel = image.relative_to(root)
        if any(p.lower() in {'label', 'labels'} for p in rel.parts[:-1]):
            continue
        parts = rel.parts[1:] if rel.parts[0].lower() in splits else rel.parts
        if len(parts) < 2:
            raise ValueError(f'Expected a class folder for {image}')
        name = re.sub(r'[^a-z0-9]+', '_', parts[0].lower()).strip('_')
        if not name:
            raise ValueError(f'Invalid class folder: {parts[0]}')
        stat = image.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            if seen[identity] != name:
                raise ValueError(f'Same physical image has conflicting classes: {image}')
            continue
        seen[identity] = name
        records.append({'path': image, 'class': name, 'relative': rel.as_posix()})
    if not records:
        raise ValueError(f'No images in class folders under {root}')
    return records


def animal_class(model, override=None):
    names = model.names
    names = dict(enumerate(names)) if isinstance(names, list) else names
    if override is not None:
        if override not in names:
            raise ValueError(f'Animal class ID {override} absent from model names {names}')
        return override
    matches = [int(i) for i, n in names.items() if str(n).strip().lower() in {'animal', 'animals'}]
    if len(matches) != 1:
        raise ValueError(f'Cannot identify animal class in {names}; specify --animal-class-id')
    return matches[0]


def boxes_from_result(result, category, threshold):
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        if int(box.cls.item()) != category or float(box.conf.item()) < threshold:
            continue
        x, y, w, h = map(float, box.xywhn[0].tolist())
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError('Non-finite predicted box')
        left, top = max(0., x-w/2), max(0., y-h/2)
        right, bottom = min(1., x+w/2), min(1., y+h/2)
        if right > left and bottom > top:
            rows.append(((left+right)/2, (top+bottom)/2, right-left, bottom-top))
    return rows


def generate(args, model):
    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir() or not args.weights.is_file():
        raise ValueError('Input directory and local weights file must exist')
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Use a new output directory outside the input tree')
    if not 0 < args.conf < 1 or args.batch < 1 or args.imgsz < 1:
        raise ValueError('Require 0 < conf < 1 and positive batch/imgsz')
    category = animal_class(model, args.animal_class_id)
    records = discover(root)
    output.mkdir(parents=True)
    incomplete = output/'INCOMPLETE.txt'
    incomplete.write_text('Pseudo-label generation unfinished. Do not train.\n')
    kept, skipped = [], Counter()
    with (output/'issues.jsonl').open('w', encoding='utf-8') as issues:
        for offset in tqdm(range(0, len(records), args.batch), desc='MDV6 pseudo-label', unit='batch'):
            batch = records[offset:offset+args.batch]
            try:
                results = list(model.predict(source=[str(r['path']) for r in batch],
                    imgsz=args.imgsz, conf=args.conf, classes=[category], device=args.device,
                    batch=args.batch, verbose=False, save=False))
                if len(results) != len(batch):
                    raise ValueError('Predictor returned an unexpected result count')
                for record, result in zip(batch, results, strict=True):
                    if Path(result.path).resolve() != record['path'].resolve():
                        raise ValueError('Predictor image/result order mismatch')
                    rows = boxes_from_result(result, category, args.conf)
                    if rows:
                        kept.append({**record, 'boxes': rows})
                    else:
                        skipped[record['class']] += 1
                        issues.write(json.dumps({'image':record['relative'], 'reason':'no_animal_detection'})+'\n')
            except Exception as exc:
                # An inference failure must not silently remove an entire batch.
                issues.write(json.dumps({'batch':[r['relative'] for r in batch], 'error':str(exc)})+'\n')
                raise
    if not kept:
        raise ValueError('No animal detections survived the confidence threshold')
    names = sorted({r['class'] for r in kept})
    ids = {n:i for i,n in enumerate(names)}
    groups = defaultdict(list)
    for record in kept:
        groups[record['class']].append(record)
    counts, boxes = defaultdict(Counter), Counter()
    rng = random.Random(args.seed)
    with (output/'selection.jsonl').open('w', encoding='utf-8') as manifest:
        for name in names:
            group = groups[name]
            rng.shuffle(group)
            # Largest-remainder allocation gives approximate 8:1:1 per class.
            sizes = [len(group)*ratio for ratio in (.8, .1, .1)]
            allocated = [int(n) for n in sizes]
            order = sorted(range(3), key=lambda i: sizes[i]-allocated[i], reverse=True)
            for i in order[:len(group)-sum(allocated)]:
                allocated[i] += 1
            offset = 0
            for split, size in zip(('train','val','test'), allocated, strict=True):
                (output/'images'/split).mkdir(parents=True, exist_ok=True)
                (output/'labels'/split).mkdir(parents=True, exist_ok=True)
                for record in group[offset:offset+size]:
                    token = hashlib.sha256(record['relative'].encode()).hexdigest()
                    target = output/'images'/split/(token+record['path'].suffix.lower())
                    if args.copy_images:
                        shutil.copy2(record['path'], target)
                    else:
                        os.link(record['path'], target)
                    label = output/'labels'/split/(token+'.txt')
                    label.write_text(''.join(str(ids[name])+' '+' '.join(f'{v:.10f}' for v in box)+'\n'
                                             for box in record['boxes']), encoding='utf-8')
                    counts[name][split] += 1
                    boxes[name] += len(record['boxes'])
                    manifest.write(json.dumps({'source':str(record['path']), 'image':str(target.relative_to(output)),
                                               'class':name, 'split':split})+'\n')
                offset += size
    (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output), 'train':'images/train',
        'val':'images/val', 'test':'images/test', 'names':dict(enumerate(names))}, sort_keys=False), encoding='utf-8')
    report = {'input':str(root), 'weights':str(args.weights.resolve()), 'conf':args.conf,
        'seed':args.seed, 'input_images':len(records), 'output_images':len(kept), 'classes':len(names),
        'skipped_no_detection':dict(skipped), 'class_images':dict(counts), 'class_boxes':dict(boxes),
        'split_images':{s:sum(c[s] for c in counts.values()) for s in ('train','val','test')}}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# MDV6 自动标注报告', '', f'输入数据集 {root}', f'权重 {args.weights.resolve()}',
        f'输出数据集 {output}', '', f'类别数 {len(names)}，保留图片 {len(kept)}，框数 {sum(boxes.values())}，'
        f'无检出跳过 {sum(skipped.values())}。', '', '原划分合并后按类别重新随机分配 8:1:1，小类别存在整数误差。',
        '动物类别取自文件夹，框来自模型预测。同图所有动物均赋为文件夹类别，混合物种图片需要人工复核。',
        '同 inode 图片去重，未做内容或相邻帧去重。以下数据为伪标注，模型漏检和误检会进入结果。', '',
        '| ID | 类别 | train | val | test | 图片合计 | 框 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        c = counts[name]
        lines.append(f'| {ids[name]} | {name} | {c["train"]} | {c["val"]} | {c["test"]} | {sum(c.values())} | {boxes[name]} |')
    c = report['split_images']
    lines.append(f'| | 合计 | {c["train"]} | {c["val"]} | {c["test"]} | {len(kept)} | {sum(boxes.values())} |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    incomplete.unlink()
    print(f'Weights: {args.weights.resolve()}\nDataset: {output}\nYOLO config: {output/"data.yaml"}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('/root/autodl-tmp/capt/code/train_yolo/datasets/farm_harmful_animals'))
    p.add_argument('--weights', type=Path, required=True, help='Local MDV6 Ultralytics-compatible .pt checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--animal-class-id', type=int, help='Override animal class ID if checkpoint names are numeric')
    p.add_argument('--conf', type=float, default=.3)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--copy-images', action='store_true', help='Copy instead of default hardlinks for different filesystems')
    args = p.parse_args()
    try:
        if not args.weights.is_file():
            raise ValueError(f'Weights file not found: {args.weights}')
        from ultralytics import YOLO
        generate(args, YOLO(str(args.weights)))
    except (ValueError, OSError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
