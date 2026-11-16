"""Visualize near, middle and far bounding boxes for one YOLO class."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw
from tqdm import tqdm
import yaml

from merge_yolo_ids import SPLITS, images_for, label_for, load_dataset, validate_label
from split_yolo import dataset_file
from visualize_yolo import box_pixels, draw_legend, load_font, parse_classes


COLORS = {'far': (28, 119, 222), 'mid': (206, 110, 0), 'near': (9, 136, 68)}
SCENES = (*COLORS, 'mixed')


def parse_ratio(value):
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError('Use a fraction or decimal, e.g. 1/100 or 0.01') from exc


def read_scene_boxes(image, dataset, target_ids, far_below, near_from):
    raw, _ = validate_label(label_for(image), dataset)
    boxes = []
    for line in raw.decode('utf-8-sig').splitlines():
        fields = line.split()
        if not fields:
            continue
        cid = int(fields[0])
        area = scene = None
        if cid in target_ids:
            area = Fraction(fields[3]) * Fraction(fields[4])
            scene = 'far' if area < far_below else 'mid' if area < near_from else 'near'
        boxes.append((cid, tuple(map(float, fields[1:])), scene, area))
    return boxes


def render_scene(path, boxes, names, font):
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    legend = []
    other_classes = set()
    # Draw other classes first so a target box remains visible when they overlap.
    for cid, box, scene, area in sorted(boxes, key=lambda item: item[2] is not None):
        color = COLORS.get(scene, (112, 112, 112))
        x1, y1, x2, y2 = box_pixels(box, *image.size)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3 if scene else 2)
        if scene is None:
            other_classes.add(cid)
            continue
        caption = f'{scene} {float(area):.3%}'
        tag = str(len(legend) + 1)
        _, _, number_width, number_height = draw.textbbox((0, 0), tag, font=font, anchor='lt')
        left = max(0, min(x1, image.width - number_width - 6))
        top = max(0, min(y1 - number_height - 6, image.height - number_height - 6))
        draw.rectangle((left, top, left + number_width + 5, top + number_height + 5), fill=color)
        draw.text((left + 3, top + 3), tag, font=font, fill='white', anchor='lt')
        legend.append((color, f'{tag}  {names[cid]}  {caption}'))
    if other_classes:
        legend.append(((112, 112, 112), ', '.join(names[cid] for cid in sorted(other_classes))))
    return draw_legend(image, legend, font)


def preview_records(records, limit):
    ordered = sorted(records, key=lambda record: (record[0], str(record[1])))
    if not limit or len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered)//2]]
    return [ordered[i*(len(ordered)-1)//(limit-1)] for i in range(limit)]


def preview(source, target_class, far_below, near_from, *, output=None, split='all', limit=100):
    if not 0 < far_below < near_from <= 1 or limit < 0:
        raise ValueError('Require 0 < far-below < near-from <= 1 and limit >= 0')
    dataset = load_dataset(dataset_file(source))
    names = dataset['names']
    target_ids = parse_classes([target_class], names)
    name = names[min(target_ids)]
    splits = SPLITS if split == 'all' else (split,)
    images = {s: sorted(set(images_for(dataset, s))) for s in splits}
    if not any(images.values()):
        raise ValueError(f'No images configured for split {split!r} in {dataset["yaml"]}')
    if output is None:
        folder = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')[:60]
        thresholds = f'far-{far_below.numerator}of{far_below.denominator}_near-{near_from.numerator}of{near_from.denominator}'
        output = dataset['yaml'].parent / 'scene_preview' / f'{min(target_ids)}_{folder}' / thresholds
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')

    records = defaultdict(list)
    box_counts = Counter()
    scanned = sum(map(len, images.values()))
    issues = []
    for split_name, paths in images.items():
        for image in tqdm(paths, desc=f'Scan {split_name} labels', unit='img', disable=not paths):
            try:
                boxes = read_scene_boxes(image, dataset, target_ids, far_below, near_from)
            except (OSError, ValueError) as exc:
                issues.append(f'{image} | {exc}')
                continue
            scenes = {scene for _, _, scene, _ in boxes if scene is not None}
            if not scenes:
                continue
            group = next(iter(scenes)) if len(scenes) == 1 else 'mixed'
            box_counts.update(scene for _, _, scene, _ in boxes if scene is not None)
            largest_area = max(area for _, _, _, area in boxes if area is not None)
            records[group, split_name].append((largest_area, image, boxes))

    output.mkdir(parents=True)
    font = load_font(size=16)
    written = Counter()
    for (group, split_name), bucket in records.items():
        folder = output / group / split_name
        folder.mkdir(parents=True)
        selected = tqdm(preview_records(bucket, limit), desc=f'Preview {group}/{split_name}', unit='img', leave=False)
        for index, (area, image, boxes) in enumerate(selected, 1):
            try:
                caption = f'{100*float(area):07.3f}pct'
                target = folder / f'{index:05d}_{caption}_{image.stem[:100]}.jpg'
                render_scene(image, boxes, names, font).save(target, quality=78, optimize=True)
                written[group, split_name] += 1
            except (OSError, ValueError, SyntaxError) as exc:
                issues.append(f'{image} | {exc}')

    matched = sum(map(len, records.values()))
    lines = ['# 景别实验', '', f'类别 `{name}`，ID {", ".join(map(str, sorted(target_ids)))}。共匹配 {matched} / {scanned} 张。', '',
             f'远景 `< {far_below}`，中景 `[{far_below}, {near_from})`，近景 `>= {near_from}`。',
             '同一张图的目标类框跨景别时归到 `mixed`。灰框标记同图其他类别。', '',
             '| 景别 | 划分 | 匹配图片 | 已生成预览 |', '| --- | --- | ---: | ---: |']
    for group in SCENES:
        for split_name in splits:
            key = group, split_name
            lines.append(f'| {group} | {split_name} | {len(records.get(key, []))} | {written[key]} |')
    lines += ['', '目标类框数（含 `mixed` 图中的框）', '',
              *(f'- {group} · {box_counts[group]}' for group in COLORS), '',
              f'读取或渲染异常 {len(issues)} 张。']
    if issues:
        lines += ['', '## 异常文件', '', *(f'- {issue}' for issue in issues)]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved {sum(written.values())}/{matched} matching previews to {output}. Issues: {len(issues)}')
    return bool(issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='YOLO dataset directory or dataset YAML')
    parser.add_argument('--class', '--classes', dest='target_class', required=True, help='One class name or numeric ID')
    parser.add_argument('--far-below', type=parse_ratio, default='1/100', help='Far cutoff (exclusive), default 1/100')
    parser.add_argument('--near-from', type=parse_ratio, default='1/16', help='Near cutoff (inclusive), default 1/16')
    parser.add_argument('--split', choices=('all', *SPLITS), default='all')
    parser.add_argument('--limit', type=int, default=100, help='Previews per scene and split; 0 exports all matches')
    parser.add_argument('--output', type=Path, help='New directory for previews and REPORT.md')
    try:
        return int(preview(**vars(parser.parse_args())))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
