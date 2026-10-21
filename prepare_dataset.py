"""Convert Kaggle Animals Detection (Open Images text boxes) to YOLO detection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

from PIL import Image, ImageDraw
import yaml
from tqdm import tqdm

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_annotation(text: str, width: int, height: int):
    """Input: class name (possibly spaces), xmin ymin xmax ymax in pixels."""
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().rsplit(maxsplit=4)
        if len(fields) != 5:
            raise ValueError(f'line {number}: expected class xmin ymin xmax ymax')
        name = ' '.join(fields[0].split()).casefold()
        if not name or name.replace('.', '', 1).isdigit():
            raise ValueError(f'line {number}: expected a class name, not a YOLO class id')
        try:
            x1, y1, x2, y2 = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f'line {number}: non-numeric box coordinates') from exc
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(f'line {number}: non-finite coordinates')
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'line {number}: invalid/out-of-image pixel box {(x1, y1, x2, y2)} for {width}x{height}')
        boxes.append((name, (x1+x2)/(2*width), (y1+y2)/(2*height),
                      (x2-x1)/width, (y2-y1)/height))
    if not boxes:
        raise ValueError('empty annotation; missing objects must not silently become background')
    return boxes


def locate_root(source: Path) -> Path:
    candidates = sorted({p.parent for p in source.rglob('*')
                         if p.is_dir() and p.name.casefold() == 'train'})
    if (source / 'train').is_dir():
        candidates = [source]
    candidates = [p for p in candidates if (p / 'test').is_dir()]
    if len(candidates) != 1:
        raise ValueError('Expected one dataset directory containing train/ and test/. '
                         'Pass that directory with --source (lowercase split names).')
    return candidates[0]


def collect(source: Path, skip_invalid: bool = False, show_progress=False):
    records, issues = {}, []
    label_directories = {}
    for split in ('train', 'val', 'test'):
        directory = source / split
        if not directory.exists():
            continue
        images = sorted(p for p in directory.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'No images found in {directory}')
        for image in tqdm(images, desc=f'Check {split}', unit='img', disable=not show_progress):
            # The original archive stores class/image.jpg and class/Label/image.txt.
            if image.parent not in label_directories:
                labels = [p for p in image.parent.iterdir()
                          if p.is_dir() and p.name.casefold() == 'label']
                label_directories[image.parent] = labels[0] if len(labels) == 1 else None
            label_dir = label_directories[image.parent]
            label = label_dir / (image.stem + '.txt') if label_dir else None
            try:
                if label is None or not label.is_file():
                    raise ValueError('missing unique Label/<image-stem>.txt')
                with Image.open(image) as im:
                    im.load()
                    if im.getexif().get(274, 1) != 1:
                        raise ValueError('EXIF orientation is not 1; normalize image and boxes before conversion')
                    boxes = parse_annotation(label.read_text(encoding='utf-8-sig'), *im.size)
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
            except (OSError, ValueError) as exc:
                if not skip_invalid:
                    raise ValueError(f'{image}: {exc}') from exc
                issues.append({'image': str(image), 'error': str(exc)})
                continue
            if digest in records:
                old = records[digest]
                if old['split'] != split:
                    raise ValueError(f'Duplicate image across original splits: {old["image"]} and {image}. '
                                     'Resolve the overlap before preparing data.')
                # The same Open Images image can occur under several animal folders.
                old['boxes'].update(boxes)
                old['sources'].append(str(image))
            else:
                records[digest] = {'image': image, 'split': split, 'boxes': set(boxes),
                                   'sources': [str(image)], 'id': digest}
    return list(records.values()), issues


def assign_validation(records, fraction: float, seed: int):
    if not 0 < fraction < 1:
        raise ValueError('--val-ratio must be between 0 and 1')
    if any(r['split'] == 'val' for r in records):
        return  # Honor an existing validation split.
    groups = defaultdict(list)
    for record in records:
        if record['split'] == 'train':
            # Approximate stratification for multi-label images; keep images indivisible.
            primary = min(box[0] for box in record['boxes'])
            groups[primary].append(record)
    rng = random.Random(seed)
    remaining = Counter(name for r in records if r['split'] == 'train'
                        for name in {box[0] for box in r['boxes']})
    for name in sorted(groups):
        group = sorted(groups[name], key=lambda r: r['id'])
        rng.shuffle(group)
        count = min(len(group)-1, max(1, round(len(group)*fraction))) if len(group) > 1 else 0
        moved = 0
        for record in group:
            if moved >= count:
                break
            labels = {box[0] for box in record['boxes']}
            if any(remaining[name] <= 1 for name in labels):
                continue
            record['split'] = 'val'
            remaining.subtract(labels)
            moved += 1
    if not any(r['split'] == 'val' for r in records):
        raise ValueError('Too few training images to create validation; provide a val/ split')


def preview(records, names, output: Path, limit: int):
    ids = {name: i for i, name in enumerate(names)}
    for split in ('train', 'val', 'test'):
        selected = [r for r in records if r['split'] == split][:limit]
        for index, record in enumerate(selected):
            with Image.open(record['image']) as im:
                im = im.convert('RGB')
                im.thumbnail((960, 720))
                draw = ImageDraw.Draw(im)
                w, h = im.size
                for name, x, y, bw, bh in sorted(record['boxes']):
                    rect = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
                    draw.rectangle(rect, outline='lime', width=2)
                    draw.text((rect[0], rect[1]), f'{ids[name]}: {name}', fill='red')
                target = output / 'preview' / f'{split}_{index:03d}.jpg'
                target.parent.mkdir(parents=True, exist_ok=True)
                im.save(target)


def prepare(source: Path, output: Path, val_ratio=0.15, seed=42,
            skip_invalid=False, preview_count=8, copy_mode='auto'):
    if copy_mode not in ('auto', 'copy', 'hardlink'):
        raise ValueError('copy_mode must be auto, copy or hardlink')
    if preview_count < 0:
        raise ValueError('--preview-count must be >= 0')
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new output directory.')
    actual = locate_root(source)
    print(f'Input dataset: {actual}', flush=True)
    records, issues = collect(actual, skip_invalid, show_progress=True)
    assign_validation(records, val_ratio, seed)
    counts = Counter(r['split'] for r in records)
    if any(counts[s] == 0 for s in ('train', 'val', 'test')):
        raise ValueError(f'Every split must contain usable images: {dict(counts)}')
    names = sorted({box[0] for r in records for box in r['boxes']})
    train_names = {box[0] for r in records if r['split'] == 'train' for box in r['boxes']}
    if set(names) - train_names:
        raise ValueError(f'Classes without training examples: {sorted(set(names)-train_names)}')
    ids = {name: i for i, name in enumerate(names)}
    # Validate everything before writing. Refuse existing destinations to avoid stale labels.
    output.mkdir(parents=True)
    distribution = {s: Counter() for s in ('train', 'val', 'test')}
    manifest = []
    storage_modes = Counter()
    for record in tqdm(records, desc='Write YOLO', unit='img'):
        split, digest = record['split'], record['id']
        image_dir, label_dir = output / 'images' / split, output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / (digest + record['image'].suffix.lower())
        mode = copy_mode
        if mode in ('auto', 'hardlink'):
            try:
                os.link(record['image'], target)
                mode = 'hardlink'
            except OSError:
                if copy_mode == 'hardlink':
                    raise
                mode = 'copy'
        if mode == 'copy':
            shutil.copy2(record['image'], target)
        storage_modes[mode] += 1
        lines = []
        for name, *coords in sorted(record['boxes']):
            lines.append(f'{ids[name]} ' + ' '.join(f'{v:.8f}' for v in coords))
            distribution[split][name] += 1
        (label_dir / f'{digest}.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        manifest.append({'id': digest, 'split': split, 'sources': record['sources']})
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'names': dict(enumerate(names))}
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    report = {'source': str(actual), 'seed': seed, 'val_ratio': val_ratio,
              'images': dict(counts), 'classes': names, 'boxes_per_class': distribution,
              'issues': issues, 'manifest': manifest, 'image_storage_modes': dict(storage_modes),
              'missing_classes': {s: sorted(set(names)-set(distribution[s])) for s in distribution}}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    preview(records, names, output, preview_count)
    print(f'Prepared {len(records)} unique images, {len(names)} classes: {dict(counts)}')
    print(f'Skipped {len(issues)} invalid images; review report.json and preview/.')
    print(f'Dataset YAML: {output / "data.yaml"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('datasets/animals_yolo'))
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-invalid', action='store_true', help='Skip whole images with invalid/missing labels; log in report.json')
    parser.add_argument('--preview-count', type=int, default=8, help='Preview images per split')
    parser.add_argument('--copy-mode', choices=['auto', 'copy', 'hardlink'], default='auto',
                        help='auto: hardlink images when supported, otherwise copy; hardlink fails instead of copying')
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.val_ratio, args.seed, args.skip_invalid, args.preview_count, args.copy_mode)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
