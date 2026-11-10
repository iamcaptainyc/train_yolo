"""Convert the extracted Kaggle Stanford Dogs images and XML boxes to YOLO."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import yaml

from merge_yolo_ids import SPLITS, SUFFIXES
from split_yolo import assign_splits


def locate_directories(source):
    roots = {'images': [], 'annotations': []}
    for path in source.rglob('*'):
        name = path.name.casefold()
        if name in ('images', 'annotation', 'annotations') and path.is_dir():
            roots['images' if name == 'images' else 'annotations'].append(path)
    found = []
    for kind, paths in roots.items():
        # Kaggle wraps Images/ and Annotation/ inside images/ and annotations/.
        candidates = [path for path in paths if not any(path in other.parents for other in paths)]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {kind} directory under {source}; found {len(candidates)}. '
                             'Pass the extracted Stanford Dogs dataset root.')
        found.append(candidates[0])
    return tuple(found)


def read_boxes(annotation, image, coordinates):
    root = ET.parse(annotation).getroot()
    if root.tag != 'annotation':
        raise ValueError('expected an annotation XML root')
    filename = root.findtext('filename', '').strip()
    if filename and Path(filename).stem != image.stem:
        raise ValueError(f'XML filename {filename!r} does not match {image.name}')
    width = int(root.findtext('size/width', '0'))
    height = int(root.findtext('size/height', '0'))
    with Image.open(image) as pixels:
        pixels.load()
        if width <= 0 or height <= 0 or pixels.size != (width, height):
            raise ValueError(f'XML size {(width, height)} does not match image size {pixels.size}')
        if pixels.getexif().get(274, 1) != 1:
            raise ValueError('image has a non-default EXIF orientation')

    boxes, clipped = [], 0
    for index, obj in enumerate(root.findall('object'), 1):
        x1, y1, x2, y2 = (float(obj.findtext(f'bndbox/{key}', 'nan'))
                           for key in ('xmin', 'ymin', 'xmax', 'ymax'))
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f'object {index}: missing or non-finite box coordinate')
        if coordinates == 'voc':
            x1 -= 1
            y1 -= 1
        bounded = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
        clipped += bounded != (x1, y1, x2, y2)
        x1, y1, x2, y2 = bounded
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f'object {index}: box has no positive area inside the image')
        boxes.append(((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height))
    if not boxes:
        raise ValueError('XML contains no object boxes')
    return boxes, clipped


def prepare(source, output, class_mode='dog', coordinates='edge', ratios=(8, 1, 1), seed=42,
            copy_mode='auto', skip_invalid=False):
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'Source directory does not exist: {source}')
    if output.exists() or output.is_relative_to(source):
        raise ValueError(f'Choose a new output directory outside the source: {output}')
    image_root, annotation_root = locate_directories(source)
    images = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
    records, issues, clipped = [], [], 0
    label_paths = set()
    for image in tqdm(images, desc='Check Stanford Dogs', unit='img'):
        relative = image.relative_to(image_root)
        label_path = relative.with_suffix('.txt')
        try:
            if len(relative.parts) < 2:
                raise ValueError('expected a breed directory above the image')
            candidates = [annotation_root / relative.with_suffix(ext) for ext in ('', '.xml')]
            annotations = [path for path in candidates if path.is_file()]
            if len(annotations) != 1:
                raise ValueError(f'expected one annotation at {candidates[0]} or {candidates[1]}')
            if label_path.as_posix().casefold() in label_paths:
                raise ValueError(f'output label path is shared by another image: {label_path}')
            boxes, clipped_count = read_boxes(annotations[0], image, coordinates)
            with image.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        except (ET.ParseError, OSError, ValueError, Image.DecompressionBombError) as exc:
            if not skip_invalid:
                raise ValueError(f'{image}: {exc}') from exc
            issues.append({'image': str(image), 'error': str(exc)})
            continue
        label_paths.add(label_path.as_posix().casefold())
        clipped += clipped_count
        breed = re.sub(r'^n\d{8}-', '', relative.parts[0])
        records.append({'image': image, 'relative': relative, 'boxes': boxes, 'identity': digest,
                        'name': 'dog' if class_mode == 'dog' else breed})
    if not records:
        detail = f' First rejected image: {issues[0]}' if issues else ''
        raise ValueError(f'No usable image/annotation pairs under {image_root}.{detail}')

    _, groups = assign_splits(records, ratios, seed)
    ids = {name: index for index, name in enumerate(sorted({record['name'] for record in records}))}
    counts = Counter(record['split'] for record in records)
    boxes_per_class = {split: Counter() for split in SPLITS}
    storage = Counter()
    output.mkdir(parents=True)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Conversion in progress. If interrupted, use a new output directory.\n', encoding='utf-8')
    for record in tqdm(records, desc='Write YOLO', unit='img'):
        split, relative, name = record['split'], record['relative'], record['name']
        image_out = output/'images'/split/relative
        label_out = (output/'labels'/split/relative).with_suffix('.txt')
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        mode = copy_mode
        if mode != 'copy':
            try:
                os.link(record['image'], image_out)
                mode = 'hardlink'
            except OSError:
                if copy_mode == 'hardlink':
                    raise
                mode = 'copy'
        if mode == 'copy':
            shutil.copy2(record['image'], image_out)
        storage[mode] += 1
        lines = [f'{ids[name]} ' + ' '.join(f'{value:.8f}' for value in box) for box in record['boxes']]
        label_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        boxes_per_class[split][name] += len(lines)
    config = {'path': output.as_posix(), **{split: f'images/{split}' for split in SPLITS},
              'names': {cid: name for name, cid in ids.items()}}
    report = {'source': str(source), 'class_mode': class_mode, 'coordinates': coordinates,
              'ratios': ratios, 'seed': seed, 'images': dict(counts), 'content_groups': groups,
              'boxes_per_class': boxes_per_class, 'clipped_boxes': clipped, 'issues': issues,
              'image_storage_modes': dict(storage)}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    marker.unlink()
    print(f'Converted {len(records)} images, {len(ids)} classes: {dict(counts)}. '
          f'Skipped {len(issues)} invalid images; clipped {clipped} boxes.\nDataset YAML: {output / "data.yaml"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Extracted Stanford Dogs dataset root')
    parser.add_argument('--output', type=Path, default=Path('datasets/stanford_dogs_yolo'))
    parser.add_argument('--class-mode', choices=('dog', 'breed'), default='dog', help='Single dog class or breed-folder classes')
    parser.add_argument('--coordinates', choices=('edge', 'voc'), default='edge',
                        help='Use XML box edges as given, or convert 1-based inclusive VOC coordinates')
    parser.add_argument('--ratios', type=float, nargs=3, default=(8, 1, 1), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy-mode', choices=('auto', 'hardlink', 'copy'), default='auto')
    parser.add_argument('--skip-invalid', action='store_true', help='Skip invalid image/label pairs and list them in report.json')
    try:
        prepare(**vars(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
