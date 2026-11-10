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

from PIL import Image, ImageOps
from tqdm import tqdm
import yaml

from merge_yolo_ids import SPLITS, SUFFIXES
from split_yolo import assign_splits


def locate_directories(source):
    roots = {'images': [], 'annotations': []}
    for path in source.rglob('*'):
        name = path.name.casefold()
        if name in ('images', 'jpegimages', 'annotation', 'annotations') and path.is_dir():
            roots['images' if name in ('images', 'jpegimages') else 'annotations'].append(path)
    found = []
    for kind, paths in roots.items():
        # Kaggle wraps Images/ and Annotation/ inside images/ and annotations/.
        candidates = [path for path in paths if not any(path in other.parents for other in paths)]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {kind} directory under {source}; found {len(candidates)}. '
                             'Pass the extracted Stanford Dogs dataset root.')
        found.append(candidates[0])
    return tuple(found)


def orient_box(box, orientation):
    """Apply the same EXIF transform as Pillow to normalized box coordinates."""
    x, y, width, height = box
    if orientation in (5, 6, 7, 8):
        x, y = y, x
        width, height = height, width
    if orientation in (2, 3, 6, 7):
        x = 1 - x
    if orientation in (3, 4, 7, 8):
        y = 1 - y
    return x, y, width, height


def read_boxes(annotation, image, coordinates, class_name):
    root = ET.parse(annotation).getroot()
    if root.tag != 'annotation':
        raise ValueError('expected an annotation XML root')
    filename = root.findtext('filename', '').strip()
    # Some source annotations retain the literal filename placeholder.
    if filename not in ('', '%s') and Path(filename).stem != image.stem:
        raise ValueError(f'XML filename {filename!r} does not match {image.name}')
    width = int(root.findtext('size/width', '0'))
    height = int(root.findtext('size/height', '0'))
    with Image.open(image) as pixels:
        stored_size = pixels.size
        # Pillow may already report the display dimensions for an oriented TIFF.
        if pixels.format == 'TIFF':
            stored_size = pixels.tag_v2[256], pixels.tag_v2[257]
        orientation = pixels.getexif().get(274, 1)
        pixels.load()
    upright_size = stored_size[::-1] if orientation in (5, 6, 7, 8) else stored_size
    if (width, height) == stored_size:
        box_orientation = orientation
    elif (width, height) == upright_size:
        box_orientation = 1
    else:
        raise ValueError(f'XML size {(width, height)} does not match stored {stored_size} or upright {upright_size} image')

    boxes, clipped = [], 0
    for index, obj in enumerate(root.findall('object'), 1):
        name = class_name if class_name is not None else obj.findtext('name', '').strip()
        if not name:
            raise ValueError(f'object {index}: missing class name')
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
        box = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
        boxes.append((name, orient_box(box, box_orientation)))
    if not boxes:
        raise ValueError('XML contains no object boxes')
    return boxes, clipped, orientation


def prepare(source, output, class_mode='dog', coordinates='edge', ratios=(8, 1, 1), seed=42,
            copy_mode='auto', skip_invalid=True):
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
            if class_mode == 'breed' and len(relative.parts) < 2:
                raise ValueError('expected a breed directory above the image')
            candidates = [annotation_root / relative.with_suffix(ext) for ext in ('', '.xml')]
            annotations = [path for path in candidates if path.is_file()]
            if len(annotations) != 1:
                raise ValueError(f'expected one annotation at {candidates[0]} or {candidates[1]}')
            if label_path.as_posix().casefold() in label_paths:
                raise ValueError(f'output label path is shared by another image: {label_path}')
            class_name = 'dog' if class_mode == 'dog' else None
            if class_mode == 'breed':
                class_name = re.sub(r'^n\d{8}-', '', relative.parts[0])
            boxes, clipped_count, orientation = read_boxes(annotations[0], image, coordinates, class_name)
            with image.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        except (ET.ParseError, OSError, ValueError, Image.DecompressionBombError) as exc:
            if not skip_invalid:
                raise ValueError(f'{image}: {exc}') from exc
            issues.append({'image': str(image), 'error': str(exc)})
            continue
        label_paths.add(label_path.as_posix().casefold())
        clipped += clipped_count
        records.append({'image': image, 'relative': relative, 'boxes': boxes, 'identity': digest,
                        'orientation': orientation})
    if not records:
        detail = f' First rejected image: {issues[0]}' if issues else ''
        raise ValueError(f'No usable image/annotation pairs under {image_root}.{detail}')

    _, groups = assign_splits(records, ratios, seed)
    names = sorted({name for record in records for name, _ in record['boxes']})
    ids = {name: index for index, name in enumerate(names)}
    counts = Counter(record['split'] for record in records)
    boxes_per_class = {split: Counter() for split in SPLITS}
    storage = Counter()
    output.mkdir(parents=True)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Conversion in progress. If interrupted, use a new output directory.\n', encoding='utf-8')
    for record in tqdm(records, desc='Write YOLO', unit='img'):
        split, relative = record['split'], record['relative']
        image_out = output/'images'/split/relative
        label_out = (output/'labels'/split/relative).with_suffix('.txt')
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        mode = copy_mode
        if record['orientation'] != 1:
            with Image.open(record['image']) as source_image:
                upright = ImageOps.exif_transpose(source_image).convert('RGB')
                upright.info.clear()
                options = {'quality': 95, 'subsampling': 0} if image_out.suffix.lower() in ('.jpg', '.jpeg') else {}
                upright.save(image_out, **options)
            mode = 'exif_normalized'
        elif mode != 'copy':
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
        lines = []
        for name, box in record['boxes']:
            lines.append(f'{ids[name]} ' + ' '.join(f'{value:.8f}' for value in box))
            boxes_per_class[split][name] += 1
        label_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
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
          f'Normalized {storage["exif_normalized"]} EXIF orientations. Skipped {len(issues)} invalid images. '
          f'Clipped {clipped} boxes.\nDataset YAML: {output / "data.yaml"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Extracted Stanford Dogs dataset root')
    parser.add_argument('--output', type=Path, default=Path('datasets/stanford_dogs_yolo'))
    parser.add_argument('--class-mode', choices=('dog', 'breed', 'xml'), default='dog',
                        help='Use one dog class, breed-folder names, or each XML object name')
    parser.add_argument('--coordinates', choices=('edge', 'voc'), default='edge',
                        help='Use XML box edges as given, or convert 1-based inclusive VOC coordinates')
    parser.add_argument('--ratios', type=float, nargs=3, default=(8, 1, 1), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy-mode', choices=('auto', 'hardlink', 'copy'), default='auto',
                        help='Storage for images that do not require EXIF correction')
    invalid = parser.add_mutually_exclusive_group()
    invalid.add_argument('--skip-invalid', action='store_true', help='Skip invalid pairs and list them in report.json (default)')
    invalid.add_argument('--strict', dest='skip_invalid', action='store_false', help='Stop at the first invalid image/label pair')
    parser.set_defaults(skip_invalid=True)
    try:
        prepare(**vars(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
