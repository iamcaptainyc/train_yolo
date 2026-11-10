"""Export YOLO detection previews, optionally filtering by class, without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import colorsys
from io import BytesIO
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml

from merge_yolo_ids import SUFFIXES, images_for, label_for, load_dataset, validate_label
from split_yolo import collect_images, dataset_file, load_split_source, original_split


def collect_source(value, split='all', data=None):
    source = Path(value).expanduser().resolve()
    metadata = None
    if source.suffix.lower() in {'.yaml', '.yml'} and source.is_file():
        metadata = source
    elif source.is_dir():
        try:
            metadata = dataset_file(source)
        except ValueError:
            pass  # A plain image folder need not contain an images/ tree.
    if metadata is not None:
        dataset = load_split_source(metadata)
        if split != 'all' and dataset['config'].get(split):
            images = images_for(dataset, split)
        else:
            images, ignored = collect_images(dataset)
            if ignored:
                print('Warning: ignored missing dataset paths: ' + ', '.join(ignored), file=sys.stderr)
            if split != 'all':
                images = [p for p in images if original_split(p, dataset) == split]
        root = metadata.parent
        names = dataset['names']
    else:
        if split != 'all':
            raise ValueError('--split requires a dataset directory or YAML')
        if source.is_file() and source.suffix.lower() in SUFFIXES:
            images, root = [source], source.parent
        elif source.is_dir():
            images = [p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES]
            root = source
        else:
            raise ValueError(f'Expected a dataset, image folder or image: {source}')
        names = {}
    if data:
        names = load_dataset(data)['names']
    images = sorted({p.resolve() for p in images}, key=str)
    if not images:
        raise ValueError(f'No images found for {source} (split={split})')
    return images, root, names


def resolve_label(image, root, labels=None):
    if labels is not None:
        # --labels mirrors the tree relative to the selected image directory.
        return Path(labels) / image.relative_to(root).with_suffix('.txt')
    try:
        return label_for(image)
    except ValueError:
        return image.with_suffix('.txt')


def read_boxes(path, names):
    raw, _ = validate_label(path, {'names': names, 'reserved': set()}, check_class_ids=bool(names))
    return [(int(row[0]), *map(float, row[1:]))
            for line in raw.decode('utf-8-sig').splitlines() if (row := line.split())]


def parse_classes(values, names):
    if values is None:
        return None
    selected = set()
    for value in values:
        text = str(value)
        if text.isdecimal():
            cid = int(text)
            if names and cid not in names:
                raise ValueError(f'Unknown class ID: {cid}')
            selected.add(cid)
        else:
            matches = {cid for cid, name in names.items() if name == text}
            if not matches:
                raise ValueError(f'Unknown class name: {text!r}; use a numeric ID or provide --data with class names')
            selected.update(matches)
    if not selected:
        raise ValueError('Provide at least one class ID or name')
    return selected


def filter_images(images, root, names, labels, class_ids):
    matches, issues = [], []
    for image in images:
        record = {'image': str(image)}
        try:
            label = resolve_label(image, root, labels)
            record['label'] = str(label)
            if not label.is_file():
                record.update(status='missing_label', error='Missing label; class membership cannot be determined')
                issues.append(record)
                continue
            if any(box[0] in class_ids for box in read_boxes(label, names)):
                matches.append(image)
        except (OSError, ValueError) as exc:
            record.update(status='failed', error=str(exc))
            issues.append(record)
    for issue in issues:
        print(f'Label scan: {issue["image"]}: {issue["error"]}', file=sys.stderr)
    return matches, issues


def load_font(path=None, size=18):
    if path:
        return ImageFont.truetype(str(path), size)
    for candidate in ('C:/Windows/Fonts/msyh.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def class_color(cid):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb((cid * 0.61803398875) % 1, 0.8, 0.75))


def draw_legend(image, entries, font):
    """Add a separate, wrapping legend above the image without covering pixels."""
    width = max(image.width, 160)
    measure = ImageDraw.Draw(image)
    padding, swatch, gap = 12, 14, 6
    line_height = max(swatch, measure.textbbox((0, 0), 'Ag国', font=font)[3]) + 6
    text_limit = width - 2*padding - swatch - gap
    x, y, row_height = padding, padding, 0
    layout = []
    for color, text in entries:
        lines, current = [], ''
        for char in text:
            if current and measure.textlength(current + char, font=font) > text_limit:
                lines.append(current)
                current = ''
            current += char
        lines.append(current)
        item_width = swatch + gap + max(measure.textlength(line, font=font) for line in lines)
        item_height = line_height * len(lines)
        if x > padding and x + item_width > width - padding:
            x, y, row_height = padding, y + row_height + gap, 0
        layout.append((x, y, color, lines))
        x += item_width + 20
        row_height = max(row_height, item_height)
    header_height = y + row_height + padding
    canvas = Image.new('RGB', (width, image.height + header_height), '#f5f5f5')
    draw = ImageDraw.Draw(canvas)
    for x, y, color, lines in layout:
        draw.rectangle((x, y + 3, x + swatch - 1, y + swatch + 2), fill=color)
        for index, line in enumerate(lines):
            left, top, _, _ = draw.textbbox((0, 0), line, font=font)
            draw.text((x + swatch + gap - left, y + index*line_height - top),
                      line, font=font, fill='#202020')
    canvas.paste(image, (0, header_height))
    return canvas


def render_image(path, boxes, names, font, max_size=1600, line_width=3, status='ok'):
    with Image.open(path) as original:
        # Coordinates refer to stored pixels. Do not rotate using EXIF alone.
        image = original.convert('RGB')
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    width, height = image.size
    draw = ImageDraw.Draw(image, 'RGBA')
    for cid, x, y, w, h in boxes:
        x1, y1 = max(0, round((x-w/2)*width)), max(0, round((y-h/2)*height))
        x2, y2 = min(width-1, round((x+w/2)*width)), min(height-1, round((y+h/2)*height))
        x1, y1 = min(x1, x2), min(y1, y2)
        color = class_color(cid)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 128), width=line_width)
    entries = [(class_color(cid), f'{cid}: {names[cid]}' if cid in names else str(cid))
               for cid in sorted({box[0] for box in boxes})]
    if not entries:
        entries = [('#b02020', 'MISSING LABEL')] if status == 'missing_label' else [('#505050', 'EMPTY LABEL')]
    return draw_legend(image, entries, font)


def visualize(source, output, *, split='all', data=None, labels=None, limit=None,
              seed=42, max_size=1600, line_width=3, font=None, classes=None):
    if (limit is not None and limit < 0) or max_size < 1 or line_width < 1:
        raise ValueError('limit must be >= 0; max-size and line-width must be > 0')
    images, root, names = collect_source(source, split, data)
    class_ids = parse_classes(classes, names)
    if limit is None:
        limit = 0 if class_ids is not None else 100
    output = Path(output).expanduser().resolve()
    labels = Path(labels).expanduser().resolve() if labels else None
    if labels is not None and not labels.is_dir():
        raise ValueError(f'Labels directory does not exist: {labels}')
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')
    loaded_font = load_font(font)
    total = len(images)
    scan_issues = []
    if class_ids is not None:
        images, scan_issues = filter_images(images, root, names, labels, class_ids)
        print(f'Class filter {sorted(class_ids)}: {len(images)}/{total} matching images')
    matched = len(images)
    if limit and matched > limit:
        images = sorted(random.Random(seed).sample(images, limit), key=str)
    output.mkdir(parents=True)
    records = []
    for index, image in enumerate(images, 1):
        record = {'image': str(image)}
        try:
            label = resolve_label(image, root, labels)
            record['label'] = str(label)
            boxes = read_boxes(label, names) if label.is_file() else []
            if class_ids is not None:
                boxes = [box for box in boxes if box[0] in class_ids]
                if not boxes:
                    raise ValueError('Selected class annotations changed since the label scan')
            status = 'missing_label' if not label.is_file() else ('ok' if boxes else 'empty_label')
            preview = render_image(image, boxes, names, loaded_font, max_size, line_width, status)
            # Index avoids collisions, including external YAML paths and jpg/png twins.
            target = output / f'{index:06d}_{image.name}.jpg'
            preview.save(target, quality=95)
            record.update(status=status, boxes=len(boxes), output=target.name)
        except (OSError, ValueError, SyntaxError) as exc:
            record.update(status='failed', error=str(exc))
            print(f'Failed: {image}: {exc}', file=sys.stderr)
        records.append(record)
    counts = dict(Counter(r['status'] for r in [*records, *scan_issues]))
    report = {'source': str(Path(source).expanduser().resolve()), 'split': split, 'seed': seed,
              'total_images': total, 'matched_images': matched, 'selected_images': len(images), 'counts': counts,
              'classes': sorted(class_ids) if class_ids is not None else None, 'limit': limit,
              'names': names, 'images': records, 'scan_issues': scan_issues}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Previewed {len(images)}/{total} images: {counts}\nOutput: {output}')
    return report


def visualize_dataset(source, classes=None):
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Expected a YOLO dataset directory: {root}')
    dataset = load_split_source(dataset_file(root))
    names = dataset['names']
    selected = parse_classes(classes, names)
    splits = ('train', 'val', 'test')
    output = root/'preview'
    if output.exists() or output.is_symlink():
        raise ValueError(f'Move the existing preview directory before generating a new preview: {output}')
    images_by_split = {split: images_for(dataset, split) for split in splits}
    if not any(dataset['config'].get(split) for split in splits):
        images, _ = collect_images(dataset)
        for image in images:
            images_by_split[original_split(image, dataset)].append(image)
    if not any(images_by_split.values()):
        raise ValueError(f'No images found in train, val or test: {root}')

    # Keep directory names portable and distinguish names that collide on Windows.
    folder_names = {cid: re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:100].rstrip(' .')
                    for cid, name in names.items()}
    collisions = Counter(name.casefold() for name in folder_names.values())
    for cid, name in folder_names.items():
        reserved = re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])', name.split('.')[0], re.I)
        if not name or reserved or name.isdecimal() or collisions[name.casefold()] > 1:
            folder_names[cid] = str(cid)
    font = load_font()
    counts, issues = defaultdict(Counter), []
    output.mkdir()
    for split, images in images_by_split.items():
        unique_images = list(dict.fromkeys(images))
        for index, image in enumerate(tqdm(unique_images, desc=f'Preview {split}', unit='img'), 1):
            try:
                boxes = read_boxes(resolve_label(image, root), names)
                class_ids = {box[0] for box in boxes}
                if selected is not None:
                    class_ids &= selected
                if not class_ids:
                    continue
                preview = render_image(image, boxes, names, font, max_size=960)
                encoded = BytesIO()
                preview.save(encoded, format='JPEG', quality=78, optimize=True)
                jpeg = encoded.getvalue()
                for cid in sorted(class_ids):
                    folder = folder_names.get(cid, str(cid))
                    target = output/folder/split/f'{index:06d}_{image.stem[:100]}.jpg'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(jpeg)
                    counts[folder][split] += 1
            except (OSError, ValueError, SyntaxError) as exc:
                issues.append({'image': str(image), 'split': split, 'error': str(exc)})
    report = {'counts': counts, 'issues': issues, 'class_folders': folder_names}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Exported {sum(sum(v.values()) for v in counts.values())} previews to {output}; issues: {len(issues)}')
    return int(bool(issues))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', help='YOLO dataset directory; save to preview/<class>/<split>')
    parser.add_argument('--source', help='Legacy mode: dataset directory/YAML, image directory, or single image')
    parser.add_argument('--output', default='runs/labels_preview', help='New directory for JPEG previews and report.json')
    parser.add_argument('--data', help='Optional dataset YAML supplying class names')
    parser.add_argument('--labels', help='Labels directory mirroring the selected image directory')
    parser.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    parser.add_argument('--classes', nargs='+', help='Class IDs or exact names; select class folders in dataset mode')
    parser.add_argument('--limit', type=int, default=None,
                        help='Random sample size; 0 exports all (default: all matches with --classes, otherwise 100)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-size', type=int, default=1600, help='Maximum preview width/height')
    parser.add_argument('--line-width', type=int, default=3, help='Box stroke in preview pixels')
    parser.add_argument('--font', help='TTF/TTC font, e.g. a Chinese font on Linux')
    args = parser.parse_args(argv)
    if bool(args.dataset) == bool(args.source):
        parser.error('Provide a dataset directory or --source')
    if args.dataset and any(value != parser.get_default(name) for name, value in vars(args).items()
                            if name not in ('dataset', 'classes')):
        parser.error('Dataset mode accepts a directory and optional --classes; use --source for other options')
    try:
        if args.dataset:
            return visualize_dataset(args.dataset, args.classes)
        del args.dataset
        report = visualize(**vars(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return int(bool(report['counts'].get('failed') or report['counts'].get('missing_label')))


if __name__ == '__main__':
    sys.exit(main())
