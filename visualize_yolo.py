"""Export sampled YOLO detection annotations as images without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter
import colorsys
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont
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


def visualize(source, output, *, split='all', data=None, labels=None, limit=100,
              seed=42, max_size=1600, line_width=3, font=None):
    if limit < 0 or max_size < 1 or line_width < 1:
        raise ValueError('limit must be >= 0; max-size and line-width must be > 0')
    images, root, names = collect_source(source, split, data)
    output = Path(output).expanduser().resolve()
    labels = Path(labels).expanduser().resolve() if labels else None
    if labels is not None and not labels.is_dir():
        raise ValueError(f'Labels directory does not exist: {labels}')
    if output.exists():
        raise ValueError(f'Use a new output directory: {output}')
    loaded_font = load_font(font)
    total = len(images)
    if limit and total > limit:
        images = sorted(random.Random(seed).sample(images, limit), key=str)
    output.mkdir(parents=True)
    records = []
    for index, image in enumerate(images, 1):
        record = {'image': str(image)}
        try:
            label = resolve_label(image, root, labels)
            record['label'] = str(label)
            boxes = read_boxes(label, names) if label.is_file() else []
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
    counts = dict(Counter(r['status'] for r in records))
    report = {'source': str(Path(source).expanduser().resolve()), 'split': split, 'seed': seed,
              'total_images': total, 'selected_images': len(images), 'counts': counts,
              'names': names, 'images': records}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Previewed {len(images)}/{total} images: {counts}\nOutput: {output}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Dataset directory/YAML, image directory, or single image')
    parser.add_argument('--output', default='runs/labels_preview', help='New directory for JPEG previews and report.json')
    parser.add_argument('--data', help='Optional dataset YAML supplying class names')
    parser.add_argument('--labels', help='Labels directory mirroring the selected image directory')
    parser.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    parser.add_argument('--limit', type=int, default=100, help='Random sample size; 0 exports all images (default: 100)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-size', type=int, default=1600, help='Maximum preview width/height')
    parser.add_argument('--line-width', type=int, default=3, help='Box stroke in preview pixels')
    parser.add_argument('--font', help='TTF/TTC font, e.g. a Chinese font on Linux')
    args = parser.parse_args(argv)
    try:
        report = visualize(**vars(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return int(bool(report['counts'].get('failed') or report['counts'].get('missing_label')))


if __name__ == '__main__':
    sys.exit(main())
