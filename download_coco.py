"""Download a COCO 2017 animal subset and export its official boxes as YOLO."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import zipfile

from tqdm import tqdm

from download_openimages import download

BASE = 'https://images.cocodataset.org'
ARCHIVE = 'annotations_trainval2017.zip'
ANIMALS = ('bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe')
SPLITS = ('train', 'val')


def select_boxes(data, names):
    class_ids = {name: cid for cid, name in enumerate(names)}
    classes = {c['id']: class_ids[c['name']] for c in data['categories'] if c['name'] in class_ids}
    available = {c['name'] for c in data['categories']}
    if missing := set(names) - available:
        raise ValueError(f'Classes absent from COCO annotations: {", ".join(sorted(missing))}')
    images = {image['id']: image for image in data['images']}
    selected = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in classes or ann.get('iscrowd'):
            continue
        image_id = ann['image_id']
        image = images[image_id]
        width, height = image['width'], image['height']
        x, y, w, h = ann['bbox']
        if not all(math.isfinite(v) for v in (x, y, w, h, width, height)) or min(w, h, width, height) <= 0:
            continue
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x+w), min(height, y+h)
        if right <= left or bottom <= top:
            continue
        coords = ((left+right)/(2*width), (top+bottom)/(2*height),
                  (right-left)/width, (bottom-top)/height)
        selected[image_id].append((ann, classes[ann['category_id']], coords))
    return images, selected


def export_split(output, split, data, names, workers, retries):
    images, selected = select_boxes(data, names)
    image_dir = output/'images'/split
    label_dir = output/'labels'/split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_images, class_boxes = Counter(), Counter()
    failures = []

    def fetch(image_id):
        filename = images[image_id]['file_name']
        if not re.fullmatch(r'[0-9]{12}\.jpg', filename):
            raise ValueError(f'Unexpected COCO 2017 file name: {filename}')
        try:
            download(f'{BASE}/{split}2017/{filename}', image_dir/filename, retries)
        except (OSError, ValueError) as exc:
            return {'file': filename, 'error': str(exc)}
        boxes = selected[image_id]
        label = label_dir/Path(filename).with_suffix('.txt')
        temp = label.with_suffix('.txt.part')
        temp.write_text(''.join(f'{cid} ' + ' '.join(format(v, '.12g') for v in box) + '\n'
                                for _, cid, box in boxes), encoding='utf-8')
        temp.replace(label)
        return None

    image_ids = sorted(selected)
    batch_size = workers * 2
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(
            total=len(image_ids), desc=f'COCO {split}2017', unit='img') as progress:
        for offset in range(0, len(image_ids), batch_size):
            batch = image_ids[offset:offset+batch_size]
            for failure in pool.map(fetch, batch):
                progress.update()
                if failure:
                    failures.append(failure)
    for boxes in selected.values():
        class_boxes.update(names[cid] for _, cid, _ in boxes)
        class_images.update({names[cid] for _, cid, _ in boxes})
    subset = {**data, 'images': [images[i] for i in image_ids],
              'annotations': [ann for boxes in selected.values() for ann, _, _ in boxes],
              'categories': [c for c in data['categories'] if c['name'] in names]}
    stats = {'images': len(image_ids), 'images_by_class': dict(class_images),
             'boxes_by_class': dict(class_boxes)}
    return subset, stats, failures


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)


def run(output, names, archive, workers, retries):
    output = output.resolve()
    meta = output/'_metadata'
    state = meta/'classes.json'
    if state.is_file():
        if json.loads(state.read_text(encoding='utf-8')) != names:
            raise ValueError('Class selection changed; use a new output directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty or previously initialized output directory')
        meta.mkdir(parents=True, exist_ok=True)
        write_json(state, names)
    if archive is None:
        archive = meta/ARCHIVE
        download(f'{BASE}/annotations/{ARCHIVE}', archive, retries, True)
    stats, failures = {}, {}
    with zipfile.ZipFile(archive) as source:
        for split in SPLITS:
            member = f'annotations/instances_{split}2017.json'
            with source.open(member) as stream:
                data = json.load(stream)
            subset, stats[split], failures[split] = export_split(output, split, data, names, workers, retries)
            write_json(meta/f'instances_{split}2017_subset.json', subset)
            del data, subset
    failure_path = meta/'failures.json'
    if any(failures.values()):
        write_json(failure_path, failures)
        raise OSError(f'Image downloads incomplete. See {failure_path} and rerun the same command.')
    if any(not stat['images'] for stat in stats.values()):
        raise ValueError('Both official splits need at least one image with a valid selected box')
    failure_path.unlink(missing_ok=True)
    write_json(output/'report.json', stats)
    write_json(output/'data.yaml', {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
                                    'nc': len(names), 'names': names})
    for split, stat in stats.items():
        print(f'{split}: {stat["images"]:,} images; {stat["images_by_class"]}')
    print(f'YOLO dataset: {output / "data.yaml"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('yolo_datasets/coco_cow_elephant'))
    parser.add_argument('--classes', nargs='+', type=str.lower, choices=(*ANIMALS, 'cattle'),
                        default=['cow', 'elephant'])
    parser.add_argument('--annotations-zip', type=Path, help='Reuse the local official train/val annotation archive')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--retries', type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0:
        parser.error('Require positive workers and nonnegative retries')
    names = sorted({'cow' if name == 'cattle' else name for name in args.classes})
    try:
        run(args.output, names, args.annotations_zip, args.workers, args.retries)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Rerun the same command to resume.\n')
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
