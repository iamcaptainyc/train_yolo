"""Download a COCO 2017 animal subset and export its official boxes as YOLO."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import IncompleteRead
import json
import math
from pathlib import Path
import re
import sys
from threading import Event
import zipfile

from tqdm import tqdm

from coco_hf_zip import RangeZip
from download_openimages import download

BASE = 'https://images.cocodataset.org'
GCS_BASE = 'https://storage.googleapis.com/images.cocodataset.org'
ARCHIVE = 'annotations_trainval2017.zip'
ARCHIVE_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ARCHIVE_SIZE = 252_907_541
HF_REPO_PATH = '/datasets/pcuenq/coco-2017-mirror/resolve/5200d2cffae8121713bec767b4693bdafc0eeb0b'
HF_MIRROR = 'https://hf-mirror.com' + HF_REPO_PATH
ARCHIVE_URLS = (
    f'{BASE}/annotations/{ARCHIVE}',
    f'{GCS_BASE}/annotations/{ARCHIVE}',
    f'https://huggingface.co{HF_REPO_PATH}/{ARCHIVE}',
)
ANIMALS = ('bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe')
SPLITS = ('train', 'val')


def valid_archive(path):
    if not path.is_file() or path.stat().st_size != ARCHIVE_SIZE:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == ARCHIVE_SHA256


def fetch_archive(path, retries, urls=ARCHIVE_URLS):
    if valid_archive(path):
        return
    path.unlink(missing_ok=True)
    partial = path.with_name(path.name+'.part')
    if valid_archive(partial):
        partial.replace(path)
        return
    if partial.is_file() and partial.stat().st_size >= ARCHIVE_SIZE:
        partial.unlink()
    errors = []
    for index, url in enumerate(urls):
        print(f'COCO annotation archive: {url}', file=sys.stderr, flush=True)
        try:
            # The endpoints serve the same archive. Verify bytes before using it.
            download(url, path, retries if index == len(urls)-1 else min(retries, 1), True)
            if not valid_archive(path):
                path.unlink(missing_ok=True)
                raise ValueError('Archive size or SHA-256 does not match COCO 2017')
            return
        except (OSError, ValueError, IncompleteRead) as exc:
            errors.append(f'{url}: {exc}')
            if partial.is_file() and partial.stat().st_size >= ARCHIVE_SIZE:
                if valid_archive(partial):
                    partial.replace(path)
                    return
                partial.unlink()
    raise OSError('COCO annotation download failed. Rerun to resume or use --annotations-zip.\n'
                  + '\n'.join(errors))


def fetch_image(relative, path, retries, prefer_gcs):
    first, second = (GCS_BASE, BASE) if prefer_gcs.is_set() else (BASE, GCS_BASE)
    try:
        download(f'{first}/{relative}', path, min(retries, 1))
    except (OSError, ValueError) as primary:
        try:
            download(f'{second}/{relative}', path, retries)
        except (OSError, ValueError) as fallback:
            raise OSError(f'{first}: {primary}; {second}: {fallback}') from fallback
        if second == GCS_BASE:
            prefer_gcs.set()
        else:
            prefer_gcs.clear()


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


def export_split(output, split, data, names, workers, retries, prefer_gcs, source, local_only):
    images, selected = select_boxes(data, names)
    image_dir = output/'images'/split
    label_dir = output/'labels'/split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_images, class_boxes = Counter(), Counter()
    failures = []
    filenames = {image_id: images[image_id]['file_name'] for image_id in selected}
    if any(not re.fullmatch(r'[0-9]{12}\.jpg', name) for name in filenames.values()):
        raise ValueError(f'Unexpected file name in COCO {split}2017 annotations')
    image_ids = sorted(selected)
    cached_images = {image_id for image_id, filename in filenames.items()
                     if (path := image_dir/filename).is_file() and path.stat().st_size}
    pending = [image_id for image_id in image_ids if image_id not in cached_images or
               not ((label := label_dir/Path(filenames[image_id]).with_suffix('.txt')).is_file()
                    and label.stat().st_size)]
    print(f'COCO {split}2017: {len(image_ids)-len(pending):,} complete, '
          f'{len(pending):,} pending ({len(image_ids)-len(cached_images):,} missing images)',
          flush=True)
    remote = None
    unavailable = 'Image not downloaded (--local-only)' if local_only else None
    if not local_only and source == 'hf-mirror' and len(cached_images) != len(image_ids):
        url = f'{HF_MIRROR}/{split}2017.zip'
        print(f'COCO image archive (HTTP Range): {url}', file=sys.stderr, flush=True)
        try:
            remote = RangeZip(url, retries)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            unavailable = f'Cannot open mirror ZIP: {exc}'
            print(f'COCO {split}2017: {unavailable}', file=sys.stderr)

    def fetch(image_id):
        filename = filenames[image_id]
        target = image_dir/filename
        try:
            if not (target.is_file() and target.stat().st_size):
                if unavailable is not None:
                    return {'file': filename, 'error': unavailable}
                if source == 'hf-mirror':
                    if remote is None:
                        raise OSError('A cached image was removed during export; rerun to retrieve it')
                    partial = target.with_suffix('.jpg.part')
                    partial.write_bytes(remote.image(split, filename))
                    partial.replace(target)
                else:
                    fetch_image(f'{split}2017/{filename}', target, retries, prefer_gcs)
            boxes = selected[image_id]
            label = label_dir/Path(filename).with_suffix('.txt')
            temp = label.with_suffix('.txt.part')
            temp.write_text(''.join(f'{cid} ' + ' '.join(format(v, '.12g') for v in box) + '\n'
                                    for _, cid, box in boxes), encoding='utf-8')
            temp.replace(label)
        except (OSError, ValueError, IncompleteRead) as exc:
            return {'file': filename, 'error': str(exc)}
        return None

    batch_size = workers * 2
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(
            total=len(pending), desc=f'COCO {split}2017 pending', unit='img', disable=not pending) as progress:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset+batch_size]
            for future in as_completed([pool.submit(fetch, image_id) for image_id in batch]):
                failure = future.result()
                progress.update()
                if failure:
                    failures.append(failure)
                    if len(failures) <= 3:
                        progress.write(f'{split}/{failure["file"]}: {failure["error"]}', file=sys.stderr)
                    progress.set_postfix(failed=len(failures), refresh=False)
    failures.sort(key=lambda failure: failure['file'])
    missing = {failure['file'] for failure in failures}
    ready_ids = [image_id for image_id in image_ids if filenames[image_id] not in missing]
    for image_id in ready_ids:
        boxes = selected[image_id]
        class_boxes.update(names[cid] for _, cid, _ in boxes)
        class_images.update({names[cid] for _, cid, _ in boxes})
    subset = {**data, 'images': [images[i] for i in ready_ids],
              'annotations': [ann for image_id in ready_ids for ann, _, _ in selected[image_id]],
              'categories': [c for c in data['categories'] if c['name'] in names]}
    stats = {'images': len(ready_ids), 'requested_images': len(image_ids), 'missing_images': len(missing),
             'images_by_class': dict(class_images),
             'boxes_by_class': dict(class_boxes)}
    return subset, stats, failures


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)


def run(output, names, archive, workers, retries, *, source='hf-mirror', local_only=False):
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
        if local_only:
            if not valid_archive(archive):
                raise ValueError(f'No valid local annotation ZIP at {archive}; provide --annotations-zip')
        else:
            urls = (f'{HF_MIRROR}/{ARCHIVE}',) if source == 'hf-mirror' else ARCHIVE_URLS
            fetch_archive(archive, retries, urls)
    stats, failures = {}, {}
    image_lists = {}
    prefer_gcs = Event()
    with zipfile.ZipFile(archive) as archive_file:
        for split in SPLITS:
            member = f'annotations/instances_{split}2017.json'
            with archive_file.open(member) as stream:
                data = json.load(stream)
            subset, stats[split], failures[split] = export_split(
                output, split, data, names, workers, retries, prefer_gcs, source, local_only)
            write_json(meta/f'instances_{split}2017_subset.json', subset)
            image_lists[split] = ''.join(f'./images/{split}/{im["file_name"]}\n' for im in subset['images'])
            del data, subset
    failure_path = meta/'failures.json'
    partial = any(failures.values())
    if partial:
        if local_only and failure_path.is_file():
            previous = json.loads(failure_path.read_text(encoding='utf-8'))
            for split, errors in failures.items():
                old_errors = {item['file']: item['error'] for item in previous.get(split, [])}
                for item in errors:
                    if item['error'] == 'Image not downloaded (--local-only)':
                        item['error'] = old_errors.get(item['file'], item['error'])
        write_json(failure_path, failures)
        counts = ', '.join(f'{split}: {len(errors)} missing' for split, errors in failures.items())
        print(f'COCO partial dataset ({counts}). See {failure_path}.', file=sys.stderr)
    else:
        failure_path.unlink(missing_ok=True)
    write_json(output/'report.json', stats)
    if any(not stat['images'] for stat in stats.values()):
        raise ValueError('Both official splits need at least one completed image and label to export data.yaml')
    split_paths = {split: f'images/{split}' for split in SPLITS}
    if partial:
        for split, contents in image_lists.items():
            manifest = output/f'{split}.txt'
            temp = manifest.with_suffix('.txt.part')
            temp.write_text(contents, encoding='utf-8')
            temp.replace(manifest)
            split_paths[split] = manifest.name
    write_json(output/'data.yaml', {'path': output.as_posix(), **split_paths,
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
    parser.add_argument('--local-only', action='store_true',
                        help='Export existing local images and labels without downloading any files')
    parser.add_argument('--source', choices=['hf-mirror', 'official'], default='hf-mirror',
                        help='Download annotations and images from hf-mirror.com by default')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--retries', type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0:
        parser.error('Require positive workers and nonnegative retries')
    names = sorted({'cow' if name == 'cattle' else name for name in args.classes})
    try:
        run(args.output, names, args.annotations_zip, args.workers, args.retries,
            source=args.source, local_only=args.local_only)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Rerun the same command to resume.\n')
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
