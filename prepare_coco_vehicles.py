"""Download/reuse COCO and classify car views using continuous PyTorch azimuths."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import zipfile

from filelock import FileLock
from PIL import Image
from tqdm import tqdm
import yaml

from download_coco import ANIMALS, SPLITS, cooccurrence_counts, run, select_boxes, write_cooccurrence_report, write_json
from vehicle_pose import CFVPredictor, VEHICLE_CLASSES, angle_class, fetch_cfv_weights


RAW_CLASSES = (*ANIMALS, 'person', 'car', 'truck', 'bus')
NAMES = sorted({*ANIMALS, 'person', *VEHICLE_CLASSES})
POSE_VERSION = 'cfv-convnext-small-rgb-linear224-tensor01-v1'


def export_state(cache, output, predictor):
    state = {'cache': str(cache), 'checkpoint_sha256': predictor.digest, 'pose_version': POSE_VERSION,
             'class_names': NAMES, 'front_through': 45, 'rear_from': 135}
    path = output/'.vehicle-state.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != state:
            raise ValueError('Vehicle export settings changed; use a new output directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Use an empty output directory or resume a vehicle export: {output}')
    return state


def image_signature(image, cars):
    stat = image.stat()
    fields = [str(image), stat.st_size, stat.st_mtime_ns,
              [(ann['id'], box) for ann, _, box in cars]]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def car_crops(image, cars, expected_size):
    with Image.open(image) as original:
        if original.size != expected_size:
            raise ValueError(f'Image size {original.size} disagrees with COCO {expected_size}')
        pixels = original.convert('RGB')
    width, height = pixels.size
    crops = []
    for _, _, (x, y, w, h) in cars:
        left = max(0, math.floor((x-w/2)*width))
        top = max(0, math.floor((y-h/2)*height))
        right = min(width, math.ceil((x+w/2)*width))
        bottom = min(height, math.ceil((y+h/2)*height))
        if right <= left or bottom <= top:
            raise ValueError('Car crop is empty')
        crops.append(pixels.crop((left, top, right, bottom)))
    return crops


def export_views(cache, output, predictor, batch=32):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if batch < 1 or cache == output or cache in output.parents or output in cache.parents:
        raise ValueError('Require positive batch and separate cache/output directories')
    state_path = output/'.vehicle-state.json'
    state = export_state(cache, output, predictor)
    output.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Vehicle annotation export is incomplete. Rerun the same command to resume.\n', encoding='utf-8')
    ids = {name: cid for cid, name in enumerate(NAMES)}
    stats, issues = {}, []
    exported = {split: [] for split in SPLITS}
    image_lists = {split: [] for split in SPLITS}
    counters = Counter()
    db = sqlite3.connect(output/'.vehicle-angles.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(split TEXT,file TEXT,signature TEXT,angles TEXT,PRIMARY KEY(split,file))')
    angle_path = output/'car_angles.csv.part'
    try:
        with angle_path.open('w', encoding='utf-8', newline='') as angle_stream:
            writer = csv.writer(angle_stream)
            writer.writerow(['split', 'image', 'annotation_id', 'angle_degrees', 'class'])
            for split in SPLITS:
                path = cache/'_metadata'/f'instances_{split}2017_subset.json'
                data = json.loads(path.read_text(encoding='utf-8'))
                available = sorted(category['name'] for category in data['categories'] if category['name'] in RAW_CLASSES)
                images, selected = select_boxes(data, available)
                image_dir, label_dir = output/'images'/split, output/'labels'/split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                class_images, class_boxes = Counter(), Counter()

                def emit(record, angles):
                    filename, source, boxes, cars, signature = record
                    angles = list(angles)
                    if len(angles) != len(cars) or any(not math.isfinite(a) or not 0 <= a <= 360 for a in angles):
                        raise ValueError(f'Invalid continuous predictions for {source}')
                    rows, names, row_counts = [], set(), Counter()
                    car_index = 0
                    for ann, cid, box in boxes:
                        name = available[cid]
                        if name == 'car':
                            angle = angles[car_index]
                            car_index += 1
                            name = angle_class(angle)
                            writer.writerow([split, filename, ann['id'], f'{angle:.8f}', name])
                        rows.append(f'{ids[name]} ' + ' '.join(format(value, '.12g') for value in box) + '\n')
                        names.add(name)
                        row_counts[name] += 1
                    target = image_dir/filename
                    if target.exists():
                        if not os.path.samefile(target, source):
                            raise ValueError(f'Existing output image is not the source hardlink: {target}')
                    else:
                        os.link(source, target)
                    label = label_dir/Path(filename).with_suffix('.txt')
                    temporary = label.with_suffix('.txt.part')
                    temporary.write_text(''.join(rows), encoding='utf-8')
                    temporary.replace(label)
                    class_boxes.update(row_counts)
                    class_images.update(names)
                    exported[split].append(names)
                    image_lists[split].append(f'./images/{split}/{filename}\n')

                pending, crops = [], []

                def flush():
                    if not pending:
                        return
                    angles = predictor.predict(crops)
                    if len(angles) != len(crops):
                        raise ValueError('Pose predictor returned the wrong batch size')
                    position = 0
                    for record in pending:
                        car_angles = angles[position:position+len(record[3])]
                        position += len(record[3])
                        if any(not math.isfinite(a) or not 0 <= a <= 360 for a in car_angles):
                            raise ValueError('Pose predictor returned invalid angles')
                        with db:
                            db.execute('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?)',
                                       (split, record[0], record[4], json.dumps(car_angles)))
                        emit(record, car_angles)
                        counters['predicted_car_boxes'] += len(car_angles)
                    pending.clear()
                    crops.clear()

                for image_id, boxes in tqdm(selected.items(), desc=f'Car angles and YOLO {split}', unit='img'):
                    filename = images[image_id]['file_name']
                    if not re.fullmatch(r'\d{12}\.jpg', filename):
                        raise ValueError(f'Unexpected COCO filename: {filename}')
                    source = cache/'images'/split/filename
                    cars = [box for box in boxes if available[box[1]] == 'car']
                    try:
                        if source.stat().st_size <= 0:
                            raise ValueError('Empty image')
                        signature = image_signature(source, cars) if cars else None
                    except (OSError, ValueError) as exc:
                        issues.append({'split': split, 'image': filename, 'error': str(exc)})
                        continue
                    record = (filename, source, boxes, cars, signature)
                    if not cars:
                        emit(record, [])
                        continue
                    cached = None
                    try:
                        previous = db.execute('SELECT signature,angles FROM predictions WHERE split=? AND file=?',
                                              (split, filename)).fetchone()
                        if previous and previous[0] == signature:
                            cached = json.loads(previous[1])
                    except ValueError:
                        pass
                    if cached is not None:
                        emit(record, cached)
                        counters['cached_car_boxes'] += len(cars)
                        continue
                    try:
                        pixels = car_crops(source, cars, (images[image_id]['width'], images[image_id]['height']))
                    except (OSError, ValueError, SyntaxError) as exc:
                        issues.append({'split': split, 'image': filename, 'error': str(exc)})
                        (image_dir/filename).unlink(missing_ok=True)
                        (label_dir/Path(filename).with_suffix('.txt')).unlink(missing_ok=True)
                        continue
                    pending.append(record)
                    crops.extend(pixels)
                    if len(crops) >= batch:
                        flush()
                flush()
                stats[split] = {'images': len(exported[split]), 'images_by_class': dict(class_images),
                                'boxes_by_class': dict(class_boxes)}
                del data, images, selected
    finally:
        db.close()

    if any(not row['images'] for row in stats.values()):
        raise ValueError('Both train and val need completed images; see INCOMPLETE.txt and rerun')
    angle_path.replace(output/'car_angles.csv')
    for split, rows in image_lists.items():
        manifest = output/f'{split}.txt'
        temporary = manifest.with_suffix('.txt.part')
        temporary.write_text(''.join(rows), encoding='utf-8')
        temporary.replace(manifest)
    config = {'path': output.as_posix(), 'train': 'train.txt', 'val': 'val.txt',
              'nc': len(NAMES), 'names': dict(enumerate(NAMES))}
    temporary = output/'data.yaml.part'
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    temporary.replace(output/'data.yaml')
    report = {'splits': stats, 'model': state, 'pose_counts_this_run': dict(counters), 'issues': issues,
              'vehicles': list(VEHICLE_CLASSES), 'car_box_source': 'COCO official bbox; class from continuous CFV azimuth'}
    download_report = cache/'report.json'
    report['source_download_stats'] = json.loads(download_report.read_text(encoding='utf-8')) if download_report.is_file() else {}
    write_json(output/'report.json', report)
    write_cooccurrence_report(output, {s: cooccurrence_counts(rows, VEHICLE_CLASSES) for s, rows in exported.items()},
                              VEHICLE_CLASSES, '按实际导出的动物、人员和五类车辆统计，每张图在每项中计一次。')
    lines = ['# COCO 动物、人员和车辆数据集', '',
             f'类别数 {len(NAMES)}，图片总数 {sum(row["images"] for row in stats.values()):,}。', '',
             '| 划分 | 图片 |', '| --- | ---: |',
             *(f'| {split} | {row["images"]:,} |' for split, row in stats.items()), '',
             '| ID | 类别 | Train 图片 | Val 图片 | 框合计 |', '| ---: | --- | ---: | ---: | ---: |']
    for name, cid in ids.items():
        counts = [stats[s]['images_by_class'].get(name, 0) for s in SPLITS]
        boxes = sum(stats[s]['boxes_by_class'].get(name, 0) for s in SPLITS)
        lines.append(f'| {cid} | {name} | {counts[0]:,} | {counts[1]:,} | {boxes:,} |')
    lines += ['', 'car_head 的绝对角度不超过 45°，car_tail 至少 135°，其余为 car_side。',
              'car_angles.csv 保存每个车框的连续角度。truck、bus、人和动物使用官方单体框。',
              f'本次推理 {counters["predicted_car_boxes"]:,} 个车框，复用 {counters["cached_car_boxes"]:,} 个。',
              f'跳过异常图片 {len(issues)} 张，详情见 report.json。', '',
              '[同框统计](COOCCURRENCE.md)']
    missing = sum(row.get('missing_images', 0) for row in report['source_download_stats'].values())
    if missing:
        lines += ['', f'COCO 缓存仍有 {missing:,} 张图片未下载，本次统计覆盖已经完成的图片。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    marker.unlink()
    print(f'Finished: {output / "data.yaml"}\nReport: {output / "REPORT.md"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True, help='Existing or new download_coco.py output')
    parser.add_argument('--output', type=Path, required=True, help='Final YOLO output, separate from cache; same filesystem')
    parser.add_argument('--pose-weights', type=Path, default=Path('weights/cfv/convnext-multi-weak.pth'))
    parser.add_argument('--download-pose-weights', action='store_true', help='Fetch the official CFV checkpoint if missing')
    parser.add_argument('--device', default='cpu', help='cpu, cuda:0, or GPU index such as 0')
    parser.add_argument('--batch', type=int, default=32, help='Car crop inference batch size')
    parser.add_argument('--workers', type=int, default=16, help='Image download workers')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--source', choices=('hf-mirror', 'official'), default='hf-mirror')
    parser.add_argument('--annotations-zip', type=Path)
    parser.add_argument('--local-only', action='store_true', help='Use downloaded COCO images and local annotation ZIP')
    args = parser.parse_args()
    try:
        cache, output = args.cache.resolve(), args.output.resolve()
        if cache == output or cache in output.parents or output in cache.parents:
            raise ValueError('Cache and final output must be separate directories')
        if args.batch < 1 or args.workers < 1 or args.retries < 0:
            raise ValueError('Batch/workers must be positive and retries nonnegative')
        if args.download_pose_weights and not args.pose_weights.is_file():
            fetch_cfv_weights(args.pose_weights, args.retries)
        predictor = CFVPredictor(args.pose_weights, args.device, args.batch)
        export_state(cache, output, predictor)
        output.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output)+'.lock', timeout=0):
            run(cache, None, args.annotations_zip, args.workers, args.retries,
                source=args.source, local_only=args.local_only, add_classes=RAW_CLASSES)
            export_views(cache, output, predictor, args.batch)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Rerun the same command to resume downloaded images and cached car angles.\n')
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, sqlite3.DatabaseError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
