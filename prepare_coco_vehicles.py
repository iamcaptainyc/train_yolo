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


def load_label_schema(path=None):
    if path is None:
        return {'names': dict(enumerate(NAMES)), 'mapping': dict(zip(NAMES, NAMES))}
    from merge_yolo_ids import UniqueKeyLoader

    with Path(path).open(encoding='utf-8-sig') as stream:
        config = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(config, dict) or config.keys() - {'names', 'aliases', 'exclude_classes'}:
        raise ValueError('Label config accepts names, aliases and exclude_classes only')
    names = config.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (not isinstance(names, dict) or not names
            or any(type(cid) is not int for cid in names)
            or set(names) != set(range(len(names)))
            or any(not isinstance(name, str) or not name.strip() for name in names.values())
            or len(set(names.values())) != len(names)):
        raise ValueError('Label names must have unique names and continuous integer IDs from 0')
    aliases = config.get('aliases', {})
    excluded = config.get('exclude_classes', [])
    if (not isinstance(aliases, dict)
            or any(key not in NAMES or value not in names.values() for key, value in aliases.items())):
        raise ValueError('Aliases must map exported COCO names to declared target names')
    if (not isinstance(excluded, list) or any(name not in NAMES for name in excluded)
            or len(set(excluded)) != len(excluded) or set(excluded) & aliases.keys()):
        raise ValueError('Excluded classes must be unique COCO names without aliases')
    # Person and vehicle names retain their group semantics in the co-occurrence report.
    if set(excluded) & {'person', *VEHICLE_CLASSES} or any(
            aliases.get(name, name) != name for name in ('person', *VEHICLE_CLASSES)):
        raise ValueError('Person and the five vehicle class names must be retained')
    mapping = {name: None if name in excluded else aliases.get(name, name) for name in NAMES}
    missing = [name for name, target in mapping.items() if target is not None and target not in names.values()]
    if missing:
        raise ValueError(f'Classes need a target name, alias or explicit exclusion: {missing}')
    return {'names': dict(sorted(names.items())), 'mapping': mapping}


def export_state(cache, output, predictor, schema=None):
    schema = schema or load_label_schema()
    state = {'cache': str(cache), 'checkpoint_sha256': predictor.digest, 'pose_version': POSE_VERSION,
             'class_names': list(schema['names'].values()), 'front_through': 45, 'rear_from': 135}
    if schema != load_label_schema():
        state['label_mapping'] = schema['mapping']
    path = output/'.vehicle-state.json'
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        # Label changes rewrite every label; cached angles depend only on the image,
        # official car boxes and predictor, so remain valid across schema changes.
        core = lambda settings: {key: value for key, value in settings.items()
                                 if key not in {'class_names', 'label_mapping'}}
        if core(previous) != core(state):
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


def export_views(cache, output, predictor, batch=32, label_schema=None):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if batch < 1 or cache == output or cache in output.parents or output in cache.parents:
        raise ValueError('Require positive batch and separate cache/output directories')
    state_path = output/'.vehicle-state.json'
    schema = label_schema or load_label_schema()
    state = export_state(cache, output, predictor, schema)
    output.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    marker = output/'INCOMPLETE.txt'
    marker.write_text('Vehicle annotation export is incomplete. Rerun the same command to resume.\n', encoding='utf-8')
    ids = {name: cid for cid, name in schema['names'].items()}
    mapping = schema['mapping']
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
                excluded_images, excluded_classes = 0, Counter()

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
                        name = mapping[name]
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
                    excluded = {available[cid] for _, cid, _ in boxes
                                if available[cid] != 'car' and mapping[available[cid]] is None}
                    if excluded:
                        excluded_images += 1
                        excluded_classes.update(excluded)
                        target = image_dir/filename
                        if target.exists():
                            if not os.path.samefile(target, source):
                                raise ValueError(f'Existing output image is not the source hardlink: {target}')
                            target.unlink()
                        (label_dir/Path(filename).with_suffix('.txt')).unlink(missing_ok=True)
                        continue
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
                                'boxes_by_class': dict(class_boxes), 'excluded_images': excluded_images,
                                'excluded_images_by_class': dict(excluded_classes)}
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
              'nc': len(ids), 'names': schema['names']}
    temporary = output/'data.yaml.part'
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    temporary.replace(output/'data.yaml')
    report = {'splits': stats, 'model': state, 'pose_counts_this_run': dict(counters), 'issues': issues,
              'vehicles': list(VEHICLE_CLASSES), 'car_box_source': 'COCO official bbox; class from continuous CFV azimuth'}
    download_report = cache/'report.json'
    report['source_download_stats'] = json.loads(download_report.read_text(encoding='utf-8')) if download_report.is_file() else {}
    write_json(output/'report.json', report)
    animal_classes = {mapping[name] for name in ANIMALS if mapping[name] is not None}
    write_cooccurrence_report(output, {s: cooccurrence_counts(rows, VEHICLE_CLASSES, animal_classes)
                                      for s, rows in exported.items()},
                              VEHICLE_CLASSES, '按实际导出的动物、人员和五类车辆统计，每张图在每项中计一次。')
    lines = ['# COCO 动物、人员和车辆数据集', '',
             f'标签表 {len(ids)} 类，实际有框 {len(set().union(*(row["boxes_by_class"] for row in stats.values())))} 类，'
             f'图片总数 {sum(row["images"] for row in stats.values()):,}。', '',
             '| 划分 | 图片 |', '| --- | ---: |',
             *(f'| {split} | {row["images"]:,} |' for split, row in stats.items()), '',
             '| ID | 类别 | Train 图片 | Val 图片 | 框合计 |', '| ---: | --- | ---: | ---: | ---: |']
    for cid, name in schema['names'].items():
        counts = [stats[s]['images_by_class'].get(name, 0) for s in SPLITS]
        boxes = sum(stats[s]['boxes_by_class'].get(name, 0) for s in SPLITS)
        lines.append(f'| {cid} | {name} | {counts[0]:,} | {counts[1]:,} | {boxes:,} |')
    lines += ['', '## 标签映射与排除', '', '| COCO 导出类别 | 最终类别 | ID |', '| --- | --- | ---: |']
    for name, target in mapping.items():
        lines.append(f'| {name} | {target if target is not None else "排除含该类有效框的整图"} | '
                     f'{ids[target] if target is not None else "—"} |')
    lines += ['', '| 划分 | 整图排除数 | 原始类别对应图片数 |', '| --- | ---: | --- |',
              *(f'| {split} | {row["excluded_images"]} | {json.dumps(row["excluded_images_by_class"], ensure_ascii=False)} |'
                for split, row in stats.items())]
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
    parser.add_argument('--label-config', type=Path, help='YAML with fixed names/IDs, aliases and explicit exclusions')
    args = parser.parse_args()
    try:
        cache, output = args.cache.resolve(), args.output.resolve()
        if cache == output or cache in output.parents or output in cache.parents:
            raise ValueError('Cache and final output must be separate directories')
        if args.batch < 1 or args.workers < 1 or args.retries < 0:
            raise ValueError('Batch/workers must be positive and retries nonnegative')
        schema = load_label_schema(args.label_config)
        if args.download_pose_weights and not args.pose_weights.is_file():
            fetch_cfv_weights(args.pose_weights, args.retries)
        predictor = CFVPredictor(args.pose_weights, args.device, args.batch)
        export_state(cache, output, predictor, schema)
        output.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output)+'.lock', timeout=0):
            run(cache, None, args.annotations_zip, args.workers, args.retries,
                source=args.source, local_only=args.local_only, add_classes=RAW_CLASSES)
            export_views(cache, output, predictor, args.batch, label_schema=schema)
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Rerun the same command to resume downloaded images and cached car angles.\n')
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, sqlite3.DatabaseError, zipfile.BadZipFile,
            yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
