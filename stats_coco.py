"""Count people, animals and vehicles in local COCO images using official annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

from tqdm import tqdm

from download_coco import ANIMALS, ARCHIVE, SPLITS, VEHICLES, cooccurrence_counts, select_boxes, write_cooccurrence_report


def analyze(dataset, archive=None, vehicle_classes=('car',)):
    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f'Dataset directory does not exist: {dataset}')
    archive = Path(archive).expanduser() if archive else dataset/'_metadata'/ARCHIVE
    wanted = {*ANIMALS, 'person', *vehicle_classes}
    by_split = {}
    with zipfile.ZipFile(archive) as source:
        for split in SPLITS:
            with source.open(f'annotations/instances_{split}2017.json') as stream:
                data = json.load(stream)
            names = sorted(category['name'] for category in data['categories'] if category['name'] in wanted)
            images, selected = select_boxes(data, names)
            class_sets = []
            for image_id, image in tqdm(images.items(), desc=f'Count local COCO {split}', unit='img'):
                filename = Path(image['file_name'])
                if filename.is_absolute() or len(filename.parts) != 1:
                    raise ValueError(f'Unexpected COCO file name: {filename}')
                path = dataset/'images'/split/filename
                if path.is_file() and path.stat().st_size:
                    class_sets.append({names[cid] for _, cid, _ in selected.get(image_id, [])})
            by_split[split] = cooccurrence_counts(class_sets, vehicle_classes)
            del data, selected, images
    scope = '按本地非空图片和完整 COCO 官方标注统计，使用有效非 crowd 单体框。人或动物有标注即可计入，与旧 YOLO 标签是否补齐无关。'
    return write_cooccurrence_report(dataset, by_split, vehicle_classes, scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Existing download_coco.py output directory')
    parser.add_argument('--annotations-zip', type=Path, help='Local full COCO 2017 annotation archive')
    parser.add_argument('--vehicle-classes', nargs='+', choices=VEHICLES, default=['car'],
                        help='COCO classes counted as vehicles (default: car)')
    args = parser.parse_args()
    try:
        result = analyze(args.dataset, args.annotations_zip, tuple(dict.fromkeys(args.vehicle_classes)))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Error: {exc}\n')
    totals = result['total']
    print(f'Images: {totals["images"]}; person/car: {totals["person_vehicle"]}; '
          f'person/animal: {totals["person_animal"]}; animal/car: {totals["animal_vehicle"]}; '
          f'all three: {totals["all_three"]}')


if __name__ == '__main__':
    sys.exit(main())
