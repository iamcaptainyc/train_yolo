"""Convert an extracted DoñaNet YOLO dataset to the camping19 class IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from tqdm import tqdm
import yaml

from merge_yolo_ids import SPLITS, images_for, label_for, load_dataset, selected_label, validate_label


POLICY = Path(__file__).resolve().parent / 'configs/camping19.yaml'
TARGETS = {
    'Ave': 'bird',
    'Felis catus': 'cat',
    'Bos taurus': 'cattle',
    'Canis familiaris': 'dog',
    'Herpestes ichneumon': 'small_mammal',
    'Meles meles': 'small_mammal',
    'Oryctolagus cuniculus': 'small_mammal',
    'Dama dama': 'deer',
    'Genetta genetta': 'small_mammal',
    'Equus sp': 'horse',
    'Lepus granatensis': 'small_mammal',
    'Cervus elaphus': 'deer',
    'Vulpes vulpes': 'small_mammal',
    'wild_boar': 'wild_boar',
}
IGNORED = {'Homo sapiens', 'Vehicle', 'Empty'}
OUTSIDE_POLICY = {'Lynx pardinus'}


def write_report(stage, names, groups, mapping, outside, images, boxes, splits, issues):
    covered = sum(any(boxes[split][cid] for split in SPLITS) for cid in groups)
    lines = [
        '# DoñaNet 转 camping19', '',
        f'类别定义 {len(groups)} 个，实际有标注 {covered} 个。图片保留来源的 train/val/test 划分。', '',
        '| 划分 | 原图片 | 保留图片 | 保留框 | 无目标框 | 含范围外动物 | 缺少标签 | 标注无效 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    totals = sum(splits.values(), Counter())
    for split, counts in [*splits.items(), ('合计', totals)]:
        values = [counts[key] for key in ('input', 'kept', 'boxes', 'no_target', 'outside', 'missing', 'invalid')]
        lines.append(f'| {split} | ' + ' | '.join(map(str, values)) + ' |')
    lines += [
        '', '## 输出类别', '',
        '每张含该类框的图片在该类下计一次。多类别图片可计入多行。', '',
        '| ID | 类别 | Train 图片 / 框 | Val 图片 / 框 | Test 图片 / 框 | 图片合计 | 框合计 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for cid, name in groups.items():
        image_totals = [images[split][cid] for split in SPLITS]
        box_totals = [boxes[split][cid] for split in SPLITS]
        cells = ' | '.join(f'{n_images} / {n_boxes}' for n_images, n_boxes in zip(image_totals, box_totals))
        lines.append(f'| {cid} | {name} | {cells} | {sum(image_totals)} | {sum(box_totals)} |')
    lines += [
        '', '## 原类别映射', '',
        '| 原 ID | 原类别 | 处理 |', '| ---: | --- | --- |',
    ]
    for cid, name in sorted(names.items()):
        if cid in mapping:
            action = f'{mapping[cid]} · {groups[mapping[cid]]}'
        else:
            action = '含该类框的整图跳过' if cid in outside else '跳过该类框'
        lines.append(f'| {cid} | {name} | {action} |')
    if issues:
        lines += ['', '## 无效标注', '', *(f'- {issue}' for issue in issues)]
    (stage / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def convert(source, output, copy_images=False):
    dataset = load_dataset(source)
    output = Path(output).expanduser().resolve()
    root = dataset['root']
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Use a new output directory outside the source dataset')

    policy = yaml.safe_load(POLICY.read_text(encoding='utf-8'))
    groups = dict(enumerate(sorted(policy['groups'])))
    ids = {name: cid for cid, name in groups.items()}
    names = dataset['names']
    unknown = sorted(set(names.values()) - TARGETS.keys() - IGNORED - OUTSIDE_POLICY)
    if unknown:
        raise ValueError(f'Unrecognized DoñaNet class names: {unknown}')
    mapping = {cid: ids[TARGETS[name]] for cid, name in names.items() if name in TARGETS}
    selected = set(mapping)
    outside = {cid for cid, name in names.items() if name in OUTSIDE_POLICY}

    output.parent.mkdir(parents=True, exist_ok=True)
    image_counts = {split: Counter() for split in SPLITS}
    box_counts = {split: Counter() for split in SPLITS}
    splits = {split: Counter() for split in SPLITS}
    issues = []
    with TemporaryDirectory(prefix='.donanet-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        for split in SPLITS:
            entry = dataset['config'].get(split)
            if not isinstance(entry, str) or not (split_root := root / entry).is_dir():
                raise ValueError(f'Expected a DoñaNet image directory for {split} in {dataset["yaml"]}')
            image_dir = stage / 'images' / split
            label_dir = stage / 'labels' / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            stats = splits[split]
            for image in tqdm(images_for(dataset, split), desc=f'DoñaNet {split}', unit='img'):
                stats['input'] += 1
                annotation = label_for(image)
                try:
                    raw, original_counts = validate_label(annotation, dataset)
                except FileNotFoundError:
                    stats['missing'] += 1
                    continue
                except ValueError as exc:
                    stats['invalid'] += 1
                    issues.append(str(exc))
                    continue
                if outside.intersection(original_counts):
                    stats['outside'] += 1
                    continue
                if not selected.intersection(original_counts):
                    stats['no_target'] += 1
                    continue
                rewritten, counts = selected_label(raw, selected, mapping)
                relative = image.relative_to(split_root)
                dest_image = image_dir / relative
                dest_label = (label_dir / relative).with_suffix('.txt')
                dest_image.parent.mkdir(parents=True, exist_ok=True)
                dest_label.parent.mkdir(parents=True, exist_ok=True)
                if copy_images:
                    shutil.copy2(image, dest_image)
                else:
                    try:
                        os.link(image, dest_image)
                    except OSError as exc:
                        raise OSError(f'Cannot hardlink {image}: {exc}. Use --copy-images if needed.') from exc
                dest_label.write_bytes(rewritten)
                stats['kept'] += 1
                stats['boxes'] += sum(counts.values())
                image_counts[split].update(counts.keys())
                box_counts[split].update(counts)

        if not any(stats['kept'] for stats in splits.values()):
            raise ValueError('No usable camping19 images found; inspect the source labels')
        config = {'path': output.as_posix(), **{split: f'images/{split}' for split in SPLITS},
                  'nc': len(groups), 'names': groups}
        (stage / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        write_report(stage, names, groups, mapping, outside, image_counts, box_counts, splits, issues)
        stage.rename(output)

    print(f'Converted {sum(stats["kept"] for stats in splits.values())} images. Report: {output / "REPORT.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='DoñaNet dataset directory or data.yaml')
    parser.add_argument('--output', required=True, type=Path, help='New output dataset directory')
    parser.add_argument('--copy-images', action='store_true', help='Copy images instead of hardlinking')
    args = parser.parse_args()
    try:
        convert(**vars(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
