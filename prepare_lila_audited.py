"""Merge the five downloaded LILA datasets with taxonomy and full coordinate provenance audits."""

from __future__ import annotations

import argparse
import hashlib
from contextlib import closing, nullcontext
from functools import partial
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw
from tqdm import tqdm

from lila_audit import TAXONOMY, inspect_sources, sha256, taxonomy_mapping
from prepare_lila import DEFAULT_DATASETS, prepare, split_fractions, split_overlap
from lila_selection import items
from lila_parallel import bounded_map
from lila_state import managed_run, fingerprint, signature, record_key


def cached_audit(record, output, names, mapping, allow_clipping=False, state=None):
    if state is None:
        return audit_one(record, output, names, mapping, allow_clipping)
    if getattr(state, 'compact', False):
        try:
            state.refresh_source(record)
            if getattr(state, 'resumed_written', False):
                from lila_resume_output import verify_existing_image

                verify_existing_image(record, output_paths(output, record)[0], state)
                reuse = True
            else:
                reuse = state._passed(record, 'audit')
            row = audit_one(
                record,
                output,
                names,
                mapping,
                allow_clipping,
                verify_image_hash=not reuse,
            )
            state.stats['audit_hash_skipped' if reuse else 'audit_checked'] += 1
            state.clear_error('audit', record)
            state.audit_done(record)
            return row
        except (OSError, ValueError) as exc:
            state.error('audit', record, exc)
            raise
    image, label = output_paths(output, record)
    try:
        state.refresh_source(record)
        stamp = signature(
            [
                fingerprint(record["source"]),
                fingerprint(image),
                fingerprint(label),
                record["digest"],
                record["source_annotations"],
                names,
                mapping,
                allow_clipping,
                record["split"],
                record["image"],
                "independent-audit-v1",
            ]
        )
        row = state.get("audit", record_key(record), stamp)
        if row is None:
            row = audit_one(record, output, names, mapping, allow_clipping)
            after = signature(
                [
                    fingerprint(record["source"]),
                    fingerprint(image),
                    fingerprint(label),
                    record["digest"],
                    record["source_annotations"],
                    names,
                    mapping,
                    allow_clipping,
                    record["split"],
                    record["image"],
                    "independent-audit-v1",
                ]
            )
            if after != stamp:
                raise ValueError("File changed during independent audit")
            state.put("audit", record_key(record), stamp, row)
        state.refresh_output_stamp(record)
        state.clear_error("audit", record)
        return row
    except (OSError, ValueError) as exc:
        state.error("audit", record, exc)
        raise


def output_paths(output, record):
    key = hashlib.sha256(
        (record['dataset'] + '/' + record['name']).encode()
    ).hexdigest()
    relative = Path(record['dataset']) / (key + record['source'].suffix.lower())
    return output / 'images' / record['split'] / relative, output / 'labels' / record[
        'split'
    ] / relative.with_suffix('.txt')


def decode_label(path, width, height, names):
    result = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f'Malformed YOLO row {path}:{number}')
        category = int(tokens[0])
        if not 0 <= category < len(names):
            raise ValueError(f'Invalid YOLO class {path}:{number}')
        cx, cy, w, h = map(float, tokens[1:])
        if (
            not all(math.isfinite(x) and 0 <= x <= 1 for x in (cx, cy, w, h))
            or min(w, h) <= 0
        ):
            raise ValueError(
                f'Invalid YOLO coordinates {path}:{number}: {line!r}; normalized center must be in [0,1] and width/height in (0,1]'
            )
        result.append(
            {
                'line': number,
                'class_id': category,
                'class_name': names[category],
                'yolo': [cx, cy, w, h],
                'xyxy': [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ],
            }
        )
    return result


def audit_one(
    record, output, names, mapping, allow_clipping=False, verify_image_hash=True
):
    """Independent reverse decoding of the written file, never using converter label tuples."""
    image_path, label_path = output_paths(output, record)
    if verify_image_hash and sha256(image_path) != record['digest']:
        raise ValueError(f'Output image bytes differ from input: {image_path}')
    if record.get('_resume_storage'):
        width, height = record['actual_dimensions']
    else:
        with Image.open(image_path) as image:
            width, height = image.size
    decoded = decode_label(label_path, width, height, names)
    class_ids = {name: i for i, name in enumerate(names)}
    expected = defaultdict(list)
    for ann in record['source_annotations']:
        target = mapping[record['dataset'] + ':' + str(ann['category_id'])]
        x, y, w, h = ann['bbox']
        original = [x, y, x + w, y + h]
        clipped = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
        if original != clipped and not allow_clipping:
            raise ValueError(
                f'Original bbox needs edge clipping: {record["dataset"]}/{record["name"]}; '
                'review it or explicitly use --allow-one-pixel-clipping'
            )
        expected[(class_ids[target], tuple(clipped))].append(ann)
    if len(decoded) != len(expected):
        raise ValueError(
            f'Written label count differs from distinct original boxes: {label_path}'
        )
    tolerance = max(width, height) * 1e-8 + 1e-7
    max_error = 0.0
    max_clipping = 0.0
    for box in decoded:
        matches = [
            key
            for key in expected
            if key[0] == box['class_id']
            and max(abs(a - b) for a, b in zip(key[1], box['xyxy'])) <= tolerance
        ]
        if not matches:
            raise ValueError(
                f'Written YOLO class/coordinates do not match original: {label_path}:{box["line"]}'
            )
        key = min(
            matches, key=lambda k: max(abs(a - b) for a, b in zip(k[1], box['xyxy']))
        )
        originals = expected.pop(key)
        error = max(abs(a - b) for a, b in zip(key[1], box['xyxy']))
        max_error = max(max_error, error)
        for ann in originals:
            x, y, w, h = ann['bbox']
            max_clipping = max(
                max_clipping,
                max(abs(a - b) for a, b in zip((x, y, x + w, y + h), key[1])),
            )
        box.update(
            {
                'source_selected_annotation_ids': [a['id'] for a in originals],
                'source_category_ids': [a['category_id'] for a in originals],
                'source_coco_xywh': [a['bbox'] for a in originals],
                'expected_clipped_xyxy': list(key[1]),
                'max_error_pixels': error,
            }
        )
    if (record['role'] == 'negative') != (not decoded):
        raise ValueError(f'Positive/negative output mismatch: {label_path}')
    return {
        'dataset': record['dataset'],
        'source_file_name': record['name'],
        'selected_image_id': record['image']['id'],
        'original_image_id': record['image']['source_image_id'],
        'split': record['split'],
        'width': width,
        'height': height,
        'role': record['role'],
        'source_sha256': record['digest'],
        'output_image': image_path.relative_to(output).as_posix(),
        'output_label': label_path.relative_to(output).as_posix(),
        'label_sha256': sha256(label_path),
        'duplicate_box_records_removed': len(record['source_annotations'])
        - len(decoded),
        'max_error_pixels': max_error,
        'max_clipping_pixels': max_clipping,
        'tolerance_pixels': tolerance,
        'boxes': decoded,
    }


def preview(record, output, names, source_names, destination):
    image_path, label_path = output_paths(output, record)
    with Image.open(record['source']) as image:
        left = image.convert('RGB')
    with Image.open(image_path) as image:
        right = image.convert('RGB')
    width, height = left.size
    scale = min(1000 / width, 750 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, right = left.resize(size), right.resize(size)
    for ann in record['source_annotations']:
        x, y, w, h = ann['bbox']
        draw = ImageDraw.Draw(left)
        draw.rectangle(
            tuple(v * scale for v in (x, y, x + w, y + h)), outline='red', width=2
        )
        draw.text(
            (max(0, x * scale), max(0, y * scale - 12)),
            source_names[str(ann['category_id'])],
            fill='red',
            stroke_width=1,
            stroke_fill='white',
        )
    for box in decode_label(label_path, width, height, names):
        x1, y1, x2, y2 = [v * scale for v in box['xyxy']]
        draw = ImageDraw.Draw(right)
        draw.rectangle((x1, y1, x2, y2), outline='lime', width=2)
        draw.text(
            (max(0, x1), max(0, y1 - 12)),
            box['class_name'],
            fill='lime',
            stroke_width=1,
            stroke_fill='black',
        )
    canvas = Image.new('RGB', (2 * size[0] + 12, size[1] + 55), 'white')
    canvas.paste(left, (0, 55))
    canvas.paste(right, (size[0] + 12, 55))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), 'COCO original (red)', fill='black')
    draw.text((size[0] + 17, 5), 'YOLO saved (green)', fill='black')
    draw.text(
        (5, 23),
        record['dataset'] + ' / ' + record['split'] + ' / ' + record['role'],
        fill='black',
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
        "",
    ]


def write_markdown(output, inventories, details, report, audit, previews, retained):
    splits = tuple(report['stats'])
    lines = [
        '# LILA 五数据集合并与框对应审计报告',
        '',
        '生成时间（UTC）：' + datetime.now(timezone.utc).isoformat(),
        '',
        '**状态：转换审计通过。此状态证明已核验的输入→输出对应关系，不等于原始人工框、物种标签和空图标签绝对正确。**',
        '',
        '## 统计口径与处理规则',
        '',
        '分类来源：[LILA 官方映射说明](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/)。本次实际分类表及 SHA-256 见 audit_sources.json。',
        '',
        '- 原始全集来自本机下载的原始 metadata/bbox 文件，绝非页面概述或历史计数。两个文件的类别 ID 空间分别统计。',
        '- 原始“标注数”包含图片级分类；“框记录”只统计 bbox 长度为 4 的记录，可能仍有无效框，不能视为有效训练框。',
        '- 每类图片数按 image_id 去重；一张图含多个类别时会重复计入各类，类别图片数之和不等于总图片数。',
        '- 所有下载筛选图片都已与原始图片 ID、文件名、分类命名空间和完整框集合核对，负样本必须有明确且无矛盾的 empty 标签。',
        '- 使用固定的 LILA 官方 iNaturalist 映射快照。同一科学分类合并；出现父类标签时，子类整体上卷到已标注父类，输出类别无祖先/后代并存。',
        '- animalia、mammalia、superorder 等过宽类别默认排除；含这些类别的整图排除，避免部分动物成为无标签背景。',
        '- 官方映射可能把复合类别近似到更大分类，例如 small antelope；这只能保证所采用分类树的一致性，不能证明生物分类绝对准确。需要细粒度任务时应人工复核或排除该来源类。',
        '- 使用 rank:scientific_name 命名，避免把粗分类伪装成一个具体物种。分组方式沿用相机→序列→保守整数据集留训练规则，不使用官方原 split。',
        '- 原图内容不缩放、不旋转、不重编码；默认硬链接，跨盘则复制。预览缩放仅用于展示。硬链接共享文件内容，转换后勿修改原图。',
        '- 字节相同且标签一致的图片保留为独立记录并放在同一划分，报告另列唯一 SHA-256 数；精确重复框会去重并计数。默认拒绝任何边缘裁剪。若显式允许 1 像素裁剪，报告单独列出，误差核验以裁剪后的框为准。',
        '',
        '## 五数据集总览',
        '',
    ]
    rows = []
    for dataset, inv in inventories.items():
        count = retained[dataset]
        rows.append(
            [
                dataset,
                inv['metadata']['images'],
                inv['bbox']['bbox_records'],
                inv['selected_positive_images'],
                inv['selected_negative_images'],
                count['images'],
                count['negative'],
                count['boxes'],
                inv['selected_images'] - count['images'],
            ]
        )
    rows.append(
        [
            '合计',
            sum(i['metadata']['images'] for i in inventories.values()),
            sum(i['bbox']['bbox_records'] for i in inventories.values()),
            sum(i['selected_positive_images'] for i in inventories.values()),
            sum(i['selected_negative_images'] for i in inventories.values()),
            sum(c['images'] for c in retained.values()),
            sum(c['negative'] for c in retained.values()),
            sum(c['boxes'] for c in retained.values()),
            sum(i['selected_images'] for i in inventories.values())
            - sum(c['images'] for c in retained.values()),
        ]
    )
    lines += table(
        [
            '数据集',
            '原始图片',
            '原始框记录',
            '入选正图',
            '入选负图',
            '最终图片',
            '最终负图',
            '最终框',
            '排除图片',
        ],
        rows,
    )
    details_by_key = {d['key']: d for d in details}
    for dataset, inv in inventories.items():
        lines += [
            '## ' + dataset,
            '',
            '### 原始 metadata 全部类别',
            '',
            '文件：`' + inv['metadata']['file'] + '`',
            '',
        ]
        dataset_splits = report['dataset_split_stats'][dataset]
        total_images = sum(v['images'] for v in dataset_splits.values())
        lines += table(
            ['划分', '最终图片', '数据集内比例', '负样本', '框数'],
            [
                [
                    split,
                    value['images'],
                    f"{value['images']/total_images:.2%}" if total_images else '0.00%',
                    value['negative_images'],
                    value['boxes'],
                ]
                for split, value in dataset_splits.items()
            ],
        )
        lines += table(
            ['原 ID', '原名称', '图片级/物体标注记录', '关联图片', '框记录'],
            [
                [c['id'], c['name'], c['annotations'], c['images'], c['bbox_records']]
                for c in inv['metadata']['categories']
            ],
        )
        lines += [
            '### 原始 bbox 类别、下载子集与最终映射',
            '',
            '文件：`' + inv['bbox']['file'] + '`',
            '',
        ]
        selected = {str(c['id']): c for c in inv['selected_categories']}
        rows = []
        for c in inv['bbox']['categories']:
            key = dataset + ':' + str(c['id'])
            d = details_by_key.get(key, {})
            rows.append(
                [
                    c['id'],
                    c['name'],
                    c['bbox_records'],
                    c['bbox_images'],
                    selected.get(str(c['id']), {}).get('boxes', 0),
                    retained[dataset]['source_boxes'][str(c['id'])],
                    d.get('target') or '排除/无入选框',
                    d.get('reason', '未进入筛选清单'),
                ]
            )
        lines += table(
            [
                '原 ID',
                '原名称',
                '原始框记录',
                '原始框图片',
                '筛选框记录',
                '保留原框记录*',
                '合并后分类',
                '原因',
            ],
            rows,
        )
        lines += [
            '*保留原框记录在相同框去重前统计；最终 YOLO 框数见总览。整图排除也会损失同图其他类别的框。',
            '',
        ]
    lines += ['## 合并后的连续 YOLO 类别', '']
    rows = []
    for class_id, name in enumerate(report['classes']):
        contributors = [
            d['key'] + ' ' + d['source_name']
            for d in details
            if d.get('target') == name
            and retained[d['dataset']]['source_boxes'][str(d['source_id'])]
        ]
        rows.append(
            [
                class_id,
                name,
                sum(report['stats'][s]['boxes'][name] for s in splits),
                *[report['stats'][split]['boxes'][name] for split in splits],
                sum(report['class_images'][split][name] for split in splits),
                *[report['class_images'][split][name] for split in splits],
                '; '.join(contributors),
            ]
        )
    lines += table(
        [
            'YOLO ID',
            '类别',
            '总框数',
            *[split + ' 框' for split in splits],
            '总图片',
            *[split + ' 图片' for split in splits],
            '原始类别来源',
        ],
        rows,
    )
    lines += [
        '## 分组与泄漏检查',
        '',
        '相机、序列、连通分量和 SHA-256 的跨集交集已逐项检查。只保证已知组和字节相同文件的隔离，不保证未知相机或不同编码近似图的隔离。',
        '',
    ]
    lines += table(
        [
            '划分',
            '目标比例',
            '实际比例',
            '图片',
            '负样本',
            '连通组',
            '框数',
            '缺席类别',
        ],
        [
            [
                s,
                f"{report['requested_split_fractions'][s]:.2%}",
                f"{report['actual_split_fractions'][s]:.2%}",
                report['stats'][s]['images'],
                report['stats'][s]['negative_images'],
                report['split_groups'][s],
                sum(report['stats'][s]['boxes'].values()),
                ', '.join(report['missing_classes'][s]) or '无',
            ]
            for s in splits
        ],
    )
    lines += table(
        ['划分对', '相机交集', '序列交集', '组交集', '连通分量交集', 'SHA-256 交集'],
        [
            [
                pair,
                values['camera'],
                values['sequence'],
                values['group'],
                values['component'],
                values['digest'],
            ]
            for pair, values in audit['cross_split_overlap'].items()
        ],
    )
    lines += [
        '缺乏可靠分组信息而整体留在训练集的数据集：`'
        + str(report['train_only_datasets_missing_group_info'])
        + '`',
        '',
        '## 全量框与文件对应证明',
        '',
        '转换：`cx=(x+w/2)/W, cy=(y+h/2)/H, bw=w/W, bh=h/H`；反算：`x1=(cx-bw/2)W, y1=(cy-bh/2)H, x2=(cx+bw/2)W, y2=(cy+bh/2)H`。',
        '',
        '审计独立读取磁盘上的 YOLO 文本与输出图片，逐框匹配原始类别和裁剪后的像素边界，不能用内存中的期望标签代替实际输出。8 位小数引入的像素容差为 `max(W,H)*1e-8+1e-7`。',
        '',
    ]
    lines += table(
        ['检查', '结果'],
        [[k, json.dumps(v, ensure_ascii=False)] for k, v in audit.items()],
    )
    lines += [
        '逐图片、逐框记录： [box_audit.jsonl](box_audit.jsonl)；整图排除清单：[excluded_images.jsonl](excluded_images.jsonl)；原始统计与来源指纹：[audit_sources.json](audit_sources.json)；类别层级决策：[taxonomy_decisions.json](taxonomy_decisions.json)。',
        '',
        '## 框图对应证据',
        '',
        '左图红框直接来自已经与原文件核对的 COCO xywh；右图绿框从实际保存的 YOLO txt 独立反算。以下样本覆盖每个保留的原始类别、各数据集可用的 train/val/test、负样本以及每数据集最多框/最小框样本。图片是抽样展示，数值核验覆盖全量。',
        '',
    ]
    for item in previews:
        lines += [
            '### ' + item['dataset'] + ' / ' + item['source_file_name'],
            '',
            '`'
            + item['split']
            + '`；原类别：'
            + (', '.join(item['source_classes']) or '明确空图')
            + '；'
            + item['reasons'],
            '',
            '[逐图标签]('
            + quote(item['output_label'], safe='/')
            + ')；`source_image_id='
            + str(item['original_image_id'])
            + '`',
            '',
            '![](' + quote(item['preview'], safe='/') + ')',
            '',
        ]
    lines += [
        '## 可复核指纹',
        '',
        '配置、原始文件、筛选清单、分类表和脚本的 SHA-256 见 audit_sources.json；每张输出图的 SHA-256 和标签指纹见逐图清单。',
        '',
        '这些检查能发现错图、错类别 ID、坐标单位/公式错误、漏框、损坏图片和不一致标签；无法从数学上证明人工框画对动物、物种命名正确、动物标注完整或空图确实没有动物。',
        '',
    ]
    (output / 'AUDIT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def make_auditor(
    inventories,
    details,
    taxonomy_file,
    mapping,
    per_class=1,
    allow_clipping=False,
    workers=8,
):
    def audit(records, legacy_mapping, report, output):
        if report["plan_only"]:
            raise ValueError("Audited merge requires actual image verification")
        provenance = {
            d: {k: v for k, v in inv.items() if k != "raw_annotation_ids"}
            for d, inv in inventories.items()
        }
        for dataset, inv in inventories.items():
            source = (
                Path(report["source"]) / dataset / "selection/selected_annotations.json"
            )
            if sha256(source) != inv["selection_sha256"]:
                raise ValueError(f"Selection changed during conversion: {source}")
            for item in inv["source_files"]:
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError(
                        f'Original annotation changed during conversion: {item["path"]}'
                    )
            for kind in ("metadata", "bbox"):
                if sha256(inv[kind]["file"]) != inv[kind]["sha256"]:
                    raise ValueError(
                        f'Original decoded JSON changed during conversion: {inv[kind]["file"]}'
                    )
        (output / "audit_sources.json").write_text(
            json.dumps(
                {
                    "datasets": provenance,
                    "taxonomy_file": str(taxonomy_file),
                    "taxonomy_sha256": sha256(taxonomy_file),
                    "taxonomy_reference": (
                        json.loads(
                            TAXONOMY.with_suffix(".source.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if Path(taxonomy_file).resolve() == TAXONOMY.resolve()
                        else {"custom_taxonomy": True}
                    ),
                    "scripts": {
                        p: sha256(Path(__file__).parent / p)
                        for p in (
                            "prepare_lila_audited.py",
                            "prepare_lila.py",
                            "lila_audit.py",
                            "lila_parallel.py",
                            "lila_state.py",
                            "lila_compact.py",
                            "lila_resume_output.py",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "taxonomy_decisions.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        splits = tuple(report["stats"])
        class_images = {s: Counter() for s in splits}
        retained = {
            d: {"images": 0, "negative": 0, "boxes": 0, "source_boxes": Counter()}
            for d in inventories
        }
        overlaps = split_overlap(records)
        if any(n for pair in overlaps.values() for n in pair.values()):
            raise ValueError(f"Cross-split grouping overlap: {overlaps}")
        choices = defaultdict(set)
        chosen_per_key = Counter()
        # Seeded ordering is stable across platform; retain a few samples from every original class.
        ordered = sorted(
            records,
            key=lambda r: hashlib.sha256(
                (str(report["seed"]) + r["dataset"] + "/" + r["name"]).encode()
            ).hexdigest(),
        )
        for r in ordered:
            key = (r["dataset"], r["name"])
            sample_keys = [("split", r["dataset"], r["split"], r["role"])]
            sample_keys += [
                ("category", r["dataset"], str(a["category_id"]))
                for a in r["source_annotations"]
            ]
            for sample in set(sample_keys):
                if chosen_per_key[sample] < per_class:
                    choices[key].add("/".join(sample))
                    chosen_per_key[sample] += 1
        for dataset in inventories:
            positives = [
                r
                for r in records
                if r["dataset"] == dataset and r["source_annotations"]
            ]
            if positives:
                crowded = max(positives, key=lambda r: len(r["source_annotations"]))
                tiny = min(
                    positives,
                    key=lambda r: min(
                        a["bbox"][2] * a["bbox"][3] for a in r["source_annotations"]
                    ),
                )
                choices[(dataset, crowded["name"])].add("most boxes")
                choices[(dataset, tiny["name"])].add("smallest box")
        excluded_count = 0
        reasons = {d["key"]: d["reason"] for d in details}
        with (output / "excluded_images.jsonl").open("w", encoding="utf-8") as stream:
            for dataset in inventories:
                selected = (
                    Path(report["source"])
                    / dataset
                    / "selection/selected_annotations.json"
                )
                blocked = defaultdict(set)
                for ann in items(selected, "annotations"):
                    category_key = dataset + ":" + str(ann["category_id"])
                    if mapping[category_key] is None:
                        blocked[str(ann["image_id"])].add(category_key)
                for image in items(selected, "images"):
                    keys = blocked.get(str(image["id"]))
                    if keys:
                        excluded_count += 1
                        stream.write(
                            json.dumps(
                                {
                                    "dataset": dataset,
                                    "file_name": image["file_name"],
                                    "original_image_id": image["source_image_id"],
                                    "reasons": {
                                        key: reasons[key] for key in sorted(keys)
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        automatic = report.get("automatic_exclusions", [])
        with (output / "excluded_images.jsonl").open("a", encoding="utf-8") as stream:
            for entry in automatic:
                stream.write(
                    json.dumps(
                        {**entry, "reasons": {"automatic_cleanup": entry["reason"]}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        excluded_count += len(automatic)
        if len(records) + excluded_count != sum(
            i["selected_images"] for i in inventories.values()
        ):
            raise ValueError(
                "Selected/retained/excluded image accounting does not balance"
            )
        max_error, max_clipping, duplicates, boxes = 0.0, 0.0, 0, 0
        previews = []
        function = partial(
            cached_audit,
            state=getattr(report, "state", None),
            output=output,
            names=report["classes"],
            mapping=mapping,
            allow_clipping=allow_clipping,
        )
        compact = getattr(getattr(report, 'state', None), 'compact', False)
        if compact:
            report.state.prepare_batches(records, 'audit')
        with (
            (
                nullcontext(None)
                if compact
                else (output / "box_audit.jsonl").open("w", encoding="utf-8")
            ) as stream,
            (output / 'repairs.jsonl').open('w', encoding='utf-8') as repairs,
            tqdm(
                total=len(records), desc="Independent output audit", unit="img"
            ) as progress,
            closing(
                (report.state.map_tasks if compact else bounded_map)(
                    function, records, workers, progress
                )
            ) as results,
        ):
            for record, row in zip(records, results):
                row["original_annotation_ids"] = inventories[record["dataset"]][
                    "raw_annotation_ids"
                ].get(str(record["image"]["id"]), [])
                original_ids = dict(
                    zip(
                        (a["id"] for a in record["source_annotations"]),
                        row["original_annotation_ids"],
                    )
                )
                for box in row["boxes"]:
                    box["source_original_annotation_ids"] = [
                        original_ids[key]
                        for key in box["source_selected_annotation_ids"]
                    ]
                max_clipping = max(max_clipping, row["max_clipping_pixels"])
                if (
                    row["max_clipping_pixels"]
                    and chosen_per_key[("clipped", record["dataset"])] < per_class
                ):
                    choices[(record["dataset"], record["name"])].add("edge clipped")
                    chosen_per_key[("clipped", record["dataset"])] += 1
                count = retained[record["dataset"]]
                count["images"] += 1
                count["negative"] += record["role"] == "negative"
                count["boxes"] += len(row["boxes"])
                class_images[record["split"]].update(
                    {b["class_name"] for b in row["boxes"]}
                )
                count["source_boxes"].update(
                    str(a["category_id"]) for a in record["source_annotations"]
                )
                boxes += len(row["boxes"])
                duplicates += row["duplicate_box_records_removed"]
                max_error = max(max_error, row["max_error_pixels"])
                if (record["dataset"], record["name"]) in choices:
                    image_path, _ = output_paths(output, record)
                    name = image_path.stem + ".jpg"
                    relative = Path("previews") / record["dataset"] / name
                    source_names = {
                        str(c["id"]): c["name"]
                        for c in inventories[record["dataset"]]["selected_categories"]
                    }
                    preview(
                        record,
                        output,
                        report["classes"],
                        source_names,
                        output / relative,
                    )
                    previews.append(
                        {
                            **{
                                k: row[k]
                                for k in (
                                    "dataset",
                                    "source_file_name",
                                    "split",
                                    "original_image_id",
                                    "output_label",
                                )
                            },
                            "preview": relative.as_posix(),
                            "source_classes": sorted(
                                {
                                    source_names[str(a["category_id"])]
                                    for a in record["source_annotations"]
                                }
                            ),
                            "reasons": "; ".join(
                                sorted(choices[(record["dataset"], record["name"])])
                            ),
                        }
                    )
                if stream is not None:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                if row['max_clipping_pixels']:
                    repairs.write(
                        json.dumps(
                            {
                                'dataset': record['dataset'],
                                'file_name': record['name'],
                                'reason': 'one_pixel_edge_clipping',
                                'max_clipping_pixels': row['max_clipping_pixels'],
                                'corrected_boxes': sum(
                                    any(
                                        [x, y, x + w, y + h]
                                        != b['expected_clipped_xyxy']
                                        for x, y, w, h in b['source_coco_xywh']
                                    )
                                    for b in row['boxes']
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + '\n'
                    )
        audit = {
            "images_verified": len(records),
            "processing_workers": workers,
            "max_in_flight_tasks": 2 * workers if workers > 1 else 1,
            "unique_image_sha256": (
                None
                if report.get("resumed_written_output")
                else len({r["digest"] for r in records})
            ),
            "duplicate_image_records": (
                None
                if report.get("resumed_written_output")
                else len(records) - len({r["digest"] for r in records})
            ),
            "boxes_verified": boxes,
            "maximum_roundtrip_error_pixels": max_error,
            "maximum_input_clipping_pixels": max_clipping,
            "excluded_images": excluded_count,
            "duplicate_box_records_removed": duplicates,
            "one_pixel_clipped_boxes": report["clipped_boxes_within_one_pixel"],
            "original_selection_images_verified": sum(
                i["selected_images"] for i in inventories.values()
            ),
            "original_selection_boxes_verified": sum(
                i["selected_boxes"] for i in inventories.values()
            ),
            "cross_split_overlap": overlaps,
            "preview_images": len(previews),
            "all_output_image_hashes_match": (
                None if report.get("resumed_written_output") else True
            ),
            "all_output_images_match_source": True,
            "image_verification": (
                'hardlink_identity_or_prior_copy_audit_or_byte_comparison'
                if report.get('resumed_written_output')
                else 'sha256'
            ),
            "prior_sha256_grouping_preserved": report.get(
                'resumed_written_output', False
            ),
            "taxonomy_ancestor_conflicts": 0,
            "allow_one_pixel_clipping": allow_clipping,
        }
        if getattr(report, "state", None):
            report["cache"] = dict(report.state.stats)
        report["audit"] = audit
        report["class_images"] = class_images
        (output / "preview_index.json").write_text(
            json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report['checkpoint_mode'] = 'compact' if compact else 'detailed'
        report['checkpoint'] = (
            {'batch_size': report.state.batch_size, 'per_image_results_saved': False}
            if compact
            else {'per_image_results_saved': True}
        )
        report['audit']['per_image_details_saved'] = not compact
        write_markdown(output, inventories, details, report, audit, previews, retained)
        if compact:
            path = output / 'AUDIT_REPORT.md'
            text = path.read_text(encoding='utf-8')
            text = text.replace(
                '逐图片、逐框记录： [box_audit.jsonl](box_audit.jsonl)；',
                '正常图片不保存逐图校验与逐框明细；修正清单：[repairs.jsonl](repairs.jsonl)；',
            )
            text = text.replace(
                '划分清单：[split_manifest.jsonl](split_manifest.jsonl)',
                '划分结果见 images/train、images/val、images/test',
            )
            path.write_text(text, encoding='utf-8')
        with (output / "AUDIT_REPORT.md").open("a", encoding="utf-8") as stream:
            if report.get('resumed_written_output'):
                stream.write('\n## Resume from completed YOLO output\n\n')
                stream.write(
                    'Input image decoding and input image hashing were skipped. Existing split assignments were preserved; camera/sequence overlap was checked again. Previous content-hash grouping was reused from unchanged complete input batch checkpoints. No new unique-image SHA count is claimed.\n\n'
                )
                stream.write(
                    'Output images were verified by hardlink identity, reused completed copy audits with matching size/mtime, or byte comparison for copies without a reusable audit. All retained label coordinates were independently decoded again.\n'
                )
            stream.write("\n## Automatic cleanup and checkpoints\n\n")
            stream.write(
                f"Automatically excluded images: {len(automatic)}. One-pixel clipped boxes: {report['clipped_boxes_within_one_pixel']}.\n\n"
            )
            stream.write(
                "Original image files are preserved. Invalid images/annotations are excluded as whole images; they are never converted into negative samples.\n\n"
            )
            stream.write(
                "Full exclusions: `excluded_images.jsonl`; validation errors: `validation_errors.jsonl`.\n\n"
            )
            stream.write("| Dataset | Automatically excluded images |\n|---|---:|\n")
            for dataset in inventories:
                stream.write(
                    f"| {dataset} | {sum(e['dataset'] == dataset for e in automatic)} |\n"
                )
            stream.write(
                "\nProcessing counters: `"
                + json.dumps(report.get("cache", {}), sort_keys=True)
                + "`.\n"
            )

    return audit


def _merge(
    source,
    output,
    *,
    datasets=None,
    taxonomy_file=TAXONOMY,
    original_files=None,
    exclusions=(),
    val_fraction=0.05,
    test_fraction=0.05,
    seed=42,
    copy_mode="auto",
    previews_per_class=1,
    allow_clipping=True,
    group_map=None,
    workers=8,
    auto_clean=True,
    state=None,
    resume_output=False,
):
    split_fractions(val_fraction, test_fraction)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("--workers must be a positive integer")
    source, output = Path(source).resolve(), Path(output).resolve()

    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must not contain each other")
    if previews_per_class < 1:
        raise ValueError("--previews-per-class must be at least 1")
    inventories = inspect_sources(source, datasets or DEFAULT_DATASETS, original_files)
    mapping, details = taxonomy_mapping(inventories, taxonomy_file, exclusions)
    if state:
        state.configure({"class_map": mapping})
    auditor = make_auditor(
        inventories,
        details,
        taxonomy_file,
        mapping,
        previews_per_class,
        allow_clipping,
        workers=workers,
    )
    return prepare(
        source,
        output,
        datasets=datasets or DEFAULT_DATASETS,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        copy_mode=copy_mode,
        class_map=mapping,
        group_map=group_map,
        audit_hook=auditor,
        strict_boxes=not allow_clipping,
        workers=workers,
        state=state,
        auto_clean=auto_clean,
        resume_output=resume_output,
    )


def merge(source, output, *, recheck_all=False, checkpoint_mode='compact', **options):
    if options.get('resume_output') and (recheck_all or checkpoint_mode != 'compact'):
        raise ValueError(
            '--resume-output cannot use --recheck-all or --detailed-audit; full validation will not run'
        )
    config = {
        k: options.get(k, default)
        for k, default in {
            "datasets": DEFAULT_DATASETS,
            "val_fraction": 0.05,
            "test_fraction": 0.05,
            "seed": 42,
            "copy_mode": "auto",
            "group_map": None,
            "exclusions": (),
        }.items()
    }
    # JSON canonical forms avoid tuple/list differences between runs.
    config = json.loads(json.dumps(config))
    if checkpoint_mode not in ('compact', 'detailed'):
        raise ValueError('checkpoint_mode must be compact or detailed')
    with managed_run(
        source, output, config, force=recheck_all, compact=checkpoint_mode == 'compact'
    ) as state:
        return _merge(source, output, state=state, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/lila"))
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/lila_yolo_audited")
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--taxonomy-file", type=Path, default=TAXONOMY)
    parser.add_argument(
        "--original-files",
        type=Path,
        help="JSON: dataset -> metadata/bbox paths, for moved original files",
    )
    parser.add_argument("--group-map", type=Path)
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        help="dataset:category_id; drop entire images containing this class",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent validation, writing and audit tasks; 1 runs serially",
    )
    parser.add_argument(
        "--copy-mode", choices=["auto", "hardlink", "copy"], default="auto"
    )
    parser.add_argument("--previews-per-class", type=int, default=1)
    parser.add_argument(
        "--allow-one-pixel-clipping",
        action="store_true",
        default=True,
        help="Compatibility option; one-pixel clipping is enabled by default",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on invalid inputs and reject clipping",
    )
    parser.add_argument(
        "--recheck-all",
        action="store_true",
        help="Ignore batch checkpoints and fully recheck inputs and outputs",
    )
    parser.add_argument(
        '--detailed-audit',
        action='store_true',
        help='Opt in to persistent per-image cache and full box/manifest logs; default saves batch progress and exceptions only',
    )
    parser.add_argument(
        '--resume-output',
        action='store_true',
        help='Resume completed Write YOLO outputs, preserve splits, and NEVER run input Validate and hash',
    )
    args = parser.parse_args()
    try:

        def load(path):
            value = json.loads(path.read_text(encoding="utf-8-sig")) if path else None
            if value is not None and not isinstance(value, dict):
                raise ValueError("JSON options must contain an object")
            return value

        restored_group_map = load(args.group_map)
        if args.resume_output:
            saved = load(args.output / '.lila-state/run.json')['config']

            def provided(flag):
                return any(
                    arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]
                )

            for field in (
                'source',
                'datasets',
                'val_fraction',
                'test_fraction',
                'seed',
                'copy_mode',
            ):
                if not provided('--' + field.replace('_', '-')) and field in saved:
                    setattr(
                        args,
                        field,
                        Path(saved[field]) if field == 'source' else saved[field],
                    )
            if not provided('--group-map'):
                restored_group_map = saved.get('group_map')
            if not provided('--exclude-category'):
                args.exclude_category = saved.get('exclusions', [])
            provenance_path = args.output / 'audit_sources.json'
            if not provided('--taxonomy-file') and provenance_path.is_file():
                args.taxonomy_file = Path(load(provenance_path)['taxonomy_file'])
            print(
                'Restored source, class/split settings and copy mode from the existing output.',
                flush=True,
            )

        merge(
            args.source,
            args.output,
            datasets=args.datasets,
            taxonomy_file=args.taxonomy_file,
            original_files=load(args.original_files),
            exclusions=args.exclude_category,
            group_map=restored_group_map,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            workers=args.workers,
            copy_mode=args.copy_mode,
            previews_per_class=args.previews_per_class,
            allow_clipping=not args.strict,
            auto_clean=not args.strict,
            recheck_all=args.recheck_all,
            resume_output=args.resume_output,
            checkpoint_mode='detailed' if args.detailed_audit else 'compact',
        )
        print("Audit passed. Report: " + str(args.output.resolve() / "AUDIT_REPORT.md"))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(
            1,
            f"Error: {exc}\nConversion is NOT certified; do not train on partial output. Check the saved error/checkpoint files; rerun the same command to resume.\n",
        )


if __name__ == "__main__":
    main()
