import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image, ImageDraw
import yaml

from lila_audit import DATASET_NAMES, TAXONOMY, inspect_sources, taxonomy_mapping
from prepare_lila_audited import merge, make_auditor


def fixture(root, dataset, names):
    folder = root / dataset
    (folder / 'annotations').mkdir(parents=True)
    (folder / 'selection').mkdir()
    (folder / 'images').mkdir()
    cats = [{'id': 0, 'name': 'empty'}] + [
        {'id': i + 4, 'name': name} for i, name in enumerate(names)
    ]
    images = []
    anns = []
    selected_images = []
    selected_anns = []
    for index in range(12):
        name = f'frame_{index}.png'
        image = Image.new('RGB', (120, 90), (index * 17, len(dataset) * 5, 11))
        ImageDraw.Draw(image).rectangle((12, 9, 48, 54), fill=(230, 80, 20))
        image.save(folder / 'images' / name)
        original = {
            'id': f'original_{index}',
            'file_name': name,
            'width': 120,
            'height': 90,
            'location': str(index // 3),
            'seq_id': str(index // 3),
        }
        images.append(original)
        selected_images.append(
            {
                **original,
                'id': index,
                'source_image_id': original['id'],
                'selection_role': 'negative' if index == 0 else 'positive',
            }
        )
        ann = {
            'id': 1000 + index,
            'image_id': original['id'],
            'category_id': 0 if index == 0 else 4 + (index - 1) % len(names),
        }
        if index:
            ann['bbox'] = [12, 9, 36, 45]
            selected_anns.append({**ann, 'id': index, 'image_id': index})
        anns.append(ann)
    original = folder / 'annotations/original.json'
    original.write_text(
        json.dumps({'categories': cats, 'images': images, 'annotations': anns})
    )
    selected = folder / 'selection/selected_annotations.json'
    selected.write_text(
        json.dumps(
            {
                'categories': cats,
                'images': selected_images,
                'annotations': selected_anns,
            }
        )
    )
    (folder / 'selection/selection_report.json').write_text(
        json.dumps({'metadata_file': str(original), 'bbox_file': str(original)})
    )
    return original, selected


class AuditedLilaTests(unittest.TestCase):
    def test_complete_report_and_numerical_box_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source', 'caltech-camera-traps', ['opossum', 'bird'])
            fixture(root / 'source', 'ena24', ['american crow'])
            report = merge(
                root / 'source',
                root / 'out',
                datasets=['caltech-camera-traps', 'ena24'],
                copy_mode='copy',
            )
            self.assertEqual(
                report['classes'], ['class:aves', 'species:didelphis virginiana']
            )
            rows = [
                json.loads(line)
                for line in (root / 'out/box_audit.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(len(r['boxes']) for r in rows), 22)
            for row in rows:
                if row['role'] == 'positive':
                    box = row['boxes'][0]
                    self.assertEqual(box['yolo'], [0.25, 0.35, 0.3, 0.5])
                    for actual, expected in zip(box['xyxy'], [12, 9, 48, 54]):
                        self.assertAlmostEqual(actual, expected, places=10)
                    self.assertTrue(row['original_annotation_ids'])
                else:
                    self.assertEqual(row['boxes'], [])
            self.assertEqual(
                report['audit']['cross_split_overlap'],
                {
                    pair: {
                        'group': 0,
                        'camera': 0,
                        'component': 0,
                        'digest': 0,
                        'sequence': 0,
                    }
                    for pair in ('train_vs_val', 'train_vs_test', 'val_vs_test')
                },
            )
            text = (root / 'out/AUDIT_REPORT.md').read_text(encoding='utf-8')
            for name in ('opossum', 'american crow', 'class:aves', 'bbox', '22'):
                self.assertIn(name, text)
            previews = json.loads((root / 'out/preview_index.json').read_text())
            self.assertTrue(
                {'opossum', 'bird', 'american crow'}
                <= {name for p in previews for name in p['source_classes']}
            )
            for item in previews:
                with Image.open(root / 'out' / item['preview']) as image:
                    self.assertEqual(image.size, (2012, 805))
            config = yaml.safe_load((root / 'out/data.yaml').read_text())
            self.assertEqual(config['test'], 'images/test')
            self.assertEqual(
                report['requested_split_fractions'],
                {'train': 0.9, 'val': 0.05, 'test': 0.05},
            )
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            self.assertEqual({p['split'] for p in previews}, {'train', 'val', 'test'})
            for name in report['classes']:
                self.assertEqual(
                    sum(
                        report['stats'][split]['boxes'][name]
                        for split in report['stats']
                    ),
                    sum(
                        box['class_name'] == name
                        for row in rows
                        for box in row['boxes']
                    ),
                )
            for split in ('train', 'val', 'test'):
                self.assertEqual(
                    report['stats'][split]['images'],
                    sum(r['split'] == split for r in rows),
                )
                self.assertEqual(
                    sum(
                        report['dataset_split_stats'][d][split]['images']
                        for d in report['dataset_split_stats']
                    ),
                    report['stats'][split]['images'],
                )
            for expected in ('test 框', 'test 图片', 'val_vs_test', '90.00%', '5.00%'):
                self.assertIn(expected, text)

    def test_parallel_conversion_matches_serial_labels_splits_and_box_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source', 'caltech-camera-traps', ['opossum', 'bird'])
            fixture(root / 'source', 'ena24', ['american crow'])
            reports = []
            for workers in (1, 4):
                reports.append(
                    merge(
                        root / 'source',
                        root / f'out_{workers}',
                        datasets=['caltech-camera-traps', 'ena24'],
                        copy_mode='copy',
                        workers=workers,
                    )
                )
            self.assertEqual(reports[0]['stats'], reports[1]['stats'])
            self.assertEqual(
                reports[0]['actual_split_fractions'],
                reports[1]['actual_split_fractions'],
            )
            for name in (
                'split_manifest.jsonl',
                'box_audit.jsonl',
                'preview_index.json',
            ):
                self.assertEqual(
                    (root / 'out_1' / name).read_bytes(),
                    (root / 'out_4' / name).read_bytes(),
                )
            for label in (root / 'out_1/labels').rglob('*.txt'):
                self.assertEqual(
                    label.read_bytes(),
                    (root / 'out_4' / label.relative_to(root / 'out_1')).read_bytes(),
                )
            self.assertEqual(reports[1]['audit']['processing_workers'], 4)
            self.assertEqual(reports[1]['audit']['max_in_flight_tasks'], 8)

    def test_parallel_validation_still_rejects_bad_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source', 'caltech-camera-traps', ['opossum'])
            (root / 'source/caltech-camera-traps/images/frame_5.png').write_bytes(
                b'corrupt image'
            )
            with self.assertRaisesRegex(
                ValueError, 'Input files or annotations invalid'
            ):
                merge(
                    root / 'source',
                    root / 'out',
                    datasets=['caltech-camera-traps'],
                    copy_mode='copy',
                    workers=4,
                )
            self.assertFalse((root / 'out/data.yaml').exists())

    def test_fully_excluded_dataset_still_has_zero_three_way_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source', 'caltech-camera-traps', ['opossum'])
            _, selected = fixture(root / 'source', 'ena24', ['american crow'])
            doc = json.loads(selected.read_text())
            doc['images'] = [
                image
                for image in doc['images']
                if image['selection_role'] == 'positive'
            ]
            selected.write_text(json.dumps(doc))
            report = merge(
                root / 'source',
                root / 'out',
                datasets=['caltech-camera-traps', 'ena24'],
                copy_mode='copy',
                exclusions=['ena24:4'],
            )
            self.assertEqual(
                report['dataset_split_stats']['ena24'],
                {
                    split: {'images': 0, 'negative_images': 0, 'boxes': 0}
                    for split in ('train', 'val', 'test')
                },
            )
            self.assertEqual(report['audit']['excluded_images'], 11)
            self.assertTrue((root / 'out/AUDIT_REPORT.md').is_file())
            self.assertTrue((root / 'out/data.yaml').is_file())

    def test_parent_species_conflicts_and_overly_broad_classes(self):
        inventories = {
            'caltech-camera-traps': {
                'selected_categories': [
                    {'id': 4, 'name': 'bird', 'boxes': 10},
                    {'id': 5, 'name': 'opossum', 'boxes': 3},
                ]
            },
            'ena24': {
                'selected_categories': [{'id': 4, 'name': 'american crow', 'boxes': 5}]
            },
            'swg-camera-traps': {
                'selected_categories': [
                    {'id': 4, 'name': 'unidentified_mammal', 'boxes': 8}
                ]
            },
        }
        mapping, details = taxonomy_mapping(inventories)
        self.assertEqual(mapping['caltech-camera-traps:4'], mapping['ena24:4'])
        self.assertNotEqual(
            mapping['caltech-camera-traps:4'], mapping['caltech-camera-traps:5']
        )
        self.assertIsNone(mapping['swg-camera-traps:4'])
        self.assertIn(
            'coarsened', next(d['reason'] for d in details if d['key'] == 'ena24:4')
        )

    def test_pinned_real_taxonomy_all_five_datasets_is_consistent(self):
        with TAXONOMY.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        inventories = {
            d: {
                'selected_categories': [
                    {'id': i, 'name': r['query'], 'boxes': 1}
                    for i, r in enumerate(rows)
                    if r['dataset_name'] == name
                ]
            }
            for d, name in DATASET_NAMES.items()
        }
        mapping, details = taxonomy_mapping(inventories)
        self.assertEqual(len(mapping), 257)
        targets = {d['target_taxon_id'] for d in details if d.get('target')}
        for d in details:
            if d.get('target'):
                lineage = d['lineage']
                position = lineage.index(d['target_taxon_id'])
                self.assertFalse(set(lineage[position + 1 :]) & targets)

    def test_broad_category_excludes_whole_image_not_only_box(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(
                root / 'source',
                'swg-camera-traps',
                ['eurasian_wild_pig', 'unidentified_mammal'],
            )
            # Mixed image: specific species and broad unknown mammal; the whole image must go.
            for path, image_id, ann_id in (
                (original, 'original_1', 9001),
                (selected, 1, 9002),
            ):
                doc = json.loads(path.read_text())
                doc['annotations'].append(
                    {
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': 5,
                        'bbox': [60, 10, 20, 30],
                    }
                )
                path.write_text(json.dumps(doc))
            report = merge(
                root / 'source',
                root / 'out',
                datasets=['swg-camera-traps'],
                copy_mode='copy',
            )
            self.assertEqual(
                report['excluded_images']['swg-camera-traps:excluded_category_images'],
                6,
            )
            rows = [
                json.loads(line)
                for line in (root / 'out/box_audit.jsonl').read_text().splitlines()
            ]
            self.assertFalse(any(r['selected_image_id'] == 1 for r in rows))
            self.assertEqual(sum(r['role'] == 'negative' for r in rows), 1)

    def test_wrong_selected_coordinates_or_image_path_are_rejected(self):
        for mutation in ('coordinate', 'path', 'dangling', 'category'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, selected = fixture(root, 'caltech-camera-traps', ['opossum'])
                doc = json.loads(selected.read_text())
                if mutation == 'coordinate':
                    doc['annotations'][0]['bbox'][0] += 1
                elif mutation == 'path':
                    doc['images'][1]['file_name'] = 'frame_2.png'
                elif mutation == 'dangling':
                    doc['annotations'][0]['image_id'] = 999
                else:
                    doc['categories'][1]['name'] = 'bird'
                selected.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    inspect_sources(root, ['caltech-camera-traps'])

    def test_false_negative_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, _ = fixture(root, 'caltech-camera-traps', ['opossum'])
            doc = json.loads(original.read_text())
            doc['annotations'][0]['category_id'] = 4
            original.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'negative image'):
                inspect_sources(root, ['caltech-camera-traps'])

    def test_written_label_or_image_tampering_prevents_training_yaml(self):
        for mutation in ('label', 'image'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fixture(root / 'source', 'caltech-camera-traps', ['opossum'])

                def factory(*args, **kwargs):
                    auditor = make_auditor(*args, **kwargs)

                    def altered(records, mapping, report, output):
                        if mutation == 'label':
                            label = next(
                                p
                                for p in (output / 'labels').rglob('*.txt')
                                if p.read_text()
                            )
                            label.write_text('0 0.9 0.9 0.1 0.1\n')
                        else:
                            image = next((output / 'images').rglob('*.png'))
                            image.write_bytes(b'wrong image')
                        auditor(records, mapping, report, output)

                    return altered

                with patch(
                    'prepare_lila_audited.make_auditor', side_effect=factory
                ), self.assertRaises(ValueError):
                    merge(
                        root / 'source',
                        root / 'out',
                        datasets=['caltech-camera-traps'],
                        copy_mode='copy',
                    )
                self.assertFalse((root / 'out/data.yaml').exists())
                self.assertFalse((root / 'out/AUDIT_REPORT.md').exists())

    def test_moved_original_files_are_resolved_by_basename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 'caltech-camera-traps', ['opossum'])
            path = root / 'caltech-camera-traps/selection/selection_report.json'
            path.write_text(
                json.dumps(
                    {
                        'metadata_file': '/old/server/original.json',
                        'bbox_file': '/old/server/original.json',
                    }
                )
            )
            inv = inspect_sources(root, ['caltech-camera-traps'])
            self.assertEqual(inv['caltech-camera-traps']['selected_boxes'], 11)

    def test_separate_metadata_and_bbox_category_namespaces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(
                root / 'source', 'caltech-camera-traps', ['opossum']
            )
            bbox = original.with_name('detections.json')
            bbox.write_bytes(original.read_bytes())
            doc = json.loads(original.read_text())
            doc['categories'][1]['id'] = 104
            for ann in doc['annotations']:
                ann.pop('bbox', None)
                if ann['category_id'] == 4:
                    ann['category_id'] = 104
            original.write_text(json.dumps(doc))
            report_file = selected.with_name('selection_report.json')
            report_file.write_text(
                json.dumps({'metadata_file': str(original), 'bbox_file': str(bbox)})
            )
            report = merge(
                root / 'source',
                root / 'out',
                datasets=['caltech-camera-traps'],
                copy_mode='copy',
            )
            self.assertEqual(report['audit']['boxes_verified'], 11)
            sources = json.loads((root / 'out/audit_sources.json').read_text())[
                'datasets'
            ]['caltech-camera-traps']
            self.assertEqual(sources['metadata']['bbox_records'], 0)
            self.assertEqual(sources['bbox']['bbox_records'], 11)
            self.assertEqual(sources['metadata']['categories'][1]['id'], 104)
            self.assertEqual(sources['bbox']['categories'][1]['id'], 4)

    def test_edge_clipping_is_opt_in_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(
                root / 'source', 'caltech-camera-traps', ['opossum']
            )
            for path in (original, selected):
                doc = json.loads(path.read_text())
                for ann in doc['annotations']:
                    if 'bbox' in ann:
                        ann['bbox'] = [12, 9, 108.5, 45]
                path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'edge clipping'):
                merge(
                    root / 'source',
                    root / 'strict',
                    datasets=['caltech-camera-traps'],
                    copy_mode='copy',
                )
            self.assertFalse((root / 'strict/data.yaml').exists())
            report = merge(
                root / 'source',
                root / 'allowed',
                datasets=['caltech-camera-traps'],
                copy_mode='copy',
                allow_clipping=True,
            )
            self.assertEqual(report['audit']['one_pixel_clipped_boxes'], 11)
            self.assertTrue(report['audit']['allow_one_pixel_clipping'])

    def test_zip_source_matches_cache_and_detects_cache_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(root, 'caltech-camera-traps', ['opossum'])
            zipped = original.with_suffix('.zip')
            with zipfile.ZipFile(zipped, 'w') as bundle:
                bundle.write(original, 'original.json')
            selected.with_name('selection_report.json').write_text(
                json.dumps({'metadata_file': str(zipped), 'bbox_file': str(zipped)})
            )
            inv = inspect_sources(root, ['caltech-camera-traps'])
            cached = Path(inv['caltech-camera-traps']['metadata']['file'])
            cached.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                inspect_sources(root, ['caltech-camera-traps'])

    def test_missing_taxonomy_and_unknown_exclusion_fail(self):
        inv = {
            'caltech-camera-traps': {
                'selected_categories': [
                    {'id': 4, 'name': 'invented_species', 'boxes': 2}
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, 'No official taxonomy'):
            taxonomy_mapping(inv)
        with self.assertRaisesRegex(ValueError, 'Unknown --exclude-category'):
            taxonomy_mapping(inv, exclusions=['wrong:4'])

    def test_duplicate_raw_boxes_are_counted_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(
                root / 'source', 'caltech-camera-traps', ['opossum']
            )
            for path in (original, selected):
                doc = json.loads(path.read_text())
                index = 1 if path == original else 0
                duplicate = {**doc['annotations'][index], 'id': 9999}
                doc['annotations'].append(duplicate)
                path.write_text(json.dumps(doc))
            report = merge(
                root / 'source',
                root / 'out',
                datasets=['caltech-camera-traps'],
                copy_mode='copy',
            )
            self.assertEqual(report['audit']['boxes_verified'], 11)
            self.assertEqual(report['audit']['duplicate_box_records_removed'], 1)


if __name__ == '__main__':
    unittest.main()
