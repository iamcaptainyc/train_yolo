import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from prepare_lila import prepare, read_records, split_records, split_overlap


def make_dataset(root, name, *, missing_groups=False):
    folder = root / name
    (folder / 'selection').mkdir(parents=True)
    (folder / 'images').mkdir()
    images, annotations = [], []
    for index in range(12):
        filename = f'frame_{index}.jpg'
        Image.new('RGB', (100, 80), (index * 15, len(name) * 7, 20)).save(
            folder / 'images' / filename
        )
        image = {
            'id': index,
            'file_name': filename,
            'width': 100,
            'height': 80,
            'selection_role': 'negative' if index == 0 else 'positive',
        }
        if not missing_groups:
            image.update(location=str(index // 3), seq_id='sequence_' + str(index // 2))
        images.append(image)
        if index:
            annotations.append(
                {
                    'id': index,
                    'image_id': index,
                    'category_id': 4,
                    'bbox': [10, 20, 30, 40],
                }
            )
    path = folder / 'selection/selected_annotations.json'
    path.write_text(
        json.dumps(
            {
                'images': images,
                'annotations': annotations,
                'categories': [{'id': 4, 'name': 'Domestic_Dog'}],
            }
        )
    )
    return path


class PrepareLilaTests(unittest.TestCase):
    def test_full_conversion_and_components_are_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / 'source', 'camera')
            make_dataset(root / 'source', 'ena', missing_groups=True)
            report = prepare(
                root / 'source',
                root / 'out',
                datasets=['camera', 'ena'],
                copy_mode='copy',
            )
            manifest = [
                json.loads(line)
                for line in (root / 'out/split_manifest.jsonl').read_text().splitlines()
            ]
            for key in ('component', 'digest'):
                self.assertFalse(
                    {r[key] for r in manifest if r['split'] == 'train'}
                    & {r[key] for r in manifest if r['split'] == 'val'}
                )
            self.assertTrue(
                all(r['split'] == 'train' for r in manifest if r['dataset'] == 'ena')
            )
            self.assertEqual(report['classes'], ['dog'])
            self.assertEqual(report['train_only_datasets_missing_group_info'], ['ena'])
            labels = list((root / 'out/labels').rglob('*.txt'))
            self.assertEqual(len(labels), 24)
            self.assertEqual(sum(not p.read_text() for p in labels), 2)
            self.assertEqual(
                next(p.read_text() for p in labels if p.read_text()).strip(),
                '0 0.25000000 0.50000000 0.30000000 0.50000000',
            )
            config = yaml.safe_load((root / 'out/data.yaml').read_text())
            self.assertEqual(config['names'], {0: 'dog'})
            self.assertTrue(report['sha256_overlap_checked'])

    def test_plan_requires_no_images_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / 'source', 'camera')
            for p in (root / 'source/camera/images').glob('*'):
                p.unlink()
            first = prepare(
                root / 'source', root / 'plan1', datasets=['camera'], plan_only=True
            )
            prepare(
                root / 'source', root / 'plan2', datasets=['camera'], plan_only=True
            )
            self.assertEqual(
                (root / 'plan1/split_manifest.jsonl').read_text(),
                (root / 'plan2/split_manifest.jsonl').read_text(),
            )
            self.assertFalse((root / 'plan1/data.yaml').exists())
            self.assertFalse(first['image_files_verified'])
            with self.assertRaisesRegex(ValueError, 'finish downloads'):
                prepare(root / 'source', root / 'actual', datasets=['camera'])

    def test_vehicle_drops_entire_image_and_custom_names_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_dataset(root, 'camera')
            doc = json.loads(path.read_text())
            doc['categories'].append({'id': 9, 'name': 'car'})
            doc['annotations'].append(
                {'image_id': 1, 'category_id': 9, 'bbox': [1, 1, 2, 2]}
            )
            path.write_text(json.dumps(doc))
            records, mapping, skipped, _ = read_records(
                root, ['camera'], {'domestic dog': 'canine'}
            )
            self.assertEqual(len(records), 11)
            self.assertEqual(skipped['camera:excluded_category_images'], 1)
            self.assertEqual(mapping[0]['target'], 'canine')
            self.assertFalse(any(r['name'] == 'frame_1.jpg' for r in records))

    def test_sha256_bridges_cameras_and_sequences(self):
        records = []
        for group, seq, digest in [
            ('a', 's1', 'x'),
            ('b', 's2', 'x'),
            ('c', 's2', 'y'),
            ('d', 's3', 'z'),
        ]:
            records.append(
                {
                    'dataset': 'd',
                    'group': group,
                    'sequence': seq,
                    'digest': digest,
                    'fallback': False,
                }
            )
        split_records(records, 0.25, 42)
        self.assertEqual(len({r['component'] for r in records[:3]}), 1)
        self.assertEqual(len({r['split'] for r in records[:3]}), 1)
        self.assertNotEqual(records[0]['split'], records[3]['split'])

    def test_missing_metadata_dimensions_use_actual_image_and_empty_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_dataset(root / 'source', 'camera')
            doc = json.loads(path.read_text())
            for image in doc['images']:
                image['width'] = None
                image.pop('height')
            path.write_text(json.dumps(doc))
            plan = prepare(
                root / 'source', root / 'plan', datasets=['camera'], plan_only=True
            )
            self.assertEqual(plan['plan_positive_images_needing_dimensions'], 11)
            actual = prepare(
                root / 'source', root / 'out', datasets=['camera'], copy_mode='copy'
            )
            self.assertTrue(actual['image_files_verified'])

    def test_conflicting_duplicate_labels_refuse_training_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root / 'source', 'camera')
            a = root / 'source/camera/images/frame_0.jpg'
            b = root / 'source/camera/images/frame_1.jpg'
            a.write_bytes(b.read_bytes())
            with self.assertRaisesRegex(ValueError, 'conflicting annotations'):
                prepare(root / 'source', root / 'out', datasets=['camera'])
            self.assertFalse((root / 'out/data.yaml').exists())

    def test_single_group_refuses_random_image_split(self):
        with self.assertRaisesRegex(ValueError, 'Only one connected group'):
            split_records(
                [
                    {
                        'dataset': 'd',
                        'group': 'one',
                        'sequence': None,
                        'fallback': False,
                    }
                    for _ in range(4)
                ],
                0.2,
                42,
            )

    def test_rare_class_camera_stays_in_train(self):
        records = [
            {
                'dataset': 'd',
                'group': str(i),
                'sequence': None,
                'fallback': False,
                'labels': [(1 if i == 0 else 0, 0.5, 0.5, 0.2, 0.2)],
            }
            for i in range(8)
        ]
        split_records(records, 0.2, 42)
        self.assertEqual(records[0]['split'], 'train')
        self.assertEqual(
            {b[0] for r in records if r['split'] == 'train' for b in r['labels']},
            {0, 1},
        )


class ThreeWaySplitTests(unittest.TestCase):
    def record(self, index, dataset='d', **overrides):
        return {
            'dataset': dataset,
            'group': dataset + ':camera:' + str(index),
            'location': str(index),
            'sequence': 'seq_' + str(index),
            'digest': dataset + ':sha:' + str(index),
            'fallback': False,
            'labels': [(0, 0.5, 0.5, 0.2, 0.2)],
            **overrides,
        }

    def test_90505_is_exact_for_many_equal_independent_groups_and_reproducible(self):
        records = [self.record(i) for i in range(1000)]
        repeated = deepcopy(records)
        split_records(records, 0.05, 42, 0.05)
        split_records(repeated, 0.05, 42, 0.05)
        self.assertEqual(
            Counter(r['split'] for r in records), {'train': 900, 'val': 50, 'test': 50}
        )
        self.assertEqual([r['split'] for r in records], [r['split'] for r in repeated])
        self.assertTrue(
            all(
                n == 0
                for pair in split_overlap(records).values()
                for n in pair.values()
            )
        )

    def test_transitive_camera_sequence_and_cross_dataset_duplicate_groups_stay_together(
        self,
    ):
        records = [self.record(i) for i in range(100)]
        records[1]['group'] = records[0]['group']
        records[1]['location'] = records[0]['location']
        records[2]['sequence'] = records[1]['sequence']
        records[3] = self.record(3, dataset='other', digest=records[2]['digest'])
        records[4]['fallback'] = True
        records[5]['labels'] = [(1, 0.5, 0.5, 0.2, 0.2)]
        split_records(records, 0.05, 42, 0.05)
        self.assertEqual(len({r['component'] for r in records[:4]}), 1)
        self.assertEqual(len({r['split'] for r in records[:4]}), 1)
        self.assertEqual(records[4]['split'], 'train')
        self.assertEqual(records[5]['split'], 'train')
        self.assertEqual({r['split'] for r in records}, {'train', 'val', 'test'})
        self.assertTrue(
            all(
                n == 0
                for pair in split_overlap(records).values()
                for n in pair.values()
            )
        )

    def test_joint_holdout_reservation_keeps_last_rare_class_in_train(self):
        records = [
            self.record(0, labels=[(1, 0.5, 0.5, 0.2, 0.2)]),
            self.record(1, labels=[(1, 0.5, 0.5, 0.2, 0.2)]),
            self.record(2),
            self.record(3),
        ]
        split_records(records, 0.05, 42, 0.05)
        self.assertEqual({r['split'] for r in records}, {'train', 'val', 'test'})
        self.assertEqual(
            {
                label[0]
                for r in records
                if r['split'] == 'train'
                for label in r['labels']
            },
            {0, 1},
        )

    def test_insufficient_independent_positive_groups_refuses_test(self):
        records = [self.record(i) for i in range(2)]
        with self.assertRaisesRegex(
            ValueError, 'insufficient independent positive groups'
        ):
            split_records(records, 0.05, 42, 0.05)
        records = [
            self.record(i, labels=[] if i else [(0, 0.5, 0.5, 0.2, 0.2)])
            for i in range(20)
        ]
        with self.assertRaises(ValueError):
            split_records(records, 0.05, 42, 0.05)

    def test_invalid_split_fractions_fail(self):
        for val, test in (
            (0.6, 0.4),
            (0.1, -0.1),
            (0, 0.05),
            (float('nan'), 0.05),
            (0.05, float('inf')),
        ):
            with self.subTest(val=val, test=test), self.assertRaises(ValueError):
                split_records([self.record(i) for i in range(10)], val, 42, test)

    def test_pairwise_audit_detects_camera_sequence_and_hash_leakage(self):
        records = [
            self.record(0, split='train', component=0),
            self.record(1, split='val', component=1, location='0'),
            self.record(
                2, split='test', component=2, sequence='seq_1', digest='d:sha:0'
            ),
        ]
        overlap = split_overlap(records)
        self.assertEqual(overlap['train_vs_val']['camera'], 1)
        self.assertEqual(overlap['val_vs_test']['sequence'], 1)
        self.assertEqual(overlap['train_vs_test']['digest'], 1)


if __name__ == '__main__':
    unittest.main()
