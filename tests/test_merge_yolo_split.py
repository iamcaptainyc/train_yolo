import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import merge_yolo_datasets as merger
from merge_yolo_split import attach_groups, fold_rare, resplit
from test_merge_yolo_datasets import fixture


class ResplitTests(unittest.TestCase):
    def test_per_species_image_ratios_with_multiple_boxes(self):
        records = []
        for name, count in [('rabbit', 100), ('bear', 200)]:
            for j in range(count):
                i = len(records)
                records.append({'rows': [(name, ('0.5',)*4)] * (1 + j % 5),
                                'group_keys': [('image', i)], 'image_stamp': {'inode': 0},
                                'split': 'train', 'image_rel': Path(f'images/train/{i}.jpg'),
                                'label_rel': Path(f'labels/train/{i}.txt')})
        result = resplit(records, .1, .1, 42)
        self.assertEqual(result['class_image_counts']['rabbit'], {'train': 80, 'val': 10, 'test': 10})
        self.assertEqual(result['class_image_counts']['bear'], {'train': 160, 'val': 20, 'test': 20})
        self.assertEqual(result['group_overlap'], 0)

    def test_wolf_application_group_merges_coyote_and_jackal_before_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / 'lila', ['jackal', 'coyote'], 'lila')
            fixture(root / 'animals', ['snake'], 'animals')
            result = merger.merge(root / 'lila', root / 'animals', root / 'out')
            self.assertIn('wolf', result['classes'])
            self.assertNotIn('jackal', result['classes'])
            self.assertNotIn('coyote', result['classes'])
            mapping = result['source_class_mapping']['lila']
            self.assertEqual([m['target'] for m in mapping], ['wolf', 'wolf'])
            self.assertEqual([m['source_name'] for m in mapping], ['jackal', 'coyote'])
            # Same class/same box after alias merging is deduplicated, not counted twice.
            self.assertEqual(result['stats']['train']['boxes']['wolf'], 6)
            self.assertEqual(result['audit']['duplicate_box_records_removed'], 6)
            self.assertIn('不代表已获得灰狼', (root / 'out/MERGE_REPORT.md').read_text(encoding='utf-8'))

    def inputs(self, root):
        fixture(root / 'lila', ['monitor_lizard', 'wild_boar', 'pig'], 'lila')
        fixture(root / 'animals', ['brown bear', 'snake'], 'animals')
        # Explicit groups span old train/val/test: old splits must not be used.
        groups = {}
        for alias in ('lila', 'animals'):
            for split in merger.SPLITS:
                for name in ('a', 'b'):
                    groups[f'{alias}/{split}/{name}.jpg'] = f'{alias}/{name}/{split if alias == "animals" else "camera"}'
        path = root / 'groups.json'
        path.write_text(json.dumps(groups))
        return path

    def test_default_folds_resplits_and_preserves_group_links_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = self.inputs(root)
            with patch('hashlib.file_digest', side_effect=AssertionError('image hash')), patch('PIL.Image.open', side_effect=AssertionError('decode')):
                report = merger.merge(root / 'lila', root / 'animals', root / 'out', workers=2, group_map=groups)
            self.assertFalse(report['audit']['source_splits_preserved'])
            self.assertIn('lizard', report['classes'])
            self.assertIn('bear', report['classes'])
            self.assertNotIn('monitor_lizard', report['classes'])
            self.assertNotIn('brown_bear', report['classes'])
            self.assertTrue({'pig', 'wild_boar'} <= set(report['classes']))
            self.assertEqual(report['resplit']['group_overlap'], 0)
            self.assertEqual(report['resplit']['class_independent_groups']['lizard'], 2)
            # Every old camera's three images must land in just one output split.
            for name in ('a', 'b'):
                found = set()
                for split in merger.SPLITS:
                    for p in (root / 'out/images' / split / 'lila').glob('*.jpg'):
                        if any(os.path.samefile(p, root / 'lila/images' / old / (name + '.jpg')) for old in merger.SPLITS):
                            found.add(split)
                self.assertEqual(len(found), 1)
            # Same command resumes with identical group assignments and no revalidation.
            again = merger.merge(root / 'lila', root / 'animals', root / 'out', workers=1, group_map=groups)
            self.assertEqual(again['audit']['resumed_output_files'], 12)
            self.assertEqual(report['stats'], again['stats'])
            self.assertIn('缺失集合', (root / 'out/MERGE_REPORT.md').read_text(encoding='utf-8'))

    def test_default_missing_lila_metadata_is_train_only_not_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.inputs(root)
            result = merger.merge(root / 'lila', root / 'animals', root / 'out')
            self.assertEqual(result['dataset_split_stats']['lila']['train']['images'], 6)
            self.assertEqual(result['dataset_split_stats']['lila']['test']['images'], 0)
            self.assertEqual(result['grouping']['animals_individual_image_groups'], 6)

    def test_no_safe_parent_kept_protected_pigs_and_alias_dedup(self):
        metadata = merger.class_metadata()
        coords = ('0.5', '0.5', '0.2', '0.2')
        records = [{'rows': [(n, coords) for n in ('monitor_lizard', 'lizard', 'jackal', 'wild_boar', 'pig', 'mink', 'weasel')], 'duplicate_boxes_removed': 0}]
        data = [{'mapping': {0: 'monitor_lizard', 1: 'jackal'}}]
        report = fold_rare(records, data, metadata, 100)
        names = [n for n, _ in records[0]['rows']]
        self.assertEqual(names.count('lizard'), 1)
        self.assertEqual(names.count('mustelid'), 1)
        self.assertIn('jackal', names)
        self.assertIn('wild_boar', names)
        self.assertIn('pig', names)
        self.assertIn('jackal', report['rare_without_safe_parent'])
        self.assertEqual(data[0]['mapping'][0], 'lizard')

    def test_metadata_camera_sequence_transitive_and_incomplete_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'lila').mkdir()
            selected = root / 'source/swg-camera-traps/selection'
            selected.mkdir(parents=True)
            images = [{'file_name': f'{i}.jpg', 'location': camera, 'seq_id': sequence}
                      for i, camera, sequence in [(0, 'A', 'x'), (1, 'B', 'x'), (2, 'B', 'y')]]
            records = []
            for image in images:
                stem = hashlib.sha256(('swg-camera-traps/' + image['file_name']).encode()).hexdigest()
                records.append({'dataset': 'lila', 'split': 'train', 'relative': 'swg-camera-traps/' + stem + '.jpg'})
            (selected / 'selected_annotations.json').write_text(json.dumps({'images': images}))
            datasets = [{'alias': 'lila', 'yaml': root / 'lila/data.yaml'}]
            report = attach_groups(records, datasets, root / 'source')
            self.assertFalse(report['lila_missing_group_images_train_only'])
            self.assertIn(('camera', 'swg-camera-traps', 'A'), records[0]['group_keys'])
            self.assertIn(('sequence', 'swg-camera-traps', 'x'), records[1]['group_keys'])
            for i, r in enumerate(records):
                r.update(rows=[('snake', ('0.5', '0.5', '0.2', '0.2'))],
                         image_stamp={'inode': i + 1, 'device': 1},
                         image_rel=Path('images/train') / r['relative'],
                         label_rel=Path('labels/train') / Path(r['relative']).with_suffix('.txt'))
            split_report = resplit(records, .05, .05, 42)
            self.assertEqual(split_report['groups'], 1)  # A-x-B-y is transitive.
            self.assertEqual({r['split'] for r in records}, {'train'})
            # Unknown camera in a dataset with known cameras forces all its groups together.
            images[2].pop('location')
            (selected / 'selected_annotations.json').write_text(json.dumps({'images': images}))
            fresh = [{k: v for k, v in r.items() if k in ('dataset', 'split', 'relative')} for r in records]
            report = attach_groups(fresh, datasets, root / 'source')
            self.assertEqual(report['lila_missing_group_images_train_only']['swg-camera-traps'], 3)
            self.assertTrue(all(r['train_only'] for r in fresh))

    def test_cross_original_split_identity_reassigned_instead_of_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = self.inputs(root)
            a = root / 'animals/images/train/a.jpg'
            b = root / 'animals/images/test/b.jpg'
            b.unlink()
            os.link(a, b)
            result = merger.merge(root / 'lila', root / 'animals', root / 'out', group_map=groups)
            self.assertEqual(result['excluded_images'], 0)
            splits = {s for s in merger.SPLITS for p in (root / 'out/images' / s / 'animals').glob('*.jpg') if os.path.samefile(a, p)}
            self.assertEqual(len(splits), 1)

    def test_changed_policy_refuses_old_output_without_destroying_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = self.inputs(root)
            merger.merge(root / 'lila', root / 'animals', root / 'out', group_map=groups)
            before = (root / 'out/data.yaml').read_bytes()
            with self.assertRaisesRegex(ValueError, 'policy changed'):
                merger.merge(root / 'lila', root / 'animals', root / 'out', group_map=groups, min_class_boxes=20)
            self.assertEqual((root / 'out/data.yaml').read_bytes(), before)

    def test_interrupted_resplit_reuses_written_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = self.inputs(root)
            original = merger.write_one
            count = 0

            def interrupt(record, output, ids):
                nonlocal count
                count += 1
                if count == 4:
                    raise KeyboardInterrupt()
                return original(record, output, ids)

            with patch.object(merger, 'write_one', interrupt), self.assertRaises(KeyboardInterrupt):
                merger.merge(root / 'lila', root / 'animals', root / 'out', workers=1, group_map=groups)
            self.assertFalse((root / 'out/data.yaml').exists())
            result = merger.merge(root / 'lila', root / 'animals', root / 'out', workers=1, group_map=groups)
            self.assertEqual(result['audit']['resumed_output_files'], 3)

    def test_optional_content_hash_groups_copied_images_after_resplit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = self.inputs(root)
            a = root / 'animals/images/train/a.jpg'
            b = root / 'animals/images/test/b.jpg'
            b.write_bytes(a.read_bytes())
            result = merger.merge(root / 'lila', root / 'animals', root / 'out', group_map=groups, check_content_hashes=True)
            self.assertEqual(result['excluded_images'], 0)
            splits = {s for s in merger.SPLITS for p in (root / 'out/images' / s / 'animals').glob('*.jpg') if p.read_bytes() == a.read_bytes()}
            self.assertEqual(len(splits), 1)

    def test_threshold_uses_images_not_boxes_strictly_less_than_limit(self):
        coords = ('0.5', '0.5', '0.2', '0.2')
        records = [{'rows': [('brown_bear', coords)] * 5 + ([('polar_bear', coords)] * 8 if i < 99 else []),
                    'duplicate_boxes_removed': 0} for i in range(100)]
        fold_rare(records, [], merger.class_metadata(), 100)
        self.assertEqual({n for r in records for n, _ in r['rows']}, {'brown_bear', 'bear'})


if __name__ == '__main__':
    unittest.main()
