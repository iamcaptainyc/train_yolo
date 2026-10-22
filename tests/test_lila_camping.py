import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import yaml

from lila_camping import camping_mapping, load_policy
import prepare_lila
import prepare_lila_audited as audited
from tests.test_lila_audit import fixture


class CampingPolicyTests(unittest.TestCase):
    def test_every_reviewed_source_name_is_covered_without_latin_targets(self):
        policy = load_policy()
        inventories = {
            ds: {'selected_categories': [
                {'id': i, 'name': c['source_name'], 'boxes': 1}
                for i, c in enumerate(rows)
            ]} for ds, rows in policy['datasets'].items()
        }
        mapping, details = camping_mapping(inventories)
        self.assertEqual(len(details), 258)
        self.assertTrue(all(v is None or ':' not in v and ' ' not in v for v in mapping.values()))
        for d in details:
            if any(word in d['source_name'].replace('_', ' ') for word in ('unidentified animal', 'unidentified mammal', 'small mammal', 'small animal', 'other animal')):
                self.assertIsNone(d['target'], d)

    def test_wild_pigs_are_unified_and_unspecified_pig_stays_separate(self):
        inventories = {
            'swg-camera-traps': {'selected_categories': [{'id': 12, 'name': 'eurasian_wild_pig', 'boxes': 1}]},
            'nkhotakota-camera-traps': {'selected_categories': [
                {'id': 6, 'name': 'bushpig', 'boxes': 1},
                {'id': 14, 'name': 'common_warthog', 'boxes': 1},
            ]},
            'caltech-camera-traps': {'selected_categories': [{'id': 39, 'name': 'pig', 'boxes': 1}]},
        }
        mapping, _ = camping_mapping(inventories)
        self.assertEqual(mapping['swg-camera-traps:12'], 'wild_boar')
        self.assertEqual(mapping['nkhotakota-camera-traps:6'], 'wild_boar')
        self.assertEqual(mapping['caltech-camera-traps:39'], 'pig')
        self.assertEqual(mapping['nkhotakota-camera-traps:14'], 'warthog')

    def test_unreviewed_name_never_guesses_a_species(self):
        with self.assertRaisesRegex(ValueError, 'No reviewed camping mapping'):
            camping_mapping({'ena24': {'selected_categories': [{'id': 999, 'name': 'unknown new species', 'boxes': 1}]}})


class CampingRemapTests(unittest.TestCase):
    datasets = ['caltech-camera-traps', 'ena24']

    def old(self, root):
        fixture(root / 'source', self.datasets[0], ['rodent', 'squirrel', 'opossum', 'fox'])
        fixture(root / 'source', self.datasets[1], ['Bird', 'American Crow', 'American Black Bear'])
        return audited.merge(root / 'source', root / 'old', datasets=self.datasets, copy_mode='hardlink', workers=1)

    def remap(self, root):
        return audited.merge(
            root / 'source', root / 'new', datasets=self.datasets,
            copy_mode='hardlink', workers=1, category_policy='camping', remap_from=root / 'old',
        )

    def test_original_fine_classes_restored_without_decoding_hashing_or_resplitting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = self.old(root)
            self.assertIn('class:aves', old['classes'])
            self.assertIn('order:rodentia', old['classes'])
            before = {p.relative_to(root / 'old'): p.read_bytes() for p in (root / 'old').rglob('*') if p.is_file() and p.suffix != '.png'}
            positions = {p.name: p.relative_to(root / 'old/images').parts[0] for p in (root / 'old/images').rglob('*.png')}
            file_digest = hashlib.file_digest

            def no_image_hash(stream, *args, **kwargs):
                if Path(stream.name).suffix == '.png':
                    raise AssertionError('Image corpus hashing ran')
                return file_digest(stream, *args, **kwargs)

            with patch.object(prepare_lila, 'validate_record', side_effect=AssertionError('Full validation ran')), patch.object(prepare_lila, 'split_records', side_effect=AssertionError('Resplit ran')), patch.object(prepare_lila.Image, 'open', side_effect=AssertionError('Image opened')), patch.object(audited, 'preview'), patch('hashlib.file_digest', side_effect=no_image_hash):
                report = self.remap(root)
            self.assertTrue({'bird', 'crow', 'rodent', 'squirrel', 'black_bear'} <= set(report['classes']))
            self.assertEqual(report['cache']['input_image_decodes'], 0)
            self.assertEqual(report['cache']['input_image_hashes'], 0)
            self.assertEqual(report['audit']['images_verified'], 24)
            for image in (root / 'new/images').rglob('*.png'):
                relative = image.relative_to(root / 'new/images')
                self.assertEqual(relative.parts[0], positions[image.name])
                self.assertTrue(os.path.samefile(image, root / 'old/images' / relative))
            for relative, value in before.items():
                self.assertEqual((root / 'old' / relative).read_bytes(), value)
            config = yaml.safe_load((root / 'new/data.yaml').read_text())
            self.assertEqual(set(config['names'].values()), set(report['classes']))
            self.assertIn('black_bear', config['names'].values())
            text = (root / 'new/AUDIT_REPORT.md').read_text(encoding='utf-8')
            self.assertIn('中文名称', text)
            self.assertIn('wild_boar', text)
            self.assertNotIn('子类整体上卷', text)
            self.assertTrue((root / 'new/class_hierarchy.json').is_file())

    def test_changed_source_refuses_without_a_slow_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.old(root)
            image = next((root / 'source/caltech-camera-traps/images').glob('*.png'))
            image.write_bytes(b'changed')
            with patch.object(prepare_lila, 'validate_record', side_effect=AssertionError('Fallback ran')):
                with self.assertRaisesRegex(ValueError, 'attributes or annotations changed'):
                    self.remap(root)
            self.assertFalse((root / 'new/data.yaml').exists())

    def test_cli_restores_source_and_split_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.old(root)
            result = subprocess.run([
                sys.executable, '-X', 'utf8', str(Path(audited.__file__)),
                '--remap-from', str(root / 'old'), '--output', str(root / 'new'), '--workers', '1',
            ], capture_output=True, text=True, encoding='utf-8', timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('Validate and hash:', result.stdout + result.stderr)
            report = json.loads((root / 'new/report.json').read_text())
            self.assertIn('black_bear', report['classes'])
            self.assertEqual(report['image_storage_modes'], {'hardlink': 24})

    def test_tampered_old_label_refuses_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.old(root)
            label = next(p for p in (root / 'old/labels').rglob('*.txt') if p.read_text())
            label.write_text('999 0.5 0.5 0.1 0.1\n')
            with self.assertRaisesRegex(ValueError, 'Prior label differs'):
                self.remap(root)
            self.assertTrue(label.read_text().startswith('999 '))
            self.assertFalse((root / 'new/data.yaml').exists())

    def test_interrupted_write_resumes_and_does_not_mutate_old_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.old(root)
            original = prepare_lila.write_record
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise KeyboardInterrupt()
                return original(*args, **kwargs)

            with patch.object(prepare_lila, 'write_record', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.remap(root)
            self.assertFalse((root / 'new/data.yaml').exists())
            self.assertTrue((root / 'old/data.yaml').exists())
            with patch.object(prepare_lila, 'validate_record', side_effect=AssertionError('Full validation ran')):
                report = self.remap(root)
            self.assertEqual(report['audit']['images_verified'], 24)
            self.assertGreaterEqual(report['cache']['write_hits'], 4)

    def test_fresh_camping_excludes_entire_uncertain_images_and_keeps_english_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'source', 'swg-camera-traps', ['unidentified_small_mammal', 'leopard_cat', 'eurasian_wild_pig'])
            report = audited.merge(root / 'source', root / 'new', datasets=['swg-camera-traps'], category_policy='camping', copy_mode='hardlink', workers=1)
            self.assertEqual(set(report['classes']), {'leopard_cat', 'wild_boar'})
            self.assertGreater(report['excluded_images']['swg-camera-traps:excluded_category_images'], 0)
            self.assertEqual(report['audit']['excluded_images'] + report['audit']['images_verified'], 12)

    def test_fine_class_with_no_training_group_is_excluded_without_leaking_or_erasing_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = fixture(root / 'source', 'ena24', ['Bird', 'American Crow'])
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if ann.get('bbox'):
                        ann['category_id'] = 4
                path.write_text(json.dumps(data))
            audited.merge(root / 'source', root / 'probe', datasets=['ena24'], copy_mode='hardlink', workers=1)
            validation_image = next((root / 'probe/images/val/ena24').glob('*.png'))
            index = next(i for i in range(1, 12) if hashlib.sha256(f'ena24/frame_{i}.png'.encode()).hexdigest() == validation_image.stem)
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if str(ann['image_id']) in (str(index), f'original_{index}'):
                        ann['category_id'] = 5
                path.write_text(json.dumps(data))
            audited.merge(root / 'source', root / 'old', datasets=['ena24'], copy_mode='hardlink', workers=1)
            result = audited.merge(root / 'source', root / 'new', datasets=['ena24'], category_policy='camping', remap_from=root / 'old', copy_mode='hardlink', workers=1)
            self.assertEqual(result['classes'], ['bird'])
            self.assertEqual(result['untrained_classes_excluded'], ['crow'])
            self.assertEqual(sum(s['images'] for s in result['stats'].values()), 11)
            self.assertFalse((root / 'new/images/val/ena24' / validation_image.name).exists())
            self.assertEqual(len(result['automatic_exclusions']), 1)


if __name__ == '__main__':
    unittest.main()
