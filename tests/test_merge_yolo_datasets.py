import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

import merge_yolo_datasets as merger


def fixture(root, names, tag):
    for split in merger.SPLITS:
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        for name in ('a', 'b'):
            # Deliberately need no image decoder; inputs are treated as prevalidated.
            (root / 'images' / split / (name + '.jpg')).write_bytes(f'{tag}:{split}:{name}'.encode())
            lines = [f'{i} 0.50000000 5e-1 0.20000000 0.20000000' for i in range(len(names))]
            (root / 'labels' / split / (name + '.txt')).write_text('\n'.join(lines) + '\n')
    (root / 'data.yaml').write_text(yaml.safe_dump({'path': str(root), **{s: 'images/' + s for s in merger.SPLITS}, 'names': dict(enumerate(names))}), encoding='utf-8')


class MergeYoloTests(unittest.TestCase):
    def data(self, root):
        fixture(root / 'lila', ['wild_boar', 'pig', 'black_bear', 'squirrel'], 'lila')
        fixture(root / 'animals', ['brown bear', 'bull', 'pig', 'squirrel', 'moths and butterflies', 'butterfly'], 'animals')

    def run_merge(self, root, **kwargs):
        return merger.merge(root / 'lila', root / 'animals', root / 'out', workers=kwargs.pop('workers', 1), **kwargs)

    def test_unified_ids_original_splits_coordinate_tokens_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            before = {p: p.read_bytes() for alias in ('lila', 'animals') for p in (root / alias).rglob('*.txt')}
            with patch('hashlib.file_digest', side_effect=AssertionError('Image hashing ran')), patch('PIL.Image.open', side_effect=AssertionError('Image decoding ran')), patch('shutil.copy2', side_effect=AssertionError('Image copied')):
                report = self.run_merge(root, workers=3)
            self.assertTrue({'black_bear', 'brown_bear', 'cattle', 'pig', 'wild_boar', 'insect', 'butterfly'} <= set(report['classes']))
            self.assertNotIn('bull', report['classes'])
            self.assertEqual(report['audit']['images_verified'], 12)
            self.assertEqual(report['audit']['image_content_hashes'], 0)
            self.assertFalse(report['audit']['independent_copied_image_duplicates_checked'])
            config = yaml.safe_load((root / 'out/data.yaml').read_text())
            self.assertEqual(list(config['names'].values()), report['classes'])
            for ds in ('lila', 'animals'):
                for split in merger.SPLITS:
                    outputs = list((root / 'out/images' / split / ds).glob('*.jpg'))
                    self.assertEqual(len(outputs), 2)
                    for image in outputs:
                        self.assertTrue(any(os.path.samefile(image, p) for p in (root / ds / 'images' / split).glob('*.jpg')))
                        label = root / 'out/labels' / split / ds / (image.stem + '.txt')
                        self.assertTrue(all(line.split()[1:] == ['0.50000000', '5e-1', '0.20000000', '0.20000000'] for line in label.read_text().splitlines()))
            self.assertTrue(all(p.read_bytes() == text for p, text in before.items()))
            self.assertIn('未检查独立复制文件', (root / 'out/MERGE_REPORT.md').read_text(encoding='utf-8'))

    def test_negative_is_preserved_but_missing_label_and_bad_coordinates_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            (root / 'lila/labels/train/a.txt').write_text('')
            (root / 'animals/labels/val/a.txt').unlink()
            (root / 'animals/labels/test/a.txt').write_text('0 nan 0.5 0.2 0.2\n')
            report = self.run_merge(root)
            self.assertEqual(report['excluded_images'], 2)
            self.assertEqual(report['stats']['train']['negative_images'], 1)
            self.assertEqual(report['audit']['images_verified'], 10)
            errors = (root / 'out/excluded_images.jsonl').read_text()
            self.assertIn('missing label', errors)
            self.assertIn('invalid normalized', errors)

    def test_cross_split_file_identity_is_excluded_without_moving_any_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            target = root / 'animals/images/val/a.jpg'
            target.unlink()
            os.link(root / 'lila/images/train/a.jpg', target)
            report = self.run_merge(root)
            self.assertEqual(report['excluded_images'], 2)
            self.assertEqual(report['audit']['images_verified'], 10)
            self.assertEqual(report['stats']['train']['images'], 3)
            self.assertEqual(report['stats']['val']['images'], 3)

    def test_optional_hash_check_finds_independent_copies_across_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            target = root / 'animals/images/test/a.jpg'
            target.write_bytes((root / 'lila/images/train/a.jpg').read_bytes())
            report = self.run_merge(root, check_content_hashes=True)
            self.assertEqual(report['excluded_images'], 2)
            self.assertTrue(report['audit']['independent_copied_image_duplicates_checked'])
            self.assertGreater(report['audit']['image_content_hashes'], 0)

    def test_interrupt_and_resume_reuses_files_without_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            original = merger.write_one
            count = 0

            def interrupted(*args):
                nonlocal count
                count += 1
                if count == 5:
                    raise KeyboardInterrupt()
                return original(*args)

            with patch.object(merger, 'write_one', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_merge(root)
            self.assertFalse((root / 'out/data.yaml').exists())
            with patch('hashlib.file_digest', side_effect=AssertionError('Hashed on resume')):
                result = self.run_merge(root)
            self.assertEqual(result['audit']['images_verified'], 12)
            self.assertGreaterEqual(result['audit']['resumed_output_files'], 4)

    def test_each_image_write_does_not_scan_directories_or_repeat_mkdir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            original = merger.write_one

            def no_directory_work(*args):
                with patch('os.scandir', side_effect=AssertionError('Per-image directory scan')), patch.object(Path, 'mkdir', side_effect=AssertionError('Per-image mkdir')):
                    return original(*args)

            with patch.object(merger, 'write_one', side_effect=no_directory_work):
                result = self.run_merge(root)
            self.assertEqual(result['audit']['images_verified'], 12)

    def test_resume_cleans_owned_temp_files_once_per_label_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            self.run_merge(root)
            label = next((root / 'out/labels/train/lila').glob('*.txt'))
            leftover = label.with_name('.' + label.name + '.' + 'a' * 32 + '.tmp')
            unrelated = label.with_name('.user-notes.tmp')
            leftover.write_text('incomplete')
            unrelated.write_text('not merger-owned')
            records = [
                {'image_rel': image.relative_to(root / 'out'), 'label_rel': merger.label_for(image).relative_to(root / 'out')}
                for image in (root / 'out/images').rglob('*.jpg')
            ]
            scandir = os.scandir
            with patch('os.scandir', wraps=scandir) as scans:
                removed = merger.prepare_output_directories(records, root / 'out')
            self.assertEqual(scans.call_count, 6)
            self.assertEqual(removed, 1)
            self.assertFalse(leftover.exists())
            self.assertEqual(unrelated.read_text(), 'not merger-owned')
            unrelated.unlink()  # Output inventory correctly rejects unrelated files.
            stable_label_temp = label.with_name(label.name + '.merge-part')
            image = root / 'out/images/train/lila' / (label.stem + '.jpg')
            stable_image_temp = image.with_name(image.name + '.merge-part')
            stable_label_temp.write_text('left by a killed process')
            stable_image_temp.write_bytes(b'left by a killed process')
            result = self.run_merge(root)
            self.assertEqual(result['audit']['resumed_output_files'], 12)
            self.assertFalse(stable_label_temp.exists())
            self.assertFalse(stable_image_temp.exists())

    def test_hardlink_failure_has_no_copy_fallback_and_no_yaml(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            with patch('os.link', side_effect=OSError(errno.EXDEV, 'different disks')), patch('shutil.copy2', side_effect=AssertionError('Copied')):
                with self.assertRaises(OSError):
                    self.run_merge(root)
            self.assertFalse((root / 'out/data.yaml').exists())
            self.assertTrue((root / 'out/PROCESSING_REPORT.md').exists())

    def test_changed_source_labels_cannot_reuse_old_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            self.run_merge(root)
            label = root / 'lila/labels/train/a.txt'
            label.write_text(label.read_text().replace('0.20000000', '0.25000000'))
            with self.assertRaisesRegex(ValueError, 'Source files/labels or exclusions changed'):
                self.run_merge(root)
            self.assertFalse((root / 'out/data.yaml').exists())
            with self.assertRaisesRegex(ValueError, 'Source files/labels or exclusions changed'):
                self.run_merge(root)

    def test_deleting_input_link_does_not_delete_merged_image_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            self.run_merge(root)
            image = next((root / 'out/images/train/lila').glob('*.jpg'))
            original = next(p for p in (root / 'lila/images/train').glob('*.jpg') if os.path.samefile(p, image))
            content = image.read_bytes()
            original.unlink()
            self.assertEqual(image.read_bytes(), content)

    def test_same_split_conflicting_labels_exclude_both_whole_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            target = root / 'animals/images/val/a.jpg'
            target.unlink()
            os.link(root / 'lila/images/val/a.jpg', target)
            result = self.run_merge(root)
            self.assertEqual(result['excluded_images'], 2)
            self.assertIn('conflicting unified labels', (root / 'out/excluded_images.jsonl').read_text())

    def test_aliases_that_collapse_to_identical_class_and_box_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'lila', ['black_bear'], 'lila')
            fixture(root / 'animals', ['bull', 'cattle'], 'animals')
            result = self.run_merge(root)
            self.assertEqual(result['audit']['duplicate_box_records_removed'], 6)
            self.assertEqual(result['audit']['boxes_verified'], 12)
            for label in (root / 'out/labels/train/animals').glob('*.txt'):
                self.assertEqual(len(label.read_text().splitlines()), 1)

    def test_all_original_80_classes_are_supported_and_bear_species_stay_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / 'lila', ['black_bear'], 'lila')
            fixture(root / 'animals', merger.ANIMALS_NAMES, 'animals')
            result = self.run_merge(root)
            self.assertTrue({'bear', 'brown_bear', 'polar_bear', 'black_bear'} <= set(result['classes']))
            rows = {r['source_id']: r for r in result['source_class_mapping']['animals']}
            self.assertEqual(rows[0]['target'], 'bear')
            self.assertEqual(rows[1]['target'], 'brown_bear')
            self.assertEqual(rows[50]['target'], 'polar_bear')
            self.assertEqual(len({rows[i]['target_id'] for i in (0, 1, 50)}), 3)
            self.assertEqual(len(rows), 80)

    def test_rejects_old_scientific_lila_class_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            config = root / 'lila/data.yaml'
            data = yaml.safe_load(config.read_text())
            data['names'][0] = 'species:sus scrofa'
            config.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, 'old scientific LILA'):
                self.run_merge(root)

    def test_cli_and_directory_or_yaml_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            result = subprocess.run([sys.executable, '-X', 'utf8', str(Path(merger.__file__)), '--lila', str(root / 'lila/data.yaml'), '--animals', str(root / 'animals'), '--output', str(root / 'out'), '--workers', '2'], capture_output=True, text=True, encoding='utf-8', timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'out/data.yaml').exists())


if __name__ == '__main__':
    unittest.main()
