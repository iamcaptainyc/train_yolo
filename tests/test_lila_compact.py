import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_lila
import prepare_lila_audited as audited
from lila_compact import CompactState
from tests.test_lila_audit import fixture


class CompactCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.small_batches = patch.object(CompactState, 'batch_size', 4)
        self.small_batches.start()
        self.addCleanup(self.small_batches.stop)

    def setup_data(self, root):
        return fixture(root / 'source', 'caltech-camera-traps', ['opossum'])

    def run_merge(self, root, **options):
        return audited.merge(
            root / 'source',
            root / 'out',
            datasets=['caltech-camera-traps'],
            copy_mode=options.pop('copy_mode', 'copy'),
            workers=options.pop('workers', 1),
            **options
        )

    def test_default_has_no_per_image_database_logs_or_hashes(self):
        for mode in ('copy', 'hardlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.setup_data(root)
                first = self.run_merge(root, copy_mode=mode)
                second = self.run_merge(root, copy_mode=mode, workers=4)
                self.assertEqual(second['checkpoint_mode'], 'compact')
                self.assertEqual(second['cache']['image_decode_skipped'], 12)
                self.assertEqual(second['cache']['audit_hash_skipped'], 12)
                for name in (
                    '.lila-state/cache.sqlite3',
                    'box_audit.jsonl',
                    'split_manifest.jsonl',
                ):
                    self.assertFalse((root / 'out' / name).exists(), name)
                progress = (root / 'out/.lila-state/progress.json').read_text()
                self.assertLess(len(progress), 2000)
                self.assertNotIn('frame_', progress)
                self.assertNotIn('labels', progress)
                self.assertEqual(len(json.loads(progress)['stages']['validation']), 3)
                self.assertEqual((root / 'out/validation_errors.jsonl').read_text(), '')
                self.assertEqual(first['stats'], second['stats'])
                markdown = (root / 'out/AUDIT_REPORT.md').read_text(encoding='utf-8')
                self.assertNotIn('[box_audit.jsonl]', markdown)

    def test_interrupt_skips_completed_batch_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            original = prepare_lila.validate_record
            calls = []

            def interrupted(record, **options):
                if len(calls) == 5:
                    raise KeyboardInterrupt()
                calls.append(record['name'])
                return original(record, **options)

            with patch.object(prepare_lila, 'validate_record', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_merge(root)
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_decode_skipped'], 4)
            self.assertEqual(report['cache']['image_checked'], 8)

    def test_changed_image_only_invalidates_its_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root)
            image = root / 'source/caltech-camera-traps/images/frame_5.png'
            image.write_bytes(b'broken')
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_decode_skipped'], 8)
            self.assertEqual(report['cache']['image_checked'], 4)
            self.assertEqual(report['audit']['images_verified'], 11)
            self.assertEqual(len(list((root / 'out/images').rglob('*.png'))), 11)
            self.assertEqual(
                len((root / 'out/validation_errors.jsonl').read_text().splitlines()), 1
            )
            # Failed image retried, successful images in the same completed batch skipped.
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_decode_skipped'], 11)
            self.assertEqual(report['cache']['image_checked'], 1)

    def test_interrupted_audit_skips_output_hashes_for_completed_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            original = audited.audit_one
            completed = []

            def interrupted(record, *args, **kwargs):
                if len(completed) == 5:
                    raise KeyboardInterrupt()
                result = original(record, *args, **kwargs)
                completed.append(record['name'])
                return result

            with patch.object(audited, 'audit_one', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_merge(root)
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_decode_skipped'], 12)
            self.assertEqual(report['cache']['audit_hash_skipped'], 4)
            self.assertEqual(report['cache']['audit_checked'], 8)

    def test_upgrading_detailed_output_removes_owned_full_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root, checkpoint_mode='detailed')
            self.assertTrue((root / 'out/.lila-state/cache.sqlite3').exists())
            migrated = self.run_merge(root)
            self.assertEqual(migrated['cache']['image_decode_skipped'], 12)
            self.assertFalse((root / 'out/.lila-state/cache.sqlite3').exists())
            self.assertFalse((root / 'out/box_audit.jsonl').exists())
            self.assertFalse((root / 'out/split_manifest.jsonl').exists())

    def test_training_label_cache_does_not_block_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root)
            cache = root / 'out/labels/train.cache'
            cache.write_bytes(b'training cache')
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_decode_skipped'], 12)
            self.assertFalse(cache.exists())

    def test_repaired_files_and_splits_still_validate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = self.setup_data(root)
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if ann.get('bbox') and ann['image_id'] in (1, 'original_1'):
                        ann['bbox'] = [12, 9, 108.5, 45]
                path.write_text(json.dumps(data))
            report = self.run_merge(root)
            repairs = [
                json.loads(line)
                for line in (root / 'out/repairs.jsonl').read_text().splitlines()
            ]
            self.assertEqual(len(repairs), 1)
            self.assertEqual(repairs[0]['file_name'], 'frame_1.png')
            self.assertEqual(repairs[0]['corrected_boxes'], 1)
            self.assertNotIn('boxes', repairs[0])
            self.assertEqual(report['clipped_boxes_within_one_pixel'], 1)
            self.assertTrue(
                all(
                    n == 0
                    for pair in report['cross_split_overlap'].values()
                    for n in pair.values()
                )
            )


if __name__ == '__main__':
    unittest.main()
