import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from tests.test_lila_audit import fixture
import prepare_lila
import prepare_lila_audited as audited


class LilaResumeTests(unittest.TestCase):
    def setup_data(self, root):
        return fixture(root / 'source', 'caltech-camera-traps', ['opossum'])

    def run_merge(self, root, **options):
        return audited.merge(
            root / 'source',
            root / 'out',
            datasets=['caltech-camera-traps'],
            checkpoint_mode="detailed",
            copy_mode=options.pop('copy_mode', 'copy'),
            workers=options.pop('workers', 1),
            **options
        )

    def test_clean_rerun_reuses_validation_write_and_independent_audit(self):
        for mode in ('copy', 'hardlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.setup_data(root)
                self.run_merge(root, copy_mode=mode)
                before = (root / 'out/box_audit.jsonl').read_bytes()
                with patch(
                    'prepare_lila.yolo_boxes',
                    side_effect=AssertionError('geometry reran'),
                ), patch.object(
                    audited, 'audit_one', side_effect=AssertionError('audit reran')
                ):
                    report = self.run_merge(root, copy_mode=mode)
                self.assertEqual(report['cache']['image_hits'], 12)
                self.assertEqual(report['cache']['geometry_hits'], 12)
                self.assertEqual(report['cache']['write_hits'], 12)
                self.assertEqual(report['cache']['audit_hits'], 12)
                self.assertEqual(before, (root / 'out/box_audit.jsonl').read_bytes())

    def test_corrupt_image_dropped_then_only_repaired_image_revalidated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            image = root / 'source/caltech-camera-traps/images/frame_5.png'
            original = image.read_bytes()
            image.write_bytes(b'broken')
            report = self.run_merge(root)
            self.assertEqual(report['audit']['images_verified'], 11)
            self.assertEqual(len(report['automatic_exclusions']), 1)
            errors = [
                json.loads(x)
                for x in (root / 'out/validation_errors.jsonl').read_text().splitlines()
            ]
            self.assertEqual(errors[0]['file_name'], 'frame_5.png')
            self.assertTrue((root / 'out/AUDIT_REPORT.md').is_file())
            image.write_bytes(original)
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_checked'], 1)
            self.assertEqual(report['cache']['image_hits'], 11)
            self.assertEqual(report['audit']['images_verified'], 12)
            self.assertEqual((root / 'out/validation_errors.jsonl').read_text(), '')
            self.assertEqual(len(list((root / 'out/images').rglob('*.png'))), 12)

    def test_completed_output_then_corruption_removes_owned_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root)
            (root / 'source/caltech-camera-traps/images/frame_5.png').write_bytes(
                b'broken'
            )
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_hits'], 11)
            self.assertEqual(report['cache']['image_checked'], 1)
            self.assertEqual(len(list((root / 'out/images').rglob('*.png'))), 11)
            self.assertEqual(len(list((root / 'out/labels').rglob('*.txt'))), 11)

    def test_interrupted_validation_resumes_committed_files(self):
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
            self.assertFalse((root / 'out/data.yaml').exists())
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_hits'], 5)
            self.assertEqual(report['cache']['image_checked'], 7)

    def test_strict_clipping_failure_can_resume_without_image_redecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = self.setup_data(root)
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if 'bbox' in ann:
                        ann['bbox'] = [12, 9, 108.5, 45]
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'edge clipping'):
                self.run_merge(root, auto_clean=False, allow_clipping=False)
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_hits'], 12)
            self.assertEqual(report['clipped_boxes_within_one_pixel'], 11)

    def test_bad_bbox_excludes_whole_image_and_clips_small_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = self.setup_data(root)
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if ann.get('bbox') and ann['image_id'] in (1, 'original_1'):
                        ann['bbox'] = [12, 9, 109.5, 45]
                    elif ann.get('bbox') and ann['image_id'] in (2, 'original_2'):
                        ann['bbox'] = [12, 9, 108.5, 45]
                path.write_text(json.dumps(data))
            report = self.run_merge(root)
            self.assertEqual(report['audit']['images_verified'], 11)
            self.assertEqual(report['clipped_boxes_within_one_pixel'], 1)
            rows = [
                json.loads(x)
                for x in (root / 'out/box_audit.jsonl').read_text().splitlines()
            ]
            self.assertNotIn('frame_1.png', [r['source_file_name'] for r in rows])
            self.assertEqual(sum(r['role'] == 'negative' for r in rows), 1)

    def test_errors_not_limited_to_twenty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            fixture(root / 'source', 'ena24', ['american crow'])
            for image in (root / 'source').rglob('images/*.png'):
                image.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'No valid positive'):
                audited.merge(
                    root / 'source',
                    root / 'out',
                    datasets=['caltech-camera-traps', 'ena24'],
                )
            self.assertEqual(
                len((root / 'out/validation_errors.jsonl').read_text().splitlines()), 24
            )
            self.assertFalse((root / 'out/data.yaml').exists())

    def test_changed_output_label_repaired_and_reaudited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root)
            label = next((root / 'out/labels').rglob('*.txt'))
            label.write_text('invalid')
            report = self.run_merge(root)
            self.assertEqual(report['cache']['write_hits'], 11)
            self.assertEqual(report['cache']['audit_checked'], 1)
            self.assertNotEqual(label.read_text(), 'invalid')

    def test_force_recheck_and_immutable_split_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            self.run_merge(root)
            report = self.run_merge(root, recheck_all=True)
            self.assertEqual(report['cache']['image_checked'], 12)
            self.assertEqual(report['cache']['audit_checked'], 12)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                self.run_merge(root, seed=23)

    def test_interrupted_write_keeps_committed_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            original = prepare_lila.write_record
            completed = []

            def interrupted(record, **options):
                if len(completed) == 5:
                    raise KeyboardInterrupt()
                result = original(record, **options)
                completed.append(record['name'])
                return result

            with patch.object(prepare_lila, 'write_record', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_merge(root)
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_hits'], 12)
            self.assertEqual(report['cache']['write_hits'], 5)
            self.assertEqual(len(list((root / 'out/images').rglob('*.png'))), 12)

    def test_interrupted_audit_only_rechecks_remaining_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            original = audited.audit_one
            completed = []

            def interrupted(record, *args):
                if len(completed) == 5:
                    raise KeyboardInterrupt()
                result = original(record, *args)
                completed.append(record['name'])
                return result

            with patch.object(audited, 'audit_one', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_merge(root)
            self.assertFalse((root / 'out/data.yaml').exists())
            report = self.run_merge(root)
            self.assertEqual(report['cache']['image_hits'], 12)
            self.assertEqual(report['cache']['audit_hits'], 5)
            self.assertEqual(report['cache']['audit_checked'], 7)

    def test_identical_images_conflicting_boxes_exclude_both(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, selected = self.setup_data(root)
            images = root / 'source/caltech-camera-traps/images'
            (images / 'frame_2.png').write_bytes((images / 'frame_1.png').read_bytes())
            for path in (original, selected):
                data = json.loads(path.read_text())
                for ann in data['annotations']:
                    if ann.get('bbox') and ann['image_id'] in (2, 'original_2'):
                        ann['bbox'] = [20, 9, 36, 45]
                path.write_text(json.dumps(data))
            report = self.run_merge(root)
            self.assertEqual(report['audit']['images_verified'], 10)
            self.assertEqual(len(report['automatic_exclusions']), 2)
            self.assertEqual(
                len((root / 'out/validation_errors.jsonl').read_text().splitlines()), 2
            )

    def test_unknown_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_data(root)
            (root / 'out').mkdir()
            (root / 'out/unknown.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'no managed checkpoint'):
                self.run_merge(root)
            self.assertEqual((root / 'out/unknown.txt').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
