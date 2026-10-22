import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_lila
import prepare_lila_audited as audited
from lila_compact import CompactState
from lila_state import check_label_precision, label_text
from tests.test_lila_audit import fixture


class YoloPrecisionTests(unittest.TestCase):
    def data(self, root, dimension='width'):
        original, selected = fixture(
            root / 'source', 'caltech-camera-traps', ['opossum']
        )
        box = [70, 60, 20, 20]
        box[2 if dimension == 'width' else 3] = 1e-8
        for path, image_id, ann_id in (
            (original, 'original_5', 9001),
            (selected, 5, 9002),
        ):
            doc = json.loads(path.read_text())
            doc['annotations'].append(
                {'id': ann_id, 'image_id': image_id, 'category_id': 4, 'bbox': box}
            )
            path.write_text(json.dumps(doc))

    def run_merge(self, root, mode='compact', **options):
        return audited.merge(
            root / 'source',
            root / 'out',
            datasets=['caltech-camera-traps'],
            workers=1,
            copy_mode='copy',
            checkpoint_mode=mode,
            **options
        )

    def test_rounded_zero_rejected_but_small_representable_boxes_preserved(self):
        for box in ((0, 0.5, 0.5, 1e-10, 0.2), (0, 0.5, 0.5, 0.2, 1e-10)):
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, 'zero width/height'):
                    check_label_precision([box])
        check_label_precision([(0, 0.5, 0.5, 1e-8, 1e-8)])
        check_label_precision([])

    def test_zero_after_serialization_excludes_whole_mixed_image(self):
        for dimension in ('width', 'height'):
            with self.subTest(
                dimension=dimension
            ), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.data(root, dimension)
                report = self.run_merge(root)
                self.assertEqual(report['audit']['images_verified'], 11)
                self.assertEqual(len(report['automatic_exclusions']), 1)
                self.assertEqual(
                    report['automatic_exclusions'][0]['file_name'], 'frame_5.png'
                )
                self.assertIn('8-decimal', report['automatic_exclusions'][0]['reason'])
                self.assertEqual(
                    sum(s['negative_images'] for s in report['stats'].values()), 1
                )
                self.assertTrue((root / 'out/AUDIT_REPORT.md').exists())
                self.assertTrue((root / 'out/data.yaml').exists())
                self.assertEqual(len(list((root / 'out/labels').rglob('*.txt'))), 11)

    def test_pre_fix_failed_output_resumes_using_existing_validation_checkpoints(self):
        for mode in ('compact', 'detailed'):
            with self.subTest(
                mode=mode
            ), tempfile.TemporaryDirectory() as temp, patch.object(
                CompactState, 'batch_size', 4
            ):
                root = Path(temp)
                self.data(root)
                # Reproduce the exact old path: row 2 is a tiny positive width rendered as zero.
                with patch.object(
                    prepare_lila, 'check_label_precision', return_value=None
                ):
                    with self.assertRaisesRegex(
                        ValueError, 'Invalid YOLO coordinates .*:2:'
                    ):
                        self.run_merge(root, mode)
                self.assertFalse((root / 'out/data.yaml').exists())
                bad_files = [
                    p
                    for p in (root / 'out/labels').rglob('*.txt')
                    if '0.00000000' in p.read_text()
                ]
                self.assertEqual(len(bad_files), 1)
                report = self.run_merge(root, mode)
                metric = 'image_decode_skipped' if mode == 'compact' else 'image_hits'
                self.assertEqual(report['cache'][metric], 12)
                self.assertEqual(report['audit']['images_verified'], 11)
                self.assertEqual(len(list((root / 'out/labels').rglob('*.txt'))), 11)
                self.assertEqual((root / 'out/audit_errors.jsonl').read_text(), '')
                self.assertTrue((root / 'out/AUDIT_REPORT.md').exists())
                self.assertTrue((root / 'out/data.yaml').exists())

    def test_strict_still_reports_invalid_input_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            with self.assertRaisesRegex(ValueError, '8-decimal'):
                self.run_merge(root, auto_clean=False)
            self.assertFalse((root / 'out/images').exists())
            self.assertFalse((root / 'out/data.yaml').exists())


if __name__ == '__main__':
    unittest.main()
