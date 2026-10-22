import hashlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import prepare_lila
import prepare_lila_audited as audited
from lila_compact import CompactState
from tests import test_yolo_precision as precision


class ResumeWrittenTests(unittest.TestCase):
    data = precision.YoloPrecisionTests.data

    def run_merge(self, root, mode='compact', **options):
        return audited.merge(
            root / 'source',
            root / 'out',
            datasets=['caltech-camera-traps'],
            workers=1,
            copy_mode=options.pop('copy_mode', 'hardlink'),
            checkpoint_mode=mode,
            **options
        )

    def failed_written(self, root, mode='hardlink'):
        self.data(root)
        with patch.object(prepare_lila, 'check_label_precision', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Invalid YOLO coordinates'):
                self.run_merge(root, copy_mode=mode)
        return {
            p.relative_to(root / 'out').as_posix()
            for p in (root / 'out/images').rglob('*.png')
        }

    def test_direct_resume_never_decodes_hashes_resplits_or_rewrites_image_corpus(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            CompactState, 'batch_size', 4
        ):
            root = Path(temp)
            before = self.failed_written(root)
            original_digest = hashlib.file_digest

            def no_image_hash(stream, *args, **kwargs):
                if Path(stream.name).suffix == '.png':
                    raise AssertionError(
                        'Image content was hashed during output resume'
                    )
                return original_digest(stream, *args, **kwargs)

            output = io.StringIO()
            with patch.object(
                prepare_lila,
                'validate_record',
                side_effect=AssertionError('Full validation ran'),
            ), patch.object(
                prepare_lila,
                'split_records',
                side_effect=AssertionError('Splits changed'),
            ), patch.object(
                prepare_lila,
                'write_record',
                side_effect=AssertionError('Images rewritten'),
            ), patch.object(
                CompactState,
                'get',
                side_effect=AssertionError('Input SHA reconstruction ran'),
            ), patch.object(
                prepare_lila.Image, 'open', side_effect=AssertionError('Image decoded')
            ), patch.object(
                audited, 'preview'
            ), patch(
                'hashlib.file_digest', side_effect=no_image_hash
            ), redirect_stdout(
                output
            ):
                report = self.run_merge(root, resume_output=True)
            self.assertNotIn('Validate and hash:', output.getvalue())
            self.assertTrue(report['resumed_written_output'])
            self.assertEqual(report['cache']['input_image_hashes'], 0)
            self.assertEqual(report['cache']['input_image_decodes'], 0)
            self.assertEqual(report['cache']['output_hardlink_identities'], 11)
            self.assertEqual(report['audit']['images_verified'], 11)
            self.assertIsNone(report['audit']['unique_image_sha256'])
            self.assertFalse(report['sha256_overlap_checked'])
            self.assertTrue(report['prior_sha256_grouping_preserved'])
            after = {
                p.relative_to(root / 'out').as_posix()
                for p in (root / 'out/images').rglob('*.png')
            }
            self.assertLess(
                after, before
            )  # A subset: no retained image moved to another split.
            self.assertEqual(len(after), 11)
            self.assertTrue((root / 'out/AUDIT_REPORT.md').exists())
            self.assertTrue((root / 'out/data.yaml').exists())
            # A second output-only resume also preserves the grouping proof.
            again = self.run_merge(root, resume_output=True)
            self.assertEqual(again['audit']['images_verified'], 11)

    def test_changed_source_refuses_without_falling_back_to_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.failed_written(root)
            (root / 'source/caltech-camera-traps/images/frame_1.png').write_bytes(
                b'changed'
            )
            with patch.object(
                prepare_lila,
                'validate_record',
                side_effect=AssertionError('Fallback ran'),
            ):
                with self.assertRaisesRegex(
                    ValueError, 'refuses to run Validate and hash'
                ):
                    self.run_merge(root, resume_output=True)

    def test_incomplete_write_refuses_without_any_full_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.failed_written(root)
            path = root / 'out/report.json'
            report = json.loads(path.read_text())
            report.pop('image_storage_modes')
            path.write_text(json.dumps(report))
            with patch.object(
                prepare_lila,
                'validate_record',
                side_effect=AssertionError('Fallback ran'),
            ):
                with self.assertRaisesRegex(ValueError, 'Write YOLO did not finish'):
                    self.run_merge(root, resume_output=True)

    def test_copied_outputs_only_compare_unfinished_audit_tail(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            CompactState, 'batch_size', 4
        ):
            root = Path(temp)
            self.failed_written(root, 'copy')
            report = self.run_merge(root, copy_mode='copy', resume_output=True)
            self.assertEqual(report['cache']['output_copies_prior_audit_reused'], 4)
            self.assertEqual(report['cache']['output_copies_byte_compared'], 7)
            self.assertEqual(report['cache']['input_image_hashes'], 0)
            self.assertEqual(report['cache']['input_image_decodes'], 0)

    def test_missing_label_is_repaired_without_full_input_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.failed_written(root)
            label = next(
                p
                for p in (root / 'out/labels').rglob('*.txt')
                if '0.00000000' not in p.read_text()
            )
            label.unlink()
            report = self.run_merge(root, resume_output=True)
            self.assertEqual(report['cache']['output_labels_repaired'], 1)
            self.assertTrue(label.is_file())

    def test_previously_excluded_corrupt_inputs_are_not_reopened(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.data(root)
            (root / 'source/caltech-camera-traps/images/frame_8.png').write_bytes(
                b'broken image'
            )
            with patch.object(prepare_lila, 'check_label_precision', return_value=None):
                with self.assertRaisesRegex(ValueError, 'Invalid YOLO coordinates'):
                    self.run_merge(root)
            report = self.run_merge(root, resume_output=True)
            self.assertEqual(report['audit']['images_verified'], 10)
            self.assertEqual(len(report['automatic_exclusions']), 2)
            self.assertEqual(report['cache']['input_image_decodes'], 0)
            self.assertEqual(report['cache']['input_image_hashes'], 0)

    def test_cli_resume_reads_old_settings_from_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.failed_written(root)
            result = subprocess.run(
                [
                    sys.executable,
                    '-X',
                    'utf8',
                    str(Path(audited.__file__)),
                    '--output',
                    str(root / 'out'),
                    '--resume-output',
                    '--workers',
                    '1',
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('Validate and hash:', result.stdout + result.stderr)
            self.assertIn('Audit passed.', result.stdout)
            report = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(report['image_storage_modes'], {'hardlink': 11})
            self.assertTrue(report['resumed_written_output'])

    def test_recheck_flag_cannot_silently_enable_slow_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'cannot use --recheck-all'):
                self.run_merge(Path(temp), resume_output=True, recheck_all=True)


if __name__ == '__main__':
    unittest.main()
