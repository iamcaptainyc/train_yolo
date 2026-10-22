import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from repair_images import decode, inspect_image


class RepairImagesTests(unittest.TestCase):
    def image(self, root, name='image.jpg'):
        path = root/name
        pixels = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', pixels)
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())
        return path

    def test_healthy_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.image(Path(tmp))
            original, mtime = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(inspect_image(path)['status'], 'healthy')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_native_warning_and_hardlink_safe_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.image(Path(tmp))
            original = path.read_bytes()
            damaged = original[:-2] + b'x' * 42 + original[-2:]
            path.write_bytes(damaged)
            sibling = Path(tmp)/'linked.jpg'
            os.link(path, sibling)
            _, warning = decode(path)
            self.assertIn('extraneous bytes', warning)
            result = inspect_image(path)
            self.assertEqual(result['status'], 'repaired', result)
            self.assertTrue(result['review_content'])
            pixels, warning = decode(path)
            self.assertEqual(pixels.shape, (64, 96, 3))
            self.assertFalse(warning)
            self.assertEqual(sibling.read_bytes(), damaged)
            self.assertNotEqual(path.stat().st_ino, sibling.stat().st_ino)
            self.assertEqual(inspect_image(path)['status'], 'healthy')

    def test_missing_end_marker_and_check_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.image(Path(tmp))
            original = path.read_bytes()[:-2]
            path.write_bytes(original)
            self.assertEqual(inspect_image(path, check_only=True)['status'], 'detected')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(inspect_image(path)['status'], 'repaired')
            self.assertEqual(inspect_image(path)['status'], 'healthy')

    def test_undecodable_image_kept_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.jpg'
            path.write_bytes(b'not a jpeg')
            result = inspect_image(path)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(path.read_bytes(), b'not a jpeg')

    def test_interrupt_drains_pending_images_and_resumes(self):
        from repair_images import scan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images'
            images.mkdir()
            for i in range(4):
                self.image(images, f'{i}.jpg')
            args = Namespace(roots=[images], output=root/'report', workers=1, restart=False, check_only=False)
            pool = Mock()
            calls = 0
            def submit(fn, *params):
                nonlocal calls
                calls += 1
                result = fn(*params)
                return Mock(result=Mock(side_effect=[KeyboardInterrupt(), result] if calls == 1 else [result]))
            pool.submit.side_effect = submit
            with patch('repair_images.ProcessPoolExecutor', return_value=pool):
                self.assertEqual(scan(args), 130)
            checkpoint = json.loads((args.output/'checkpoint.json').read_text())
            self.assertEqual(checkpoint['next_index'], 2)
            self.assertEqual(checkpoint['counts'], {'healthy': 2})
            with patch('repair_images.ProcessPoolExecutor', return_value=pool):
                self.assertEqual(scan(args), 0)
            checkpoint = json.loads((args.output/'checkpoint.json').read_text())
            self.assertEqual(checkpoint['next_index'], 4)
            self.assertEqual(checkpoint['counts'], {'healthy': 4})

    def test_cli_recursive_resume_only_problem_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images'/'train'
            images.mkdir(parents=True)
            self.image(images, 'healthy.jpg')
            broken = self.image(images, 'broken.jpg')
            broken.write_bytes(broken.read_bytes()[:-2])
            output = root/'report'
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve().parents[1]/'repair_images.py'),
                       str(root/'images'), '--workers', '2', '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((output/'checkpoint.json').read_text())
            self.assertEqual(state['next_index'], 2)
            self.assertEqual(state['counts'], {'repaired': 1, 'healthy': 1})
            records = (output/'issues.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])['path'], str(broken))
            before = broken.stat().st_mtime_ns
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Already complete', result.stdout)
            self.assertEqual(broken.stat().st_mtime_ns, before)
            self.assertIn('扫描完成', (output/'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
