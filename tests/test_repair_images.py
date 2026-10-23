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
            self.assertEqual(sibling.read_bytes(), path.read_bytes())
            self.assertEqual(path.stat().st_ino, sibling.stat().st_ino)
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

    def test_undecodable_image_deleted_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.jpg'
            path.write_bytes(b'not a jpeg')
            result = inspect_image(path)
            self.assertEqual(result['status'], 'deleted')
            self.assertFalse(path.exists())

    def test_deletion_label_report_counts_and_resume(self):
        import yaml
        from repair_images import scan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'images/train/animals'
            labels = root/'labels/train/animals'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            self.image(images, 'healthy.jpg')
            (images/'bad.jpg').write_bytes(b'broken')
            (labels/'healthy.txt').write_text('0 .5 .5 .2 .2\n0 .3 .3 .1 .1\n')
            (labels/'bad.txt').write_text('1 .5 .5 .2 .2\n')
            (root/'data.yaml').write_text(yaml.safe_dump({'names': ['cat','dog']}))
            (root/'SUBSET_REPORT.md').write_text('OLD COUNT 2')
            (root/'labels/train.cache').write_bytes(b'old cache')
            args = Namespace(roots=[root/'images'], output=root/'repair', workers=2, restart=False, check_only=False)
            pool = Mock()
            pool.submit.side_effect = lambda fn,*params: Mock(result=Mock(return_value=fn(*params)))
            with patch('repair_images.ProcessPoolExecutor',return_value=pool):
                self.assertEqual(scan(args),0)
                self.assertEqual(scan(args),0)
            self.assertFalse((images/'bad.jpg').exists())
            self.assertFalse((labels/'bad.txt').exists())
            self.assertFalse((root/'labels/train.cache').exists())
            report = json.loads((root/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['stats']['train']['images'],1)
            self.assertEqual(report['stats']['train']['class_images'],{'cat':1})
            self.assertEqual(report['stats']['train']['boxes'],{'cat':2})
            self.assertEqual(report['image_repair']['deleted_labels_this_operation_log'],1)
            self.assertEqual(report['excluded_images'],1)
            self.assertIn('OLD COUNT 2',(root/'SUBSET_REPORT.before_image_repair.md').read_text())
            self.assertNotIn('OLD COUNT 2',(root/'SUBSET_REPORT.md').read_text(encoding='utf-8'))

    def test_interrupt_after_deletion_resumes_changed_path_list(self):
        from repair_images import scan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images=root/'images/train'; images.mkdir(parents=True)
            labels=root/'labels/train'; labels.mkdir(parents=True)
            (images/'0.jpg').write_bytes(b'bad')
            (labels/'0.txt').write_text('0 .5 .5 .1 .1')
            for i in range(1,4):
                self.image(images,f'{i}.jpg')
            args=Namespace(roots=[root/'images'],output=root/'repair',workers=1,restart=False,check_only=False)
            calls=0
            def submit(fn,*params):
                nonlocal calls
                calls+=1
                result=fn(*params)
                return Mock(result=Mock(side_effect=[KeyboardInterrupt(),result] if calls==1 else [result]))
            pool=Mock(); pool.submit.side_effect=submit
            with patch('repair_images.ProcessPoolExecutor',return_value=pool):
                self.assertEqual(scan(args),130)
                self.assertEqual(scan(args),0)
            state=json.loads((args.output/'checkpoint.json').read_text())
            self.assertEqual(state['counts'],{'deleted':1,'healthy':3})
            self.assertEqual(state['deleted_labels'],1)
            self.assertEqual(state['next_index'],4)

    def test_delete_failure_is_not_reported_as_success_and_is_recoverable(self):
        from repair_operations import recover
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'images/train/bad.jpg'; image.parent.mkdir(parents=True)
            label=root/'labels/train/bad.txt'; label.parent.mkdir(parents=True)
            image.write_bytes(b'bad'); label.write_text('0 .5 .5 .1 .1')
            operations=root/'operations'
            with patch('repair_operations.finish',side_effect=PermissionError('blocked')):
                result=inspect_image(image,operations=operations)
            self.assertEqual(result['status'],'failed')
            self.assertTrue(image.exists())
            label.unlink()  # Simulate a crash between label and image deletion.
            recover(operations)
            self.assertFalse(image.exists())
            self.assertFalse(label.exists())

    def test_check_only_keeps_undecodable_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'images/train/bad.png'; image.parent.mkdir(parents=True)
            label=root/'labels/train/bad.txt'; label.parent.mkdir(parents=True)
            image.write_bytes(b'bad'); label.write_text('0 .5 .5 .1 .1')
            self.assertEqual(inspect_image(image,True)['status'],'detected')
            self.assertTrue(image.exists())
            self.assertTrue(label.exists())

    def test_recover_incomplete_inplace_write_preserves_hardlinks(self):
        from repair_operations import recover
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=self.image(root)
            image.write_bytes(image.read_bytes()[:-2])
            alias=root/'alias.jpg'; os.link(image,alias)
            operations=root/'operations'
            with patch('repair_operations.finish', side_effect=OSError('simulated kill')):
                self.assertEqual(inspect_image(image,operations=operations)['status'],'failed')
            image.write_bytes(b'partial write')
            recover(operations)
            self.assertTrue(os.path.samefile(image,alias))
            self.assertEqual(inspect_image(alias)['status'],'healthy')
            self.assertFalse(list(root.glob('*.repair-*.jpg')))

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
