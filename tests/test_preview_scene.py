from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'preview_scene.py'


class ScenePreviewTests(unittest.TestCase):
    def test_thresholds_use_only_target_boxes_and_keep_other_boxes_visible(self):
        far = '0 0.25 0.5 0.09 0.1\n'
        mid = '0 0.25 0.5 0.1 0.1\n'
        near = '0 0.75 0.5 0.25 0.25\n'
        other = '1 0.75 0.5 0.25 0.25\n'
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / 'data.yaml').write_text('names: [bear, wolf]\ntrain: images/train\nval: images/val\n',
                                               encoding='utf-8')
            samples = [('train', 'far', far + other), ('train', 'mid', mid), ('val', 'near', near),
                       ('val', 'mixed', near + far), ('val', 'other', other)]
            originals = {}
            for split, stem, annotation in samples:
                image = dataset / 'images' / split / f'{stem}.png'
                label = dataset / 'labels' / split / f'{stem}.txt'
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (400, 200), 'white').save(image)
                label.write_text(annotation, encoding='utf-8')
                originals[image] = image.read_bytes()
                originals[label] = label.read_bytes()

            command = [sys.executable, '-X', 'utf8', str(SCRIPT), str(dataset), '--class', 'bear', '--limit', '0']
            run = subprocess.run(command, capture_output=True, encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            root = dataset / 'scene_preview' / '0_bear'
            preview = root / 'far-1of100_near-1of16'
            self.assertEqual({path.parent.parent.name for path in preview.rglob('*.jpg')}, {'far', 'mid', 'near', 'mixed'})
            self.assertEqual(len(list(preview.rglob('*.jpg'))), 4)
            self.assertTrue(any((preview / 'mid/train').glob('*mid.jpg')))
            self.assertTrue(any((preview / 'near/val').glob('*near.jpg')))
            with Image.open(next((preview / 'far/train').glob('*.jpg'))) as image:
                body_y = image.height - 200
                r, g, b = image.getpixel((100, body_y + 91))
                self.assertGreater(b - r, 75)
                gray = image.getpixel((280, body_y + 76))
                self.assertLess(max(gray), 200)
                self.assertLess(max(gray) - min(gray), 20)
            report = (preview / 'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('| mixed | val | 1 | 1 |', report)
            self.assertIn('共匹配 4 / 5 张', report)

            run = subprocess.run(command + ['--far-below', '0.02', '--near-from', '1/10'], capture_output=True,
                                 encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            alternate = root / 'far-1of50_near-1of10'
            self.assertEqual(len(list((alternate / 'far/train').glob('*.jpg'))), 2)
            self.assertTrue(any((alternate / 'mid/val').glob('*near.jpg')))
            self.assertTrue(any((preview / 'near/val').glob('*near.jpg')))
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)


if __name__ == '__main__':
    unittest.main()
