import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import yaml

from visualize_yolo import class_color, collect_source, load_font, read_boxes, render_image, visualize


class VisualizeTests(unittest.TestCase):
    def make_image(self, root, relative, label='0 0.5 0.5 0.4 0.4\n'):
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (400, 200), 'white').save(image)
        target = Path(*['labels' if p == 'images' else p for p in image.parts]).with_suffix('.txt')
        if label is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(label, encoding='utf-8')
        return image

    def run_quiet(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return visualize(*args, **kwargs)

    def test_layouts_names_and_split_selection(self):
        for layout in ('images/train', 'train/images'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = self.make_image(root/'dataset', layout+'/nested/cat.png')
                config = {'names': ['猫'], 'train': layout}
                (root/'dataset/data.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
                report = self.run_quiet(root/'dataset', root/'out', split='train')
                self.assertEqual(report['names'], {0: '猫'})
                self.assertEqual(report['counts'], {'ok': 1})
                self.assertEqual(report['images'][0]['image'], str(image.resolve()))
                self.assertTrue((root/'out'/report['images'][0]['output']).is_file())
                self.assertEqual(json.loads((root/'out/report.json').read_text(encoding='utf-8'))['selected_images'], 1)

    def test_flat_without_yaml_and_custom_label_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root/'dataset', 'images/a.png', label=None)
            (root/'custom').mkdir()
            (root/'custom/a.txt').write_text('7 0.5 0.5 0.4 0.4\n')
            report = self.run_quiet(image.parent, root/'out', labels=root/'custom')
            self.assertEqual(report['counts'], {'ok': 1})
            self.assertEqual(report['names'], {})
            # Flat datasets without metadata are also accepted.
            images, _, names = collect_source(root/'dataset')
            self.assertEqual(images, [image.resolve()])
            self.assertEqual(names, {})

    def test_external_yaml_lists_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root/'external', 'images/a.png')
            (root/'dataset').mkdir()
            (root/'dataset/train.txt').write_text('../external/images/a.png\n../external/images/a.png\n')
            config = {'names': {0: 'cat'}, 'train': ['train.txt', str(image.parent)]}
            path = root/'dataset/data.yaml'
            path.write_text(yaml.safe_dump(config))
            report = self.run_quiet(path, root/'out', split='train')
            self.assertEqual(report['total_images'], 1)
            self.assertEqual(report['counts'], {'ok': 1})

    def test_missing_empty_invalid_and_broken_images_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_image(root/'dataset', 'images/empty.png', '')
            self.make_image(root/'dataset', 'images/missing.png', None)
            self.make_image(root/'dataset', 'images/invalid.png', '0 nan 0.5 0.4 0.4')
            broken = self.make_image(root/'dataset', 'images/broken.png')
            broken.write_bytes(b'broken')
            report = self.run_quiet(root/'dataset', root/'out', limit=0)
            self.assertEqual(report['counts'], {'failed': 2, 'empty_label': 1, 'missing_label': 1})
            self.assertEqual(len(list((root/'out').glob('*.jpg'))), 2)
            self.assertEqual(broken.read_bytes(), b'broken')

    def test_box_scaling_and_coordinate_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root, 'images/a.png')
            label = root/'labels/a.txt'
            boxes = read_boxes(label, {0: 'cat'})
            preview = render_image(image, boxes, {0: 'cat'}, load_font(), max_size=200)
            self.assertEqual(preview.size, (200, 100))
            self.assertEqual(preview.getpixel((100, 70)), class_color(0))
            self.assertEqual(preview.getpixel((100, 80)), (255, 255, 255))
            for row in ('0 0.5 0.5 0 0.4', '0 1 0.5 0.4 0.4', '1 0.5 0.5 0.4 0.4',
                        '0 0.5 0.5 0.4 0.4 0.6 0.7', '-1 0.5 0.5 0.4 0.4'):
                label.write_text(row)
                with self.subTest(row=row), self.assertRaises(ValueError):
                    read_boxes(label, {0: 'cat'})

    def test_sampling_output_protection_and_filename_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a.png', 'a.jpg', 'b.png', 'c.png', 'd.png'):
                self.make_image(root/'dataset', 'images/'+name)
            first = self.run_quiet(root/'dataset', root/'one', limit=3, seed=42)
            second = self.run_quiet(root/'dataset', root/'two', limit=3, seed=42)
            self.assertEqual(first['images'], second['images'])
            self.assertEqual(first['total_images'], 5)
            all_images = self.run_quiet(root/'dataset', root/'all', limit=0)
            self.assertEqual(len({r['output'] for r in all_images['images']}), 5)
            with self.assertRaisesRegex(ValueError, 'new output'):
                self.run_quiet(root/'dataset', root/'one')
            for output in (root/'dataset/preview', root/'dataset/images/preview'):
                nested = self.run_quiet(root/'dataset', output, limit=0)
                self.assertEqual(nested['counts'], {'ok': 5})
                self.assertEqual(len(list(output.glob('*.jpg'))), 5)

    def test_cli_single_image_class_names_and_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root, 'images/a.png')
            data = root/'data.yaml'
            data.write_text('names: [cat]\n')
            script = Path(__file__).resolve().parents[1]/'visualize_yolo.py'
            command = [sys.executable, '-X', 'utf8', str(script), '--source', str(image), '--data', str(data)]
            result = subprocess.run(command + ['--output', str(root/'out')], capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            (root/'labels/a.txt').unlink()
            result = subprocess.run(command + ['--output', str(root/'missing')], capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((root/'missing/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['counts'], {'missing_label': 1})


if __name__ == '__main__':
    unittest.main()
