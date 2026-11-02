import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw
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
                report = self.run_quiet(root/'dataset', root/'out', split='train', classes=['猫'])
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
            report = self.run_quiet(image.parent, root/'out', labels=root/'custom', classes=['7'])
            self.assertEqual(report['counts'], {'ok': 1})
            self.assertEqual(report['names'], {})
            # Flat datasets without metadata are also accepted.
            images, _, names = collect_source(root/'dataset')
            self.assertEqual(images, [image.resolve()])
            self.assertEqual(names, {})

    def test_class_name_exports_all_matches_beyond_default_sample_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(101):
                self.make_image(root/'dataset', f'images/{index:03d}.png')
            self.make_image(root/'dataset', 'images/other.png', '1 0.5 0.5 0.4 0.4\n')
            self.make_image(root/'dataset', 'images/empty.png', '')
            (root/'dataset/data.yaml').write_text('names: [left_green, person]\n')
            report = self.run_quiet(root/'dataset', root/'all', classes=['left_green'])
            self.assertEqual(report['total_images'], 103)
            self.assertEqual(report['matched_images'], 101)
            self.assertEqual(report['selected_images'], 101)
            self.assertEqual(report['counts'], {'ok': 101})
            self.assertEqual(report['classes'], [0])
            self.assertEqual(report['limit'], 0)
            self.assertEqual(len(list((root/'all').glob('*.jpg'))), 101)
            unfiltered = self.run_quiet(root/'dataset', root/'default')
            self.assertEqual(unfiltered['selected_images'], 100)
            self.assertEqual(unfiltered['limit'], 100)

    def test_class_filter_samples_after_matching_and_draws_only_selected_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(8):
                self.make_image(root/'dataset', f'images/target{index}.png',
                                '0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.2 0.2\n2 0.8 0.8 0.2 0.2\n')
            for index in range(12):
                self.make_image(root/'dataset', f'images/other{index}.png', '1 0.5 0.5 0.4 0.4\n')
            (root/'dataset/data.yaml').write_text('names: [left_green, person, right_green]\n')
            with patch('visualize_yolo.render_image', wraps=render_image) as render:
                first = self.run_quiet(root/'dataset', root/'one', classes=['left_green', '2'], limit=3)
            second = self.run_quiet(root/'dataset', root/'two', classes=['0', 'right_green'], limit=3)
            self.assertEqual(first['images'], second['images'])
            self.assertEqual(first['matched_images'], 8)
            self.assertEqual(first['selected_images'], 3)
            self.assertEqual(first['classes'], [0, 2])
            self.assertEqual(render.call_count, 3)
            for call in render.call_args_list:
                self.assertEqual({box[0] for box in call.args[1]}, {0, 2})
            self.assertTrue(all(record['boxes'] == 2 for record in first['images']))

    def test_class_filter_reports_scan_errors_and_handles_no_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_image(root/'dataset', 'images/good.png')
            self.make_image(root/'dataset', 'images/missing.png', None)
            self.make_image(root/'dataset', 'images/bad.png', '0 nan 0.5 0.4 0.4\n')
            self.make_image(root/'dataset', 'images/empty.png', '')
            (root/'dataset/data.yaml').write_text('names: [left_green, person]\n')
            report = self.run_quiet(root/'dataset', root/'out', classes=['0'])
            self.assertEqual(report['matched_images'], 1)
            self.assertEqual(report['counts'], {'ok': 1, 'failed': 1, 'missing_label': 1})
            self.assertEqual(len(report['scan_issues']), 2)
            self.assertTrue(all(issue.get('error') for issue in report['scan_issues']))
            single = root/'dataset/images/good.png'
            empty = self.run_quiet(single, root/'none', data=root/'dataset/data.yaml', classes=['person'])
            self.assertEqual(empty['matched_images'], 0)
            self.assertEqual(empty['selected_images'], 0)
            self.assertEqual(empty['images'], [])
            self.assertTrue((root/'none/report.json').is_file())
            for classes in (['unknown'], ['99'], []):
                with self.subTest(classes=classes), self.assertRaises(ValueError):
                    self.run_quiet(root/'dataset', root/'invalid', classes=classes)
                self.assertFalse((root/'invalid').exists())

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
            header = preview.height - 100
            self.assertEqual(preview.width, 200)
            self.assertGreater(header, 0)
            expected_border = tuple(round(c * 128/255 + 255 * 127/255) for c in class_color(0))
            self.assertEqual(preview.getpixel((100, header + 70)), expected_border)
            self.assertEqual(preview.getpixel((100, header + 80)), (255, 255, 255))
            # The entire box interior stays untouched; only the border is blended.
            interior = preview.crop((63, header + 33, 138, header + 68))
            self.assertIsNone(ImageChops.difference(interior, Image.new('RGB', interior.size, 'white')).getbbox())
            for row in ('0 0.5 0.5 0 0.4', '0 1 0.5 0.4 0.4', '1 0.5 0.5 0.4 0.4',
                        '0 0.5 0.5 0.4 0.4 0.6 0.7', '-1 0.5 0.5 0.4 0.4'):
                label.write_text(row)
                with self.subTest(row=row), self.assertRaises(ValueError):
                    read_boxes(label, {0: 'cat'})

    def test_legend_deduplicates_present_classes_and_wraps_above_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root, 'images/a.png')
            boxes = [(1, 0.2, 0.2, 0.2, 0.2), (1, 0.8, 0.8, 0.2, 0.2),
                     (2, 0.5, 0.5, 0.2, 0.2)]
            names = {0: 'absent', 1: 'cat', 2: 'long category name ' * 4}
            with patch.object(ImageDraw.ImageDraw, 'text', autospec=True,
                              side_effect=ImageDraw.ImageDraw.text) as text:
                preview = render_image(image, boxes, names, load_font(), max_size=200)
            captions = [call.args[2] for call in text.call_args_list]
            self.assertEqual(captions.count('1: cat'), 1)
            self.assertNotIn('absent', ''.join(captions))
            self.assertIn('2: long category name', ''.join(captions))
            self.assertGreater(len(captions), 2)  # Long names wrap.
            header = preview.height - 100
            for call in text.call_args_list:
                draw, position, caption = call.args[:3]
                bounds = draw.textbbox(position, caption, font=call.kwargs['font'])
                self.assertGreaterEqual(bounds[0], 0)
                self.assertLessEqual(bounds[2], preview.width)
                self.assertLess(bounds[3], header)
            # The old box label area is now untouched image content.
            self.assertEqual(preview.getpixel((20, header + 1)), (255, 255, 255))

    def test_empty_and_missing_label_text_does_not_cover_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.make_image(root, 'images/a.png', '')
            for status in ('empty_label', 'missing_label'):
                with self.subTest(status=status):
                    preview = render_image(image, [], {}, load_font(), status=status)
                    body = preview.crop((0, preview.height - 200, 400, preview.height))
                    self.assertIsNone(ImageChops.difference(body, Image.new('RGB', (400, 200), 'white')).getbbox())

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
            result = subprocess.run(command + ['--classes', 'cat', '--output', str(root/'cat')],
                                    capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root/'cat/report.json').read_text())['classes'], [0])
            (root/'labels/a.txt').unlink()
            result = subprocess.run(command + ['--output', str(root/'missing')], capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((root/'missing/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['counts'], {'missing_label': 1})
            result = subprocess.run(command + ['--classes', '0', '--output', str(root/'scan_missing')],
                                    capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((root/'scan_missing/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['matched_images'], 0)
            self.assertEqual(report['counts'], {'missing_label': 1})


if __name__ == '__main__':
    unittest.main()
