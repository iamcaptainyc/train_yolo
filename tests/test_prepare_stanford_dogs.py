import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / 'prepare_stanford_dogs.py'
SIZE = (80, 48)
BOXES = {'dog': (6, 4, 32, 18), 'cat': (46, 25, 65, 41)}


def write_pair(image, annotation, orientation=1, upright_xml=False, missing_name=False):
    pixels = Image.new('RGB', SIZE)
    draw = ImageDraw.Draw(pixels)
    for name, color in (('dog', 'red'), ('cat', (0, 255, 0))):
        x1, y1, x2, y2 = BOXES[name]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=color)
    exif = pixels.getexif()
    exif[274] = orientation
    pixels.save(image, exif=exif, quality=100, subsampling=0)

    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = '%s'
    size = ET.SubElement(root, 'size')
    width, height = SIZE[::-1] if upright_xml else SIZE
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    for name, (x1, y1, x2, y2) in BOXES.items():
        obj = ET.SubElement(root, 'object')
        if not (missing_name and name == 'cat'):
            ET.SubElement(obj, 'name').text = f' {name} '
        coords = (SIZE[1] - y2, x1, SIZE[1] - y1, x2) if upright_xml else (x1, y1, x2, y2)
        box = ET.SubElement(obj, 'bndbox')
        for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), coords):
            ET.SubElement(box, key).text = str(value)
    ET.ElementTree(root).write(annotation)


def convert(source, output, *flags):
    return subprocess.run([sys.executable, '-X', 'utf8', str(SCRIPT), '--source', str(source),
                           '--output', str(output), '--copy-mode', 'copy', *flags],
                          capture_output=True, encoding='utf-8')


class StanfordDogsTests(unittest.TestCase):
    def test_xml_mode_keeps_each_object_class_and_aligns_exif_corrected_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'voc', root/'yolo'
            images, annotations = source/'JPEGImages', source/'Annotations'
            images.mkdir(parents=True)
            annotations.mkdir()
            for orientation in range(9):
                write_pair(images/f'{orientation}.jpg', annotations/f'{orientation}.xml', orientation)
            write_pair(images/'upright.jpg', annotations/'upright.xml', 6, upright_xml=True)
            write_pair(images/'rotated.tiff', annotations/'rotated.xml', 7)

            result = convert(source, output, '--class-mode', 'xml')
            self.assertEqual(result.returncode, 0, result.stderr)
            config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf-8'))
            self.assertEqual(config['names'], {0: 'cat', 1: 'dog'})
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['image_storage_modes'], {'exif_normalized': 10, 'copy': 1})
            exported = [path for path in (output/'images').rglob('*') if path.is_file()]
            self.assertEqual(len(exported), 11)
            for image in exported:
                label = (output/'labels'/image.relative_to(output/'images')).with_suffix('.txt')
                with self.subTest(image=image.name), Image.open(image) as pixels:
                    self.assertEqual(pixels.getexif().get(274, 1), 1)
                    channels = {0: pixels.getchannel('G'), 1: pixels.getchannel('R')}
                    seen = set()
                    for row in label.read_text().splitlines():
                        cid, *values = row.split()
                        seen.add(int(cid))
                        x, y, w, h = map(float, values)
                        width, height = pixels.size
                        bounds = (round((x-w/2)*width), round((y-h/2)*height),
                                  round((x+w/2)*width), round((y+h/2)*height))
                        self.assertEqual(bounds, channels[int(cid)].point(lambda p: 255 if p > 160 else 0).getbbox())
                    self.assertEqual(seen, {0, 1})

    def test_invalid_xml_pair_is_reported_and_other_pairs_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'stanford'
            breed = 'n02085620-Chihuahua'
            images, annotations = source/'images/Images'/breed, source/'annotations/Annotation'/breed
            images.mkdir(parents=True)
            annotations.mkdir(parents=True)
            for i in range(4):
                write_pair(images/f'{i}.jpg', annotations/str(i), i+1, missing_name=i == 3)

            strict = convert(source, root/'strict', '--class-mode', 'xml', '--strict')
            self.assertEqual(strict.returncode, 2, strict.stderr)
            self.assertIn('missing class name', strict.stderr)
            self.assertFalse((root/'strict').exists())

            result = convert(source, root/'xml', '--class-mode', 'xml')
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'xml/report.json').read_text(encoding='utf-8'))
            self.assertEqual(sum(report['images'].values()), 3)
            self.assertEqual(len(report['issues']), 1)
            self.assertIn('missing class name', report['issues'][0]['error'])

            for flags, name in (((), 'dog'), (('--class-mode', 'breed'), 'Chihuahua')):
                with self.subTest(mode=name):
                    output = root/name
                    result = convert(source, output, *flags)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf-8'))
                    self.assertEqual(config['names'], {0: name})
                    self.assertEqual(len(list((output/'labels').rglob('*.txt'))), 4)


if __name__ == '__main__':
    unittest.main()
