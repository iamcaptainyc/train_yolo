import tempfile
import errno
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image
import yaml

from prepare_dataset import assign_validation, collect, parse_annotation, prepare


class ConversionTests(unittest.TestCase):
    def image(self, root, split, category, stem, color, annotation):
        folder = root / split / category
        (folder / 'Label').mkdir(parents=True, exist_ok=True)
        target = folder / f'{stem}.png'
        Image.new('RGB', (100, 80), color).save(target)
        (folder / 'Label' / f'{stem}.txt').write_text(annotation, encoding='utf-8')
        return target

    def test_multi_word_class_and_geometry(self):
        boxes = parse_annotation('Polar bear 10 20 50 60\nCat 0 0 100 80', 100, 80)
        self.assertEqual(boxes[0], ('polar bear', 0.3, 0.5, 0.4, 0.5))
        self.assertEqual(boxes[1], ('cat', 0.5, 0.5, 1.0, 1.0))

    def test_invalid_boxes_are_rejected(self):
        for value in ['', 'Cat nan 0 5 5', 'Cat 0 0 101 10', 'Cat 5 0 1 4',
                      '0 0.5 0.5 0.1 0.1', 'Cat 1 2 3']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_annotation(value, 100, 80)

    def test_full_conversion_and_reproducible_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            for split, colors in [('train', ['red', 'green', 'blue']), ('test', ['white'])]:
                for i, color in enumerate(colors):
                    self.image(root, split, 'Cat', str(i), color, 'Cat 10 10 50 50')
            # Repeated image in another class folder must merge annotations.
            self.image(root, 'train', 'Dog', 'duplicate', 'red', 'Dog 0 0 20 20')
            first = prepare(root, Path(tmp)/'out1', preview_count=1)
            second = prepare(root, Path(tmp)/'out2', preview_count=0)
            self.assertEqual(first['images'], {'train': 2, 'val': 1, 'test': 1})
            self.assertEqual(first['manifest'], second['manifest'])
            self.assertEqual(first['classes'], ['cat', 'dog'])
            sets = {s: {r['id'] for r in first['manifest'] if r['split'] == s}
                    for s in ['train', 'val', 'test']}
            self.assertFalse(sets['train'] & sets['val'])
            self.assertFalse((sets['train'] | sets['val']) & sets['test'])
            config = yaml.safe_load((Path(tmp)/'out1/data.yaml').read_text())
            self.assertEqual(config['names'], {0: 'cat', 1: 'dog'})
            self.assertTrue(list((Path(tmp)/'out1/preview').glob('*.jpg')))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(root, Path(tmp)/'out1')

    def test_cross_split_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ['train', 'test']:
                self.image(root, split, 'Cat', 'a', 'red', 'Cat 0 0 10 10')
            with self.assertRaisesRegex(ValueError, 'across original splits'):
                collect(root)

    def test_missing_annotation_skips_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.image(root, 'train', 'Cat', 'a', 'red', 'Cat 0 0 10 10')
            (image.parent / 'Label/a.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                collect(root)
            records, issues = collect(root, skip_invalid=True)
            self.assertEqual(records, [])
            self.assertEqual(len(issues), 1)

    def test_validation_retains_rare_secondary_class_in_train(self):
        records = [{'id': str(i), 'split': 'train', 'boxes': {('cat', .5, .5, .2, .2)}} for i in range(5)]
        records[0]['boxes'].add(('dog', .5, .5, .1, .1))
        assign_validation(records, .8, 42)
        self.assertEqual(records[0]['split'], 'train')
        self.assertTrue(any(r['split']=='val' for r in records))

    def test_auto_hardlinks_and_cross_device_fallback_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            for split, colors in [('train', ['red','green','blue']), ('test',['white'])]:
                for i, color in enumerate(colors):
                    self.image(root, split, 'Cat', str(i), color, 'Cat 0 0 20 20')
            first = prepare(root, Path(tmp)/'linked', preview_count=0, copy_mode='hardlink')
            self.assertEqual(first['image_storage_modes'], {'hardlink':4})
            source = Path(first['manifest'][0]['sources'][0])
            target = next((Path(tmp)/'linked/images').rglob(first['manifest'][0]['id']+'.png'))
            self.assertTrue(os.path.samefile(source,target))
            with patch('prepare_dataset.os.link', side_effect=OSError(errno.EXDEV, 'cross-device link')):
                second = prepare(root, Path(tmp)/'copied', preview_count=0)
            self.assertEqual(second['image_storage_modes'], {'copy':4})


if __name__ == '__main__':
    unittest.main()
