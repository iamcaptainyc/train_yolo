import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import yaml

from convert_donanet import convert


class DonanetTests(unittest.TestCase):
    def test_maps_names_to_camping_ids_and_preserves_splits_and_coordinates(self):
        names = {
            0: 'Ave', 1: 'Felis catus', 2: 'Bos taurus', 3: 'Canis familiaris',
            4: 'Herpestes ichneumon', 5: 'Meles meles', 6: 'Oryctolagus cuniculus',
            7: 'Dama dama', 8: 'Genetta genetta', 9: 'Equus sp', 10: 'Homo sapiens',
            11: 'Lepus granatensis', 12: 'Lynx pardinus', 13: 'Cervus elaphus',
            14: 'Vulpes vulpes', 15: 'wild_boar', 16: 'Vehicle', 17: 'Empty',
        }
        coords = '0.500000 0.4 0.30 0.2'
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            output = Path(temporary) / 'output'

            def add_pair(split, stem, class_ids):
                image = source / 'images' / split / 'camera' / f'{stem}.jpg'
                label = source / 'labels' / split / 'camera' / f'{stem}.txt'
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (32, 32)).save(image)
                if class_ids is not None:
                    label.parent.mkdir(parents=True, exist_ok=True)
                    label.write_text(''.join(f'{cid} {coords}\n' for cid in class_ids), encoding='utf-8')
                return image, label

            original, original_label = add_pair('train', 'mixed', (15, 10, 4, 8, 16))
            add_pair('train', 'with_lynx', (1, 12))
            add_pair('train', 'missing', None)
            add_pair('val', 'deer', (7, 13))
            add_pair('val', 'invalid', (999,))
            add_pair('test', 'other', (0, 1, 2, 3, 5, 6, 9, 11, 14))
            add_pair('test', 'empty', (17,))
            config = {'path': '.', **{s: f'images/{s}' for s in ('train', 'val', 'test')}, 'names': names}
            (source / 'data.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')

            convert(source, output)

            result = yaml.safe_load((output / 'data.yaml').read_text(encoding='utf-8'))
            self.assertEqual(list(result['names'].values()), [
                'bear', 'bird', 'cat', 'cattle', 'crocodile', 'deer', 'dog', 'eagle', 'elephant', 'horse',
                'leopard', 'lion', 'monkey', 'sheep', 'small_mammal', 'snake', 'tiger', 'wild_boar', 'wolf',
            ])
            self.assertEqual(result['nc'], 19)
            self.assertTrue(os.path.samefile(original, output / 'images/train/camera/mixed.jpg'))
            self.assertEqual(len(list((output / 'images').rglob('*.jpg'))), 3)
            self.assertEqual(len(list((output / 'labels').rglob('*.txt'))), 3)
            self.assertEqual((output / 'labels/train/camera/mixed.txt').read_text(encoding='utf-8'),
                             ''.join(f'{cid} {coords}\n' for cid in (17, 14, 14)))
            self.assertEqual((output / 'labels/val/camera/deer.txt').read_text(encoding='utf-8'), f'5 {coords}\n' * 2)
            self.assertEqual((output / 'labels/test/camera/other.txt').read_text(encoding='utf-8'),
                             ''.join(f'{cid} {coords}\n' for cid in (1, 2, 3, 6, 14, 14, 9, 14, 14)))
            self.assertIn(f'15 {coords}', original_label.read_text(encoding='utf-8'))
            report = (output / 'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('| 合计 | 7 | 3 | 1 | 1 | 1 | 1 |', report)
            self.assertIn('| 14 | small_mammal | 1 | 0 | 1 | 2 | 6 |', report)
            self.assertIn('| 0 | bear | 0 | 0 | 0 | 0 | 0 |', report)


if __name__ == '__main__':
    unittest.main()
