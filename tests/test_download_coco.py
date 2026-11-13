import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from download_coco import BASE, run


class DownloadCOCOTests(unittest.TestCase):
    def test_selects_only_boxed_animals_and_resumes_without_changing_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'annotations.zip'
            categories = [{'id': 1, 'name': 'person'}, {'id': 21, 'name': 'cow'},
                          {'id': 22, 'name': 'elephant'}]
            images = [{'id': i, 'file_name': f'{i:012}.jpg', 'width': 100, 'height': 100}
                      for i in range(1, 5)]
            annotations = [
                {'id': 1, 'image_id': 1, 'category_id': 21, 'bbox': [-10, 10, 40, 20]},
                {'id': 2, 'image_id': 1, 'category_id': 22, 'bbox': [20, 20, 10, 10]},
                {'id': 3, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 10, 10]},
                {'id': 4, 'image_id': 2, 'category_id': 21, 'bbox': [0, 0, 10, 10], 'iscrowd': 1},
                {'id': 5, 'image_id': 3, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
                {'id': 6, 'image_id': 4, 'category_id': 22, 'bbox': [0, 0, 10, 10]},
            ]
            with zipfile.ZipFile(archive, 'w') as source:
                for split, boxes in (('train', annotations[:-1]), ('val', annotations[-1:])):
                    source.writestr(f'annotations/instances_{split}2017.json',
                                    json.dumps(dict(images=images, categories=categories, annotations=boxes)))
            output = root/'yolo'
            fetched = []
            failed = True

            def fetch(url, path, retries):
                if path.is_file():
                    return
                fetched.append(url)
                if failed and 'val2017' in url:
                    raise OSError('simulated disconnect')
                path.write_bytes(b'local image')

            with patch('download_coco.download', side_effect=fetch):
                with self.assertRaisesRegex(OSError, 'rerun the same command'):
                    run(output, ['cow', 'elephant'], archive, 2, 0)
                self.assertFalse((output/'data.yaml').exists())
                failed = False
                run(output, ['cow', 'elephant'], archive, 2, 0)
                with self.assertRaisesRegex(ValueError, 'Class selection changed'):
                    run(output, ['cow'], archive, 2, 0)

            self.assertEqual(fetched, [f'{BASE}/train2017/000000000001.jpg',
                                       f'{BASE}/val2017/000000000004.jpg',
                                       f'{BASE}/val2017/000000000004.jpg'])
            label = output/'labels/train/000000000001.txt'
            self.assertEqual(label.read_text().splitlines(), ['0 0.15 0.2 0.3 0.2', '1 0.25 0.25 0.1 0.1'])
            subset = json.loads((output/'_metadata/instances_train2017_subset.json').read_text())
            self.assertEqual(subset['annotations'], annotations[:2])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['train']['images'], 1)
            self.assertEqual(report['train']['images_by_class'], {'cow': 1, 'elephant': 1})
            self.assertEqual(report['val']['boxes_by_class'], {'elephant': 1})
