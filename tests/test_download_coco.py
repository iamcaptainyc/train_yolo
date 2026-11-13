import hashlib
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

from download_coco import ARCHIVE_URLS, BASE, GCS_BASE, fetch_archive, fetch_image, run


class DownloadCOCOTests(unittest.TestCase):
    def test_image_download_prefers_working_fallback_after_503(self):
        first = 'train2017/000000000001.jpg'
        second = 'val2017/000000000002.jpg'
        prefer_gcs = Event()
        with patch('download_coco.download', side_effect=[
            HTTPError(f'{BASE}/{first}', 503, 'Service Unavailable', {}, None), None, None,
        ]) as download:
            fetch_image(first, Path('first.jpg'), 5, prefer_gcs)
            fetch_image(second, Path('second.jpg'), 5, prefer_gcs)
        self.assertEqual([call.args[0] for call in download.call_args_list],
                         [f'{BASE}/{first}', f'{GCS_BASE}/{first}', f'{GCS_BASE}/{second}'])

    def test_annotation_503_falls_back_and_validates_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'annotations.zip'
            partial = path.with_name(path.name+'.part')
            partial.write_bytes(b'official ')
            content = b'official archive'

            def fetch(url, destination, retries, progress):
                if url == ARCHIVE_URLS[0]:
                    raise HTTPError(url, 503, 'Service Unavailable', {}, None)
                if url == ARCHIVE_URLS[1]:
                    self.assertEqual(partial.read_bytes(), b'official ')
                    destination.write_bytes(b'corrupt archive!')
                else:
                    self.assertFalse(destination.exists())
                    destination.write_bytes(content)

            with patch('download_coco.download', side_effect=fetch) as download, \
                    patch('download_coco.ARCHIVE_SIZE', len(content)), \
                    patch('download_coco.ARCHIVE_SHA256', hashlib.sha256(content).hexdigest()):
                fetch_archive(path, 5)
                self.assertEqual([call.args[0] for call in download.call_args_list], list(ARCHIVE_URLS))
                self.assertEqual(path.read_bytes(), content)
                download.reset_mock()
                fetch_archive(path, 5)
                download.assert_not_called()

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
                                       f'{GCS_BASE}/val2017/000000000004.jpg',
                                       f'{BASE}/val2017/000000000004.jpg'])
            label = output/'labels/train/000000000001.txt'
            self.assertEqual(label.read_text().splitlines(), ['0 0.15 0.2 0.3 0.2', '1 0.25 0.25 0.1 0.1'])
            subset = json.loads((output/'_metadata/instances_train2017_subset.json').read_text())
            self.assertEqual(subset['annotations'], annotations[:2])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['train']['images'], 1)
            self.assertEqual(report['train']['images_by_class'], {'cow': 1, 'elephant': 1})
            self.assertEqual(report['val']['boxes_by_class'], {'elephant': 1})
