import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

from coco_hf_zip import RangeZip
from download_coco import ARCHIVE_URLS, BASE, GCS_BASE, fetch_archive, fetch_image, run


class DownloadCOCOTests(unittest.TestCase):
    def test_local_only_finalizes_partial_dataset_and_resume_completes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'yolo'
            archive = Path(tmp)/'annotations.zip'
            categories = [{'id': 21, 'name': 'cow'}, {'id': 22, 'name': 'elephant'}]
            with zipfile.ZipFile(archive, 'w') as source:
                for split, ids in (('train', (1, 2)), ('val', (3,))):
                    images = [{'id': i, 'file_name': f'{i:012}.jpg', 'width': 100, 'height': 100}
                              for i in ids]
                    annotations = [{'image_id': i, 'category_id': 21 if i != 2 else 22,
                                    'bbox': [0, 0, 10, 10]} for i in ids]
                    source.writestr(f'annotations/instances_{split}2017.json',
                                    json.dumps(dict(images=images, categories=categories, annotations=annotations)))
            for split, image_id in (('train', 1), ('val', 3)):
                target = output/'images'/split/f'{image_id:012}.jpg'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'already downloaded')
            meta = output/'_metadata'
            meta.mkdir()
            (meta/'classes.json').write_text('["cow", "elephant"]')
            (meta/'failures.json').write_text(json.dumps({
                'train': [{'file': '000000000002.jpg', 'error': 'Remote end closed connection'}], 'val': [],
            }))

            with patch('download_coco.download') as download, patch('download_coco.RangeZip') as remote:
                run(output, ['cow', 'elephant'], archive, 2, 0, local_only=True)
                download.assert_not_called()
                remote.assert_not_called()
                config = json.loads((output/'data.yaml').read_text())
                self.assertEqual((config['train'], config['val']), ('train.txt', 'val.txt'))
                self.assertEqual((output/'train.txt').read_text(), './images/train/000000000001.jpg\n')
                self.assertEqual((output/'labels/train/000000000001.txt').read_text(), '0 0.05 0.05 0.1 0.1\n')
                subset = json.loads((output/'_metadata/instances_train2017_subset.json').read_text())
                self.assertEqual([ann['image_id'] for ann in subset['annotations']], [1])
                report = json.loads((output/'report.json').read_text())
                self.assertEqual(report['train'], {
                    'images': 1, 'requested_images': 2, 'missing_images': 1,
                    'images_by_class': {'cow': 1}, 'boxes_by_class': {'cow': 1},
                })
                failures = json.loads((output/'_metadata/failures.json').read_text())
                self.assertEqual(failures['train'], [
                    {'file': '000000000002.jpg', 'error': 'Remote end closed connection'},
                ])

                remote.return_value.image.return_value = b'mirror image'
                run(output, ['cow', 'elephant'], archive, 2, 0)
                remote.return_value.image.assert_called_once_with('train', '000000000002.jpg')
                config = json.loads((output/'data.yaml').read_text())
                self.assertEqual(config['train'], 'images/train')
                self.assertFalse((output/'_metadata/failures.json').exists())

    def test_remote_zip_fetches_only_requested_member_and_checks_crc(self):
        buffer = io.BytesIO()
        selected = 'train2017/000000000001.jpg'
        unrelated = 'train2017/000000000002.jpg'
        # Force ZIP64 metadata like the multi-GB COCO training archive.
        with patch('zipfile.ZIP64_LIMIT', 100_000), zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr(selected, b'selected image', compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr(unrelated, b'x' * 200_000)
        content = buffer.getvalue()
        ranges = []

        def respond(request, timeout):
            start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
            ranges.append((start, end))
            response = io.BytesIO(content[start:end+1])
            response.status = 206
            response.headers = {'Content-Range': f'bytes {start}-{end}/{len(content)}'}
            return response

        with patch('coco_hf_zip.urlopen', side_effect=respond):
            remote = RangeZip('https://example.com/train2017.zip', retries=0)
            self.assertEqual(remote.image('train', selected.split('/')[1]), b'selected image')
            self.assertLess(sum(end-start+1 for start, end in ranges), 1000)
            info, end = remote.members[selected]
            with patch.object(remote, 'read_range', return_value=content[info.header_offset:end-1]+b'\xff'):
                with self.assertRaisesRegex(ValueError, 'ZIP'):
                    remote.image('train', selected.split('/')[1])
        response = io.BytesIO(content)
        response.status = 200
        with patch('coco_hf_zip.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'ignored HTTP Range'):
                RangeZip('https://example.com/train2017.zip', retries=0)

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
                with self.assertRaisesRegex(ValueError, 'at least one completed image'):
                    run(output, ['cow', 'elephant'], archive, 2, 0, source='official')
                self.assertFalse((output/'data.yaml').exists())
                failure_report = json.loads((output/'_metadata/failures.json').read_text())
                self.assertEqual(failure_report['train'], [])
                self.assertEqual(failure_report['val'][0]['file'], '000000000004.jpg')
                label = output/'labels/train/000000000001.txt'
                with patch('download_coco.Path.replace', autospec=True, side_effect=Path.replace) as replace:
                    failed = False
                    run(output, ['cow', 'elephant'], archive, 2, 0, source='official')
                    self.assertNotIn(label, [call.args[1] for call in replace.call_args_list])
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
            self.assertFalse((output/'_metadata/failures.json').exists())

            mirror_output = root/'mirror'
            with patch('download_coco.fetch_archive') as annotation, patch('download_coco.RangeZip') as remote, \
                    patch('download_coco.fetch_image') as official:
                annotation.side_effect = lambda path, *_: path.write_bytes(archive.read_bytes())
                remote.return_value.image.return_value = b'mirror image'
                run(mirror_output, ['cow', 'elephant'], None, 2, 0)
                self.assertTrue(annotation.call_args.args[2][0].startswith('https://hf-mirror.com/'))
                self.assertEqual([call.args[0].rsplit('/', 1)[-1] for call in remote.call_args_list],
                                 ['train2017.zip', 'val2017.zip'])
                self.assertTrue(all(call.args[0].startswith('https://hf-mirror.com/')
                                    for call in remote.call_args_list))
                official.assert_not_called()
                self.assertEqual((mirror_output/'images/train/000000000001.jpg').read_bytes(), b'mirror image')
                remote.reset_mock()
                run(mirror_output, ['cow', 'elephant'], archive, 2, 0)
                remote.assert_not_called()
