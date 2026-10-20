import io
import json
from pathlib import Path
from threading import Event
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image
import requests

from lila_selection import download_selected, image_url, select_images, write_selection, rebalance_negatives
from download_lila import run_datasets


def document(images, annotations):
    return {'images': images, 'annotations': annotations, 'categories': [
        {'id': 0, 'name': 'empty'}, {'id': 10, 'name': 'deer'}, {'id': 1, 'name': 'human'}]}


def image(key, **extra):
    return {'id': key, 'file_name': key+'.jpg', 'width': 100, 'height': 80, **extra}


def annotation(key, category=10, box=None):
    value = {'image_id': key, 'category_id': category}
    if box is not None:
        value['bbox'] = box
    return value


def response(body, status=200, **headers):
    value = requests.Response()
    value.status_code = status
    value.url = 'https://example.test/image.png'
    value.headers.update(headers)
    value.raw = io.BytesIO(body)
    return value


class SelectionTests(unittest.TestCase):
    def test_only_boxed_positives_and_explicit_empty_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positives = [image(f'p{i}') for i in range(19)]
            empties = [image(f'e{i}') for i in range(10)]
            unknown = image('unboxed_animal')
            human = image('person')
            private = image('hidden', file_name='private/hidden.jpg')
            mixed = image('contradictory_empty')
            meta = document(positives+empties+[unknown, human, private, mixed],
                            [annotation(i['id']) for i in positives]+[annotation(i['id'], 0) for i in empties]+
                            [annotation('unboxed_animal'), annotation('person', 1), annotation('hidden', 0),
                             annotation('contradictory_empty', 0), annotation('contradictory_empty', 10)])
            boxes = document(positives+[human], [annotation(i['id'], box=[1, 2, 10, 20]) for i in positives]+
                             [annotation('person', 1, [1, 2, 10, 20])])
            metadata, bbox = root/'metadata.json', root/'bbox.json'
            metadata.write_text(json.dumps(meta)); bbox.write_text(json.dumps(boxes))
            first, report = select_images(metadata, bbox)
            second, _ = select_images(metadata, bbox)
            self.assertEqual(first, second)
            self.assertEqual(report['positive_images'], 19)
            self.assertEqual(report['negative_images'], 1)
            self.assertEqual(report['actual_negative_fraction'], 0.05)
            self.assertFalse({'unboxed_animal.jpg', 'person.jpg', 'private/hidden.jpg', 'contradictory_empty.jpg'} & set(first))
            write_selection(first, report, root/'selection')
            subset = json.loads((root/'selection/selected_annotations.json').read_text())
            self.assertEqual(len(subset['images']), 20)
            self.assertEqual(len(subset['annotations']), 19)

    def test_no_empty_class_does_not_make_missing_boxes_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ena.json'
            doc = document([image('p'), image('unboxed')], [annotation('p', 0, [1, 2, 10, 20]), annotation('unboxed', 0)])
            doc['categories'] = [{'id': 0, 'name': 'Bird'}]
            path.write_text(json.dumps(doc))
            selected, report = select_images(path, negative_fraction=0.5)
            self.assertEqual(set(selected), {'p.jpg'})
            self.assertEqual(report['negative_shortfall'], 1)

    def test_empty_bbox_placeholder_is_not_an_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'meta.json'
            path.write_text(json.dumps(document([image('p'), image('e')],
                             [annotation('p', box=[1, 2, 10, 20]), annotation('e', 0, [])])))
            selected, report = select_images(path, negative_fraction=0.5)
            self.assertEqual(selected['e.jpg']['role'], 'negative')
            self.assertEqual(report['negative_images'], 1)

    def test_invalid_boxes_and_corrupt_images_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'meta.json'
            path.write_text(json.dumps(document([image('good'), image('invalid'), image('corrupt', corrupt=True)],
                             [annotation('good', box=[1, 2, 10, 20]), annotation('invalid', box=[1, 2, -1, 20]),
                              annotation('corrupt', box=[1, 2, 10, 20])])))
            selected, _ = select_images(path)
            self.assertEqual(set(selected), {'good.jpg'})

    def test_zip_metadata_and_no_boxes_refuse_full_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'meta.zip'
            with zipfile.ZipFile(path, 'w') as bundle:
                bundle.writestr('metadata.json', json.dumps(document([image('unboxed')], [annotation('unboxed')])))
            with self.assertRaisesRegex(ValueError, 'refusing full-image fallback'):
                select_images(path)

    def test_http_urls_preserve_nested_names_and_spaces(self):
        url = image_url('s3://bucket/folder', 'sub/a (1).JPG')
        self.assertEqual(url, 'https://s3.us-west-2.amazonaws.com/bucket/folder/sub/a%20%281%29.JPG')

    def test_batch_backfills_only_from_confirmed_empty_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = []
            for name, empties in [('no_empty', 0), ('with_empty', 10)]:
                root = Path(tmp)/name
                root.mkdir()
                path = root/'meta.json'
                positives = [image(f'p{i}') for i in range(19)]
                blank = [image(f'e{i}') for i in range(empties)]
                path.write_text(json.dumps(document(positives+blank,
                    [annotation(i['id'], box=[1, 2, 10, 20]) for i in positives]+
                    [annotation(i['id'], 0) for i in blank])))
                directory = root/'selection'
                selected, report = select_images(path, candidate_file=directory/'negative_candidates.jsonl')
                write_selection(selected, report, directory)
                prepared.append({'directory': directory, 'report': report, 'item': {'dataset': name}})
            summary = rebalance_negatives(prepared, 0.05, 42)
            self.assertEqual((summary['positive_images'], summary['negative_images']), (38, 2))
            self.assertEqual(summary['actual_negative_fraction'], 0.05)
            self.assertEqual([p['report']['negative_images'] for p in prepared], [0, 2])
            subset = json.loads((prepared[1]['directory']/'selected_annotations.json').read_text())
            positive_ids = {i['id'] for i in subset['images'] if i['selection_role'] == 'positive'}
            self.assertEqual({a['image_id'] for a in subset['annotations']}, positive_ids)
            original = (prepared[1]['directory']/'negatives.txt').read_text()
            rebalance_negatives(prepared, 0.05, 42)
            self.assertEqual(original, (prepared[1]['directory']/'negatives.txt').read_text())

    def test_audit_scheduler_fetches_labels_without_syncing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'meta.json'
            path.write_text(json.dumps(document([image('p')], [annotation('p', box=[1, 2, 10, 20])])))
            plan = [{'dataset': 'test', 'kind': 'images', 'source': 's3://bucket/images', 'target': str(root/'images')},
                    {'dataset': 'test', 'kind': 'metadata', 'source': 's3://bucket/meta.json', 'target': str(path)}]
            def fetch(items, *args, **kwargs):
                self.assertTrue(all(i['kind'] != 'images' for i in items))
                return [{**i, 'status': 'ok'} for i in items]
            with patch('download_lila.run_plan', side_effect=fetch), patch('lila_selection.download_selected') as transfer:
                result = run_datasets(plan, 'aws', image_policy='boxed', audit_only=True, state_dir=root/'state')
            self.assertFalse(transfer.called)
            self.assertEqual(result[-1]['status'], 'audited')
            self.assertTrue((root/'selection/selected_annotations.json').is_file())


class SelectedDownloadTests(unittest.TestCase):
    def png(self):
        stream = io.BytesIO()
        Image.new('RGB', (10, 10), 'red').save(stream, format='PNG')
        return stream.getvalue()

    def test_partial_image_resumes_and_completed_file_skips_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self.png()
            midpoint = len(payload)//2
            first = response(b'', **{'Content-Length': str(len(payload)), 'ETag': 'test-etag'})

            def chunks(*args, **kwargs):
                yield payload[:midpoint]
                raise requests.exceptions.ReadTimeout('interrupted')

            first.iter_content = chunks
            resumed = response(payload[midpoint:], 206, **{'Content-Range': f'bytes {midpoint}-{len(payload)-1}/{len(payload)}'})
            session = Mock()
            session.get.side_effect = [first, resumed]
            selection = {'sub/a.png': {}}
            item = {'source': 's3://bucket/images', 'target': str(root/'images'), 'dataset': 'test'}
            with patch('requests.Session', return_value=session):
                result = download_selected(selection, item, workers=1, retries=1, retry_delay=0,
                                           max_retry_delay=0, state_dir=root/'state', stop_event=Event())
                again = download_selected(selection, item, workers=1, retries=0, retry_delay=0,
                                          max_retry_delay=0, state_dir=root/'state', stop_event=Event())
            self.assertEqual(result['status'], 'ok')
            self.assertEqual((root/'images/sub/a.png').read_bytes(), payload)
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(session.get.call_args_list[1].kwargs['headers']['Range'], f'bytes={midpoint}-')
            self.assertEqual(again['skipped_images'], 1)

    def test_missing_image_is_reported_without_downloading_unlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Mock()
            session.get.side_effect = [response(self.png(), **{'Content-Length': str(len(self.png()))}), response(b'', 404)]
            item = {'source': 's3://bucket/images', 'target': str(root/'images'), 'dataset': 'test'}
            with patch('requests.Session', return_value=session):
                result = download_selected({'a.png': {}, 'missing.png': {}}, item, workers=1, retries=3,
                                           retry_delay=0, max_retry_delay=0, state_dir=root/'state', stop_event=Event())
            self.assertEqual(result['failed_images'], 1)
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(json.loads((root/'state/failed_images.json').read_text())[0]['file_name'], 'missing.png')


if __name__ == '__main__':
    unittest.main()
