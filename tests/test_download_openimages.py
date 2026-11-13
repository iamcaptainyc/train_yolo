import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from download_openimages import BOX_CLASSES, BOX_URLS, DEFAULT_BOX_CLASSES, IMAGE_BASE, download, download_boxed, export_annotations, select_images


class DownloadOpenImagesTests(unittest.TestCase):
    def test_download_all_boxed_classes_reuses_local_images_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'openimages'
            meta = root/'_metadata'
            meta.mkdir(parents=True)
            (meta/'download_config.json').write_text('{"classes": ["Wolf", "Cat"]}')
            (meta/'train_human_labels.csv').write_text('unused')
            (root/'cat').mkdir()
            local = root/'cat'/'train_0000000000000001.jpg'
            local.write_bytes(b'local image')
            class_ids = dict(enumerate(DEFAULT_BOX_CLASSES, 1))
            header = 'ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf,IsDepiction\n'
            metadata = {
                BOX_CLASSES: ''.join(f'/m/{i},{name}\n' for i, name in class_ids.items()),
                BOX_URLS['train']: header + ''.join(
                    f'{i:016x},/m/{cid},0.1,0.5,0.2,0.6,0,0\n'
                    for i, cid in [(1, 4), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 6), (7, 1)]
                ),
                BOX_URLS['validation']: header + '0000000000000008,/m/1,0.1,0.5,0.2,0.6,0,0\n',
                BOX_URLS['test']: header + '0000000000000009,/m/2,0.1,0.5,0.2,0.6,0,0\n',
            }
            dest = Path(tmp)/'yolo'
            args = SimpleNamespace(output=root, yolo_output=dest, classes=None,
                                   splits=['train', 'validation', 'test'], retries=0, workers=2)
            fail_url = f'{IMAGE_BASE}/train/0000000000000007.jpg'
            failures = {fail_url}

            def fetch_fixture(url, path, retries, progress=False):
                path.parent.mkdir(parents=True, exist_ok=True)
                if url in metadata:
                    path.write_text(metadata[url])
                elif url in failures:
                    raise OSError('simulated network failure')
                else:
                    self.assertTrue(url.startswith(IMAGE_BASE))
                    path.write_bytes(b'downloaded image')

            with patch('download_openimages.download', side_effect=fetch_fixture) as fetch:
                with self.assertRaisesRegex(OSError, 'Rerun the same command'):
                    download_boxed(args)
                self.assertFalse(dest.exists())
                self.assertTrue((meta/'boxed_download_failures.json').is_file())
                fetch.reset_mock()
                failures.clear()
                self.assertEqual(download_boxed(args), 0)
                calls = [call.args[0] for call in fetch.call_args_list if call.args[0].startswith(IMAGE_BASE)]
                self.assertEqual(calls, [fail_url])

            report = json.loads((dest/'bbox_report.json').read_text())
            self.assertEqual(set(report['class_ids']), {name.lower() for name in DEFAULT_BOX_CLASSES})
            self.assertEqual(report['images_by_split'], {'train':7, 'val':1, 'test':1})
            self.assertEqual(report['images_by_split_and_class']['train']['elephant'], 3)
            self.assertEqual((report['reused_images'], report['downloaded_images']), (8, 1))
            self.assertTrue(os.path.samefile(local, dest/'images/train'/local.name))
            label = dest/'labels/train'/local.with_suffix('.txt').name
            self.assertEqual({int(row.split()[0]) for row in label.read_text().splitlines()},
                             {report['class_ids'][name] for name in ('cat', 'elephant')})
            self.assertFalse((meta/'boxed_download_failures.json').exists())
            self.assertEqual((meta/'train_human_labels.csv').read_text(), 'unused')

    def test_export_only_official_boxes_for_downloaded_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'openimages'
            meta = root/'_metadata'
            meta.mkdir(parents=True)
            (meta/'download_config.json').write_text(json.dumps({'classes':['Wolf', 'Lion', 'Crocodile']}))
            (meta/'train_human_labels.csv').write_bytes(b'unused image labels')
            local = {}
            for group, split, index in (('lion', 'train', 1), ('lion', 'train', 2),
                                         ('crocodile', 'validation', 3), ('wolf', 'test', 4)):
                image_id = f'{index:016x}'
                image = root/group/f'{split}_{image_id}.jpg'
                image.parent.mkdir(exist_ok=True)
                image.write_bytes(b'local image')
                local[index] = image

            header = ('ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf,IsDepiction\n')
            contents = {
                BOX_CLASSES: '/m/lion,Lion\n/m/croc,Crocodile\n',
                BOX_URLS['train']: header + ''.join(
                    f'{i:016x},{mid},0.1,0.5,0.2,0.6,{group},{depiction}\n'
                    for i, mid, group, depiction in (
                        (1, '/m/lion', 0, 0), (1, '/m/croc', 0, 0), (1, '/m/lion', 1, 0),
                        (1, '/m/lion', 0, 1), (2, '/m/croc', 0, 0), (9, '/m/lion', 0, 0),
                    )
                ),
                BOX_URLS['validation']: header + f'{3:016x},/m/croc,0.1,0.5,0.2,0.6,0,0\n',
            }

            def write_fixture(url, path, retries, progress):
                path.write_text(contents[url], encoding='utf-8')

            dest = root/'yolo_bbox'
            args = SimpleNamespace(output=root, yolo_output=None, classes=None,
                                   splits=['train', 'validation', 'test'], retries=1)
            with patch('download_openimages.download', side_effect=write_fixture) as fetch:
                self.assertEqual(export_annotations(args), 0)
            self.assertEqual({call.args[0] for call in fetch.call_args_list}, set(contents))
            report = json.loads((dest/'bbox_report.json').read_text())
            self.assertEqual(report['classes_without_official_boxes'], ['Wolf'])
            self.assertEqual(report['images_skipped_by_folder'], {'lion':1, 'wolf':1})
            self.assertEqual(report['images_by_split'], {'train':1, 'val':1})
            self.assertEqual(report['boxes_by_split_and_class']['train'], {'crocodile':1, 'lion':1})
            self.assertEqual(json.loads((dest/'data.yaml').read_text())['names'], ['crocodile', 'lion'])
            self.assertTrue(os.path.samefile(local[1], dest/'images/train'/local[1].name))
            self.assertFalse((dest/'images/train'/local[2].name).exists())
            label = dest/'labels/train'/local[1].with_suffix('.txt').name
            self.assertEqual([list(map(float, row.split())) for row in label.read_text().splitlines()],
                             [[0, 0.3, 0.4, 0.4, 0.4], [1, 0.3, 0.4, 0.4, 0.4]])
            self.assertEqual((meta/'train_human_labels.csv').read_bytes(), b'unused image labels')

    def test_cap_positive_only_wolf_merge_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ImageID','LabelName','Confidence'])
                for i in range(20):
                    writer.writerow([f'{i:016x}','wolf',1])
                writer.writerow([f'{100:016x}','red',1])
                writer.writerow([f'{100:016x}','jackal',1])
                writer.writerow([f'{101:016x}','cat',0])
                writer.writerow([f'{102:016x}','cat',1])
                writer.writerow([f'{102:016x}','lion',1])
            classes = {'wolf':'Wolf','red':'Red wolf','jackal':'Jackal','cat':'Cat','lion':'Lion'}
            records, available, conflicts = select_images([('train',path)], classes, 5, 42)
            self.assertEqual(len(records),5)
            self.assertEqual({r['group'] for r in records},{'wolf'})
            self.assertEqual(available['Wolf'],20)
            self.assertEqual(available['Cat'],1)
            self.assertEqual(len(conflicts),1)
            again = select_images([('train',path)], classes, 5, 42)
            self.assertEqual(records,again[0])
            self.assertTrue(any('Red wolf' in r['original_classes'] for r in records))

    def test_merged_wolf_cap_balances_all_four_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            classes = {str(i):name for i,name in enumerate(['Wolf','Red wolf','Coyote','Jackal','Cat'])}
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ImageID','LabelName','Confidence'])
                for cid in classes:
                    for index in range(20):
                        writer.writerow([f'{int(cid)*100+index:016x}',cid,1])
            records, _, _ = select_images([('train',path)],classes,8,42)
            wolves = [r for r in records if r['group']=='wolf']
            self.assertEqual(len(wolves),8)
            self.assertEqual(len([r for r in records if r['group']=='cat']),8)
            for name in ['Wolf','Red wolf','Coyote','Jackal']:
                self.assertEqual(sum(name in r['original_classes'] for r in wolves),2)

    def test_resume_range_and_reuse_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.jpg'
            path.with_name('image.jpg.part').write_bytes(b'abc')
            response = io.BytesIO(b'def')
            response.status = 206
            response.headers = {'Content-Range':'bytes 3-5/6'}
            with patch('urllib.request.urlopen',return_value=response) as request:
                download('https://example.com/image',path)
                self.assertEqual(request.call_args.args[0].get_header('Range'),'bytes=3-')
            self.assertEqual(path.read_bytes(),b'abcdef')
            with patch('urllib.request.urlopen') as request:
                download('https://example.com/image',path)
                request.assert_not_called()

    def test_server_ignores_range_restart_without_duplicating_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            path.with_name('labels.csv.part').write_bytes(b'old')
            response = io.BytesIO(b'new-full')
            response.status = 200
            response.headers = {'Content-Length':'8'}
            with patch('urllib.request.urlopen',return_value=response):
                download('https://example.com/labels',path)
            self.assertEqual(path.read_bytes(),b'new-full')
