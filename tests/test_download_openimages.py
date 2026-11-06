import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_openimages import download, select_images


class DownloadOpenImagesTests(unittest.TestCase):
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
