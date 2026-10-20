import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import requests

from download_kaggle import build_jobs, download_with_retry
from kaggle_resume import extract_archive, resumable_dataset_download, transfer_archive


def response(body, *, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = 'https://storage.example.test/archive'
    result.headers.update(headers or {})
    result.raw = io.BytesIO(body)
    return result


class ArchiveTransferTests(unittest.TestCase):
    def test_appends_only_missing_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'1.archive'
            archive.write_bytes(b'abc')
            checksum = base64.b64encode(hashlib.md5(b'abcdef').digest()).decode()
            first = response(b'abcdef', headers={'Content-Length': '6', 'X-Goog-Hash': 'md5='+checksum})
            resumed = response(b'def', status=206, headers={'Content-Range': 'bytes 3-5/6'})
            with patch('requests.get', return_value=resumed) as get:
                transfer_archive(first, archive)
            self.assertEqual(archive.read_bytes(), b'abcdef')
            self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=3-')

    def test_ignored_or_wrong_range_preserves_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'1.archive'
            for status, headers in [(200, {}), (206, {'Content-Range': 'bytes 0-5/6'}),
                                    (206, {'Content-Range': 'bytes 3-6/7'})]:
                archive.write_bytes(b'abc')
                with self.subTest(status=status, headers=headers), \
                     patch('requests.get', return_value=response(b'abcdef', status=status, headers=headers)), \
                     self.assertRaises(ValueError):
                    transfer_archive(response(b'abcdef', headers={'Content-Length': '6'}), archive)
                self.assertEqual(archive.read_bytes(), b'abc')

    def test_timeout_keeps_written_bytes_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'1.archive'

            def chunks(**kwargs):
                yield b'abc'
                raise requests.exceptions.ReadTimeout('interrupted')

            first = response(b'', headers={'Content-Length': '6'})
            first.iter_content = chunks
            with self.assertRaises(requests.exceptions.ReadTimeout):
                transfer_archive(first, archive)
            self.assertEqual(archive.read_bytes(), b'abc')
            resumed = response(b'def', status=206, headers={'Content-Range': 'bytes 3-5/6'})
            with patch('requests.get', return_value=resumed):
                transfer_archive(response(b'', headers={'Content-Length': '6'}), archive)
            self.assertEqual(archive.read_bytes(), b'abcdef')

    def test_equal_size_validates_hash_without_redownloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'1.archive'
            archive.write_bytes(b'abc')
            with patch('requests.get') as get, self.assertRaisesRegex(ValueError, 'MD5'):
                transfer_archive(response(b'', headers={'Content-Length': '3', 'X-Goog-Hash': 'md5=bad'}), archive)
            get.assert_not_called()
            self.assertEqual(archive.read_bytes(), b'abc')

    def test_retry_does_not_turn_resume_into_force(self):
        download = Mock(side_effect=[requests.exceptions.ReadTimeout(), '/output'])
        with patch('download_kaggle.time.sleep'):
            download_with_retry(download, 'owner/data', output_dir='/output',
                                force_download=False, restart_output=False)
        self.assertFalse(download.call_args.kwargs['force_download'])


class ArchiveExtractionTests(unittest.TestCase):
    def test_extracts_only_valid_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            archive = output/'1.archive'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('train/a.txt', 'data')
            extract_archive(archive, output)
            self.assertEqual((output/'train/a.txt').read_text(), 'data')

    def test_traversal_and_reserved_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            archive = output/'1.archive'
            for name in ['../outside.txt', '.kaggle-resume.json', '1.archive', '.complete/bundle.complete']:
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr('valid.txt', 'do not extract before validation')
                    bundle.writestr(name, 'bad')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    extract_archive(archive, output)
                self.assertFalse((output/'valid.txt').exists())

    def test_crc_failure_does_not_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            archive = output/'1.archive'
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
                bundle.writestr('data.txt', b'abcdef')
            data = bytearray(archive.read_bytes())
            data[30 + len('data.txt')] ^= 1
            archive.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'CRC'):
                extract_archive(archive, output)
            self.assertFalse((output/'data.txt').exists())
            self.assertTrue(archive.exists())


class ResumeIntegrationTests(unittest.TestCase):
    def test_migrates_old_archive_and_pins_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            archive = output/'7.archive'
            archive.write_bytes(b'partial')
            api = Mock()
            api.datasets.dataset_api_client.download_dataset.return_value = Mock()

            def complete_download(resp, target):
                self.assertEqual(target, archive)
                self.assertEqual(target.read_bytes(), b'partial')
                with zipfile.ZipFile(target, 'w') as bundle:
                    bundle.writestr('train/image.txt', 'finished')

            with patch('kagglehub.clients.build_kaggle_client') as client, \
                 patch('kagglehub.http_resolver._get_current_version') as current_version, \
                 patch('kaggle_resume.transfer_archive', side_effect=complete_download):
                client.return_value.__enter__.return_value = api
                self.assertEqual(resumable_dataset_download('owner/data', output_dir=str(output)), str(output.resolve()))
                state = json.loads((output/'.kaggle-resume.json').read_text())
                self.assertEqual(state, {'dataset': 'owner/data', 'version': 7})
                request = api.datasets.dataset_api_client.download_dataset.call_args.args[0]
                self.assertEqual(request.dataset_version_number, 7)
                current_version.assert_not_called()
                self.assertFalse(archive.exists())
                self.assertEqual((output/'train/image.txt').read_text(), 'finished')
                # A second invocation recognizes the completion marker without another transfer.
                api.datasets.dataset_api_client.download_dataset.reset_mock()
                resumable_dataset_download('owner/data', output_dir=str(output))
                api.datasets.dataset_api_client.download_dataset.assert_not_called()

    def test_nonempty_resume_allowed_but_unknown_files_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output/'existing.txt'
            sentinel.write_text('keep')
            build_jobs(['owner/data/versions/7'], output, resume=True)
            with patch('kagglehub.clients.build_kaggle_client'), self.assertRaisesRegex(ValueError, 'No resumable'):
                resumable_dataset_download('owner/data/versions/7', output_dir=str(output))
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_state_from_another_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/'.kaggle-resume.json').write_text(json.dumps({'dataset': 'other/data', 'version': 7}))
            with self.assertRaisesRegex(ValueError, 'another dataset'):
                resumable_dataset_download('owner/data', output_dir=str(output))


if __name__ == '__main__':
    unittest.main()
