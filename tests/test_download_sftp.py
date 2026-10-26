import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_sftp as downloader


class DownloadTests(unittest.TestCase):
    def run_download(self, target, transfer, **kwargs):
        with patch.object(downloader.shutil, 'which', return_value='sftp'), \
                patch.object(downloader, 'transfer', side_effect=transfer), \
                patch.object(downloader.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            downloader.download('26080-yc', '/data/animals.zip', target, **kwargs)

    def test_existing_scp_partial_survives_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'animals.zip'
            target.write_bytes(b'existing')
            offsets = []
            def transfer(cmd, script, path, timeout):
                offsets.append(path.stat().st_size)
                self.assertIn('reget "/data/animals.zip" "./animals.zip"', script)
                self.assertIn('BatchMode=yes', cmd)
                with path.open('ab') as f:
                    f.write(b'123')
                return 255 if len(offsets) == 1 else 0
            self.run_download(target, transfer, retries=1)
            self.assertEqual(offsets, [8, 11])
            self.assertEqual(target.read_bytes(), b'existing123123')

    def test_retry_exhaustion_keeps_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'x.zip'
            target.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                self.run_download(target, [255, 255], retries=1)
            self.assertEqual(target.read_bytes(), b'partial')

    def test_unlimited_retry_and_hash_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'x.zip'
            target.write_bytes(b'complete')
            self.run_download(target, [255, 255, 0], retries=0,
                              sha256=hashlib.sha256(b'complete').hexdigest())

    def test_hash_mismatch_is_not_retried_or_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'x.zip'
            target.write_bytes(b'wrong-prefix')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.run_download(target, [0], sha256='0'*64)
            self.assertEqual(target.read_bytes(), b'wrong-prefix')

    def test_interrupt_keeps_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'x.zip'
            target.write_bytes(b'partial')
            with self.assertRaises(KeyboardInterrupt):
                self.run_download(target, KeyboardInterrupt())
            self.assertEqual(target.read_bytes(), b'partial')

    def test_literal_paths_and_no_batch_injection(self):
        script = downloader.batch('/data/a "quoted" file.zip', Path('local file.zip'))
        self.assertIn('"./local file.zip"', script)
        self.assertIn('\\"quoted\\"', script)
        for remote in ['/data/*.zip', '/data/a\n!bad', 'relative.zip']:
            with self.assertRaises(ValueError):
                downloader.batch(remote, Path('file.zip'))
        for host in ['-oProxyCommand=bad', 'host\nbye', '']:
            with self.assertRaises(ValueError):
                downloader.command(host)

    def test_success_without_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'missing'):
                self.run_download(Path(tmp)/'x.zip', [0])


if __name__ == '__main__':
    unittest.main()
