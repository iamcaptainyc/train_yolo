import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import download_sftp as downloader


class DownloadTests(unittest.TestCase):
    def test_remote_listing_and_symlink_unknown(self):
        self.assertEqual(downloader.remote_size('-rw-r--r-- 1 1000 1000 17000000000 Oct 26 12:00 animals file.zip'), 17000000000)
        self.assertEqual(downloader.remote_size('-rw------- 1 root root 0 Jan 1 2026 empty.zip'), 0)
        self.assertEqual(downloader.remote_size('-rw-r--r--    ? 0        0        17000000000 Oct 26 12:00 /data/animals.zip'), 17000000000)
        self.assertEqual(downloader.remote_size('-rw-r--r--+   ? 1000     1000      1234 Oct 26 12:00 spaced file.zip'), 1234)
        self.assertIsNone(downloader.remote_size('lrwxrwxrwx 1 0 0 9 Oct 26 12:00 link -> file'))
        self.assertIsNone(downloader.remote_size('Permission denied'))

    def test_progress_resume_speed_single_line_and_eta(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        progress = downloader.Progress(stream)
        with patch.object(downloader.time, 'monotonic', side_effect=[100, 102]), \
                patch.object(downloader.shutil, 'get_terminal_size', return_value=type('Size', (), {'columns': 160})()):
            progress.show(6 * 1024**3, 10 * 1024**3, force=True)
            progress.show(6 * 1024**3 + 2 * 1024**2, 10 * 1024**3, state='Downloading', force=True)
        value = stream.getvalue()
        self.assertIn('60.0%', value)
        self.assertIn('1.0MiB/s', value)
        self.assertIn('ETA 01:08:14', value)
        self.assertEqual(value.count('\r'), 2)
        self.assertNotIn('\n', value)
        progress.finish()
        self.assertTrue(stream.getvalue().endswith('\n'))

    def test_transfer_reads_listing_and_keeps_errors(self):
        # Real local subprocess/pipe reader, no remote connection or dataset access.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'file.zip'
            target.write_bytes(b'prefix')
            program = ("import sys,time; from pathlib import Path; sys.stdin.read(); "
                       "print('-rw-r--r--    ? 0        0               9 Oct 26 12:00 file.zip',flush=True); "
                       "time.sleep(.3); p=Path('file.zip'); p.write_bytes(p.read_bytes()+b'123')")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = downloader.transfer([sys.executable, '-u', '-c', program],
                                           downloader.batch('/file.zip', target), target)
            self.assertEqual(code, 0)
            self.assertEqual(target.read_bytes(), b'prefix123')
            self.assertIn('100.0%', stdout.getvalue())
            self.assertIn('Transferred', stdout.getvalue())
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                code = downloader.transfer([sys.executable, '-u', '-c',
                    "import sys; sys.stdin.read(); print('Permission denied',file=sys.stderr); sys.exit(255)"],
                    'bye\n', target)
            self.assertEqual(code, 255)
            self.assertIn('Permission denied', stderr.getvalue())

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
