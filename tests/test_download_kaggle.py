import unittest
import tempfile
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock, patch

import requests

from download_kaggle import build_jobs, dataset_handle, default_output_dir, download_with_retry, main, run_downloads


class DestinationTests(unittest.TestCase):
    def test_url_and_versions_have_separate_directories(self):
        self.assertEqual(default_output_dir('alice/animals'), Path('datasets/alice/animals'))
        self.assertNotEqual(default_output_dir('alice/animals'), default_output_dir('bob/animals'))
        handle = dataset_handle('https://www.kaggle.com/datasets/alice/animals?datasetVersionNumber=2')
        self.assertEqual(default_output_dir(handle), Path('datasets/alice/animals/version-2'))

    def test_default_custom_and_cache_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            automatic, custom = Path(tmp)/'automatic', Path(tmp)/'custom'
            for flags, expected in [([], automatic), (['--output', str(custom)], custom), (['--cache'], None)]:
                module = Mock()
                module.dataset_download.return_value = str(Path(tmp)/'downloaded')
                with self.subTest(flags=flags), patch.dict('sys.modules', {'kagglehub': module}), \
                     patch('sys.argv', ['download_kaggle.py', 'owner/data', *flags]), \
                     patch('download_kaggle.default_output_dir', return_value=automatic):
                    main()
                kwargs = module.dataset_download.call_args.kwargs
                self.assertEqual(kwargs.get('output_dir'), str(expected.resolve()) if expected else None)

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output / 'existing.txt'
            sentinel.write_text('keep', encoding='utf-8')
            module = Mock()
            with patch.dict('sys.modules', {'kagglehub': module}), \
                 patch('sys.argv', ['download_kaggle.py', 'owner/data']), \
                 patch('download_kaggle.default_output_dir', return_value=output), \
                 self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 1)
            module.dataset_download.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')


class RetryTests(unittest.TestCase):
    @patch('download_kaggle.time.sleep')
    def test_timeout_then_success(self, sleep):
        download = Mock(side_effect=[requests.exceptions.ReadTimeout(), '/cache/dataset'])
        result = download_with_retry(download, 'owner/data', force_download=False)
        self.assertEqual(result, '/cache/dataset')
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(5.0)
        self.assertFalse(download.call_args.kwargs['force_download'])

    @patch('download_kaggle.time.sleep')
    def test_bounded_backoff_and_exhaustion(self, sleep):
        error = requests.exceptions.ConnectionError('disconnected')
        download = Mock(side_effect=error)
        with self.assertRaises(requests.exceptions.ConnectionError) as raised:
            download_with_retry(download, 'owner/data', retries=4, retry_delay=5, max_retry_delay=12)
        self.assertIs(raised.exception, error)
        self.assertEqual(download.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 12, 12])

    @patch('download_kaggle.time.sleep')
    def test_http_status_policy(self, sleep):
        for status in [400, 401, 403, 404, 408, 429, 500, 503]:
            with self.subTest(status=status):
                response = requests.Response()
                response.status_code = status
                error = requests.HTTPError(response=response)
                download = Mock(side_effect=[error, '/ok'])
                if status in (408, 429, 500, 503):
                    self.assertEqual(download_with_retry(download, 'owner/data'), '/ok')
                    self.assertEqual(download.call_count, 2)
                else:
                    with self.assertRaises(requests.HTTPError):
                        download_with_retry(download, 'owner/data')
                    self.assertEqual(download.call_count, 1)

    @patch('download_kaggle.time.sleep')
    def test_direct_output_replaces_own_partial_download(self, sleep):
        download = Mock(side_effect=[TimeoutError(), '/output'])
        download_with_retry(download, 'owner/data', output_dir='/output', force_download=False, path='data.csv')
        self.assertFalse(download.call_args_list[0].kwargs['force_download'])
        self.assertTrue(download.call_args_list[1].kwargs['force_download'])
        self.assertEqual(download.call_args.kwargs['path'], 'data.csv')

    @patch('download_kaggle.time.sleep')
    def test_no_retries_and_permanent_errors(self, sleep):
        for error in [ValueError('bad handle'), PermissionError('disk'),
                      requests.exceptions.SSLError('certificate'), KeyboardInterrupt()]:
            download = Mock(side_effect=error)
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                download_with_retry(download, 'owner/data')
            self.assertEqual(download.call_count, 1)
        download = Mock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            download_with_retry(download, 'owner/data', retries=0)
        self.assertEqual(download.call_count, 1)
        sleep.assert_not_called()

    def test_invalid_retry_settings(self):
        for kwargs in [{'retries': -1}, {'retry_delay': -1}, {'retry_delay': float('nan')},
                       {'max_retry_delay': float('inf')}]:
            download = Mock()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                download_with_retry(download, 'owner/data', **kwargs)
            download.assert_not_called()


class ParallelTests(unittest.TestCase):
    def test_two_downloads_really_overlap_and_keep_separate_outputs(self):
        barrier = Barrier(2)
        lock = Lock()
        active, peak = 0, 0
        outputs = []

        def download(handle, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                outputs.append(kwargs['output_dir'])
            try:
                barrier.wait(timeout=5)
                return kwargs['output_dir']
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_jobs(['alice/data', 'bob/data'], Path(tmp))
            successes, failures = run_downloads(download, jobs, workers=2)
            self.assertEqual(peak, 2)
            self.assertEqual(len(set(outputs)), 2)
            self.assertEqual({h for h, _ in successes}, {'alice/data', 'bob/data'})
            self.assertEqual(failures, [])

    def test_failure_isolated_and_retries_per_dataset(self):
        calls = {}
        lock = Lock()

        def download(handle, **kwargs):
            with lock:
                calls[handle] = calls.get(handle, 0) + 1
                count = calls[handle]
            if handle == 'owner/bad':
                raise ValueError('invalid resource')
            if handle == 'owner/retry' and count == 1:
                raise requests.exceptions.ReadTimeout()
            return '/cache/' + handle

        jobs = [(h, None) for h in ['owner/bad', 'owner/retry', 'owner/good']]
        successes, failures = run_downloads(download, jobs, workers=2, retry_delay=0)
        self.assertEqual({h for h, _ in successes}, {'owner/retry', 'owner/good'})
        self.assertEqual([h for h, _ in failures], ['owner/bad'])
        self.assertEqual(calls, {'owner/bad': 1, 'owner/retry': 2, 'owner/good': 1})

    def test_overlapping_directories_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'overlap'):
                build_jobs(['owner/data', 'owner/data/versions/2'], Path(tmp))
        with self.assertRaisesRegex(ValueError, 'share a cache'):
            build_jobs(['owner/data', 'owner/data/versions/2'], cache=True)

    def test_batch_cli_deduplicates_and_logs_in_once(self):
        module = Mock()
        module.dataset_download.side_effect = lambda handle, **kwargs: '/cache/' + handle
        with patch.dict('sys.modules', {'kagglehub': module}), \
             patch('sys.argv', ['download_kaggle.py', 'owner/first', 'owner/second',
                                'https://www.kaggle.com/datasets/owner/first', '--cache', '--login', '--workers', '2']):
            main()
        self.assertEqual(module.dataset_download.call_count, 2)
        module.login.assert_called_once_with()

    def test_stop_event_prevents_next_attempt(self):
        stop = Event()
        stop.set()
        download = Mock()
        from concurrent.futures import CancelledError
        with self.assertRaises(CancelledError):
            download_with_retry(download, 'owner/data', stop_event=stop)
        download.assert_not_called()

    def test_batch_reports_failure_exit_status(self):
        module = Mock()
        module.dataset_download.side_effect = [ValueError('missing'), '/cache/good']
        with patch.dict('sys.modules', {'kagglehub': module}), \
             patch('sys.argv', ['download_kaggle.py', 'owner/bad', 'owner/good', '--cache', '--workers', '1']), \
             self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(module.dataset_download.call_count, 2)


if __name__ == '__main__':
    unittest.main()
