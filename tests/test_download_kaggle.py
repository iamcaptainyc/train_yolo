import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from download_kaggle import dataset_handle, default_output_dir, download_with_retry, main


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


if __name__ == '__main__':
    unittest.main()
