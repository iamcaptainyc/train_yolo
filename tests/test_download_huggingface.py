from pathlib import Path
import errno
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from download_huggingface import download, write_snake_config


class SnakeConfigTests(unittest.TestCase):
    def test_network_retry_reuses_same_destination(self):
        fetch = Mock(side_effect=[TimeoutError('timeout'), 'done'])
        with patch.dict('sys.modules', {'huggingface_hub':SimpleNamespace(snapshot_download=fetch)}), patch('download_huggingface.time.sleep'):
            self.assertEqual(download('owner/data', Path('destination'), retries=1), 'done')
        self.assertEqual(fetch.call_count, 2)
        self.assertTrue(all(c.kwargs['local_dir']=='destination' for c in fetch.call_args_list))

    def test_disk_full_is_not_retried(self):
        fetch = Mock(side_effect=OSError(errno.ENOSPC, 'No space left on device'))
        with patch.dict('sys.modules', {'huggingface_hub':SimpleNamespace(snapshot_download=fetch)}), patch('download_huggingface.time.sleep') as sleep:
            with self.assertRaises(OSError):
                download('owner/data', Path('destination'), retries=3)
        self.assertEqual(fetch.call_count, 1)
        sleep.assert_not_called()

    def test_relocated_yolo_yaml_preserves_classes_and_upstream_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = 'train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames: [a, b]\n'
            (root/'data.yaml').write_text(upstream)
            for split in ('train', 'valid', 'test'):
                for kind in ('images', 'labels'):
                    (root/split/kind).mkdir(parents=True)
            target = write_snake_config(root)
            config = yaml.safe_load(target.read_text())
            self.assertEqual(config['names'], ['a', 'b'])
            self.assertEqual(config['nc'], 2)
            for key in ('train', 'val', 'test'):
                self.assertTrue((Path(config['path'])/config[key]).is_dir())
            self.assertEqual((root/'data.yaml').read_text(), upstream)

    def test_incomplete_layout_does_not_produce_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data.yaml').write_text('nc: 1\nnames: [snake]\n')
            with self.assertRaisesRegex(ValueError, 'Missing YOLO'):
                write_snake_config(root)
            self.assertFalse((root/'data.local.yaml').exists())
