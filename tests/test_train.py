import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import train


class TrainFlipTests(unittest.TestCase):
    def invoke(self, root, flags=(), saved_fliplr=0.5):
        data = root/'data.yaml'
        data.write_text('names: [cat]\n')
        checkpoint = root/'last.pt'
        checkpoint.touch()
        model = Mock(ckpt={'train_args': {'fliplr': saved_fliplr}})
        model.train.return_value = SimpleNamespace(save_dir=root/'output')
        sdk = SimpleNamespace(YOLO=Mock(return_value=model))
        modules = {'ultralytics': sdk,
                   'torch': SimpleNamespace(__version__='fake', cuda=SimpleNamespace(is_available=lambda: False))}
        with patch('sys.argv', ['train.py', '--data', str(data), *flags]), \
                patch.dict('sys.modules', modules), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train.main()
        return model

    def test_new_training_forwards_default_zero_and_custom_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for probability in (None, 0, 0.25, 1):
                with self.subTest(probability=probability):
                    flags = [] if probability is None else ['--fliplr', str(probability)]
                    model = self.invoke(root, flags)
                    self.assertEqual(model.train.call_args.kwargs['fliplr'], 0.5 if probability is None else probability)

    def test_invalid_probabilities_and_non_training_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for flags in (['--fliplr=-0.1'], ['--fliplr', '1.1'], ['--fliplr', 'nan'],
                          ['--fliplr', 'inf'], ['--mode', 'val', '--fliplr', '0']):
                with self.subTest(flags=flags), self.assertRaises(SystemExit) as exc:
                    self.invoke(root, flags)
                self.assertEqual(exc.exception.code, 2)

    def test_resume_preserves_saved_flip_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = ['--resume', str(root/'last.pt'), '--device', 'cpu']
            for extra in ([], ['--fliplr', '0']):
                model = self.invoke(root, flags + extra, saved_fliplr=0)
                model.train.assert_called_once_with(resume=True, device='cpu')
            with self.assertRaises(SystemExit) as exc:
                self.invoke(root, flags + ['--fliplr', '0.5'], saved_fliplr=0)
            self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
