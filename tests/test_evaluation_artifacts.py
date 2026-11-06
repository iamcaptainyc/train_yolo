import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evaluation_artifacts import evaluate_logged


class Evaluator:
    def __init__(self):
        self.callbacks = {'on_val_end':[]}

    def add_callback(self, name, callback):
        self.callbacks[name].append(callback)

    def val(self, **options):
        matrix = SimpleNamespace(names={0:'bear',1:'dog'},
            matrix=[[7,2,1],[3,8,2],[4,0,0]])
        for callback in self.callbacks['on_val_end']:
            callback(SimpleNamespace(confusion_matrix=matrix, save_dir=options['project']))
        return SimpleNamespace(box=SimpleNamespace(mp=.7,mr=.6,map50=.8,map=.5),
                               speed={'inference':1.2}, summary=lambda:[{'Class':'bear'}])


class EvaluationArtifactTests(unittest.TestCase):
    def test_callbacks_matrix_axes_normalization_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Evaluator()
            for split in ('train','val','test'):
                result = evaluate_logged(model, tmp, split, 'best.pt', project=tmp)
                matrix = json.loads((Path(tmp)/f'{split}_confusion_matrix.json').read_text())
                self.assertEqual(matrix['matrix'], [[7,2,1],[3,8,2],[4,0,0]])
                self.assertAlmostEqual(matrix['normalized_matrix'][0][0], .5)
                self.assertEqual(matrix['labels'], ['bear','dog','background'])
                self.assertEqual(result['metrics']['map50_95'], .5)
                self.assertTrue((Path(tmp)/f'{split}.log').is_file())
            self.assertEqual(model.callbacks['on_val_end'], [])

    def test_failed_retry_clears_old_success_and_logs_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Evaluator()
            evaluate_logged(model, tmp, 'test', 'best.pt', project=tmp)
            def fail(**options):
                raise RuntimeError('evaluation failed')
            model.val = fail
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                evaluate_logged(model, tmp, 'test', 'best.pt', project=tmp)
            result = json.loads((Path(tmp)/'test_result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse((Path(tmp)/'test_confusion_matrix.json').exists())
            self.assertIn('evaluation failed', (Path(tmp)/'test.log').read_text())
            self.assertEqual(model.callbacks['on_val_end'], [])
