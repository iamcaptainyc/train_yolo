import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from train_batch import load_config, run_batch


class TrainBatchTests(unittest.TestCase):
    def test_independent_training_defaults_overrides_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'model.pt'
            model.touch()
            for name in ('first','broken','last'):
                dataset = root/name
                dataset.mkdir()
                (dataset/'data.yaml').write_text('names: [fox]\n')
            stub = root/'train_stub.py'
            stub.write_text('''
import json
from pathlib import Path
import sys
args = sys.argv
get = lambda flag: args[args.index(flag)+1]
project = Path(get('--project'))
output = project/get('--name')
assert get('--model') == str(project.parents[2]/'model.pt')
if project.name == 'broken':
    sys.exit(3)
assert get('--batch') == ('32' if project.name == 'last' else '64')
assert ('--no-amp' in args) == (project.name == 'last')
(output/'weights').mkdir(parents=True)
(output/'weights/best.pt').touch()
for split in ('val','test'):
    result = {'split':split, 'status':'ok',
              'metrics':{'precision':.5,'recall':.4,'map50':.6,'map50_95':.3}}
    if split == 'test' and project.name == 'last':
        result = {'split':split, 'status':'skipped', 'reason':'No test split'}
    (output/f'{split}_result.json').write_text(json.dumps(result))
''', encoding='utf-8')
            config = root/'batch.yaml'
            config.write_text(yaml.safe_dump({
                'model':'model.pt', 'output':'runs', 'defaults':{'batch':64,'amp':True},
                'datasets':[{'data':'first'}, {'data':'broken'},
                            {'data':'last/data.yaml','params':{'batch':32,'amp':False}}],
            }), encoding='utf-8')
            with patch('train_batch.TRAIN_SCRIPT', stub):
                self.assertEqual(run_batch(config),1)
            directory = next((root/'runs').iterdir())
            records = json.loads((directory/'summary.json').read_text(encoding='utf-8'))['experiments']
            self.assertEqual([row['status'] for row in records],['complete','failed','complete'])
            self.assertEqual(records[1]['exit_code'],3)
            self.assertEqual(records[2]['evaluation']['test']['status'],'skipped')
            self.assertEqual(records[2]['params'],{'batch':32,'amp':False})
            with (directory/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['test_map50_95'],'0.3')
            self.assertIn('30.00%', (directory/'SUMMARY.md').read_text(encoding='utf-8'))

    def test_rejects_missing_local_model_and_per_dataset_model_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'batch.yaml'
            config.write_text('model: model.pt\ndatasets: [{data: data.yaml, params: {model: other.pt}}]\n')
            with self.assertRaisesRegex(ValueError,'existing local'):
                load_config(config)
            (root/'model.pt').touch()
            (root/'data.yaml').touch()
            with self.assertRaisesRegex(ValueError,'unsupported parameters: model'):
                load_config(config)
