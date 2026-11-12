import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from audit_light_ids import main
from merge_yolo_ids import merge


def make_dataset(root, names, rows):
    for split in ('train', 'val', 'test'):
        (root/'images'/split).mkdir(parents=True)
        (root/'labels'/split).mkdir(parents=True)
        (root/'images'/split/'image.jpg').write_bytes(split.encode())
        (root/'labels'/split/'image.txt').write_text(rows if split != 'test' else rows.splitlines()[0] + '\n')
    (root/'data.yaml').write_text(yaml.safe_dump({
        'names': names, 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
    }))


class AuditLightIDsTests(unittest.TestCase):
    def test_audit_checks_source_mapping_checkpoint_and_label_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, merged = root/'source', root/'merged'
            row = '0 0.5 0.5 0.2 0.2\n'
            make_dataset(source, {0:'red_left', 1:'green_straight', 2:'cat'}, row * 2)
            output = root/'audit.json'
            checkpoint = root/'best.pt'
            checkpoint.touch()
            fake_load = Mock(return_value={'ema': SimpleNamespace(
                names={0:'cat', 1:'green_straight', 2:'red_left'},
            )})
            args = ['audit_light_ids.py', '--source', str(source), '--merged', str(merged/'data.yaml'),
                    '--weights', str(checkpoint), '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                merge([source], merged, id_map={0:2, 2:0})
                (source/'labels/val/image.txt').write_text(row * 2 + '8 0.5 0.5 0.2 0.2\n')
                with patch('sys.argv', args), patch.dict('sys.modules', {'torch': SimpleNamespace(load=fake_load)}):
                    main()
            fake_load.assert_called_once_with(checkpoint.resolve(), map_location='cpu', weights_only=False)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual([row['name'] for row in result['classes']], ['green_straight', 'red_left'])
            red = result['classes'][1]
            self.assertEqual(red['sources'], [{'ids':[0], 'expected_merged_ids':[2], 'status':'mapped'}])
            self.assertEqual(red['checkpoint_vs_merged'], 'match')
            self.assertEqual(result['sources'][0]['splits']['val']['boxes_by_id'], {'0':2})
            self.assertEqual(result['sources'][0]['splits']['val']['issues'], {'unknown_id_rows':1})
            self.assertEqual(result['merged']['splits']['train']['boxes_by_id'], {'2':2})
            self.assertEqual(result['merged']['splits']['test']['boxes_by_id'], {'2':1})

            fake_load.return_value = {'model': SimpleNamespace(names={0:'red_left', 1:'green_straight'})}
            args[-1] = str(root/'mismatch.json')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with patch('sys.argv', args), patch.dict('sys.modules', {'torch': SimpleNamespace(load=fake_load)}):
                    main()
            result = json.loads((root/'mismatch.json').read_text(encoding='utf-8'))
            self.assertEqual(result['classes'][1]['checkpoint_vs_merged'], 'mismatch')
