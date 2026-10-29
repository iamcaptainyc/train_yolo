import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from split_yolo_inplace import WORK, recover, run_in_place_batch, split_in_place
from test_split_yolo import dataset


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.name != '.split_yolo.lock' and WORK not in p.parts}


class InPlaceTests(unittest.TestCase):
    def quiet(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return run_in_place_batch(*args,**kwargs)

    def test_in_place_flat_and_old_split_layouts_keep_bytes_inodes_and_counts(self):
        for layout in ('flat','inside','outside','complete'):
            with self.subTest(layout=layout),tempfile.TemporaryDirectory() as tmp:
                root=dataset(Path(tmp)/'data',layout,count=20)
                old_images={p.read_bytes():(p.stat().st_dev,p.stat().st_ino) for p in root.rglob('*.jpg')}
                old_labels=sorted(p.read_bytes() for p in root.rglob('*.txt'))
                result=self.quiet([root],workers=2)
                self.assertEqual(result[0]['status'],'ok',result)
                images=list(root.glob('images/*/*.jpg'))
                self.assertEqual(len(images),20)
                for image in images:
                    self.assertEqual((image.stat().st_dev,image.stat().st_ino),old_images[image.read_bytes()])
                self.assertEqual(sorted(p.read_bytes() for p in root.glob('labels/*/*.txt')),old_labels)
                self.assertFalse((root/WORK).exists())
                self.assertEqual(len(list(root.rglob('*.jpg'))),20)
                self.assertEqual([len(list((root/'images'/s).glob('*.jpg'))) for s in ('train','val','test')],[16,2,2])
                config=yaml.safe_load((root/'data.yaml').read_text())
                self.assertEqual(config['path'],root.as_posix())
                self.assertEqual(config['names'],{0:'cat',1:'dog'})
                report=json.loads((root/'split_report.json').read_text())
                self.assertTrue(report['in_place'])
                self.assertEqual(report['output'],str(root))

    def test_auxiliary_files_and_orphan_labels_preserved_caches_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=dataset(Path(tmp)/'data','outside')
            (root/'train/images/license.json').write_text('{"license":"test"}')
            (root/'train/labels/orphan.txt').write_text('unrelated metadata')
            (root/'val/labels/labels.cache').write_bytes(b'stale')
            (root/'report.json').write_text('{"original_report":true}')
            self.assertEqual(self.quiet([root],workers=2)[0]['status'],'ok')
            self.assertEqual((root/'train/images/license.json').read_text(),'{"license":"test"}')
            self.assertEqual((root/'train/labels/orphan.txt').read_text(),'unrelated metadata')
            self.assertFalse((root/'val/labels/labels.cache').exists())
            self.assertEqual((root/'report.json').read_text(),'{"original_report":true}')
            self.assertEqual(len(list(root.rglob('*.jpg'))),20)

    def test_publish_failure_restores_every_original_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=dataset(Path(tmp)/'data','inside')
            before=snapshot(root)
            original=Path.rename
            def fail_labels(path,target):
                if path.parent == root/WORK/'prepared' and path.name == 'labels':
                    raise OSError('simulated publication failure')
                return original(path,target)
            with patch.object(Path,'rename',fail_labels):
                result=self.quiet([root],workers=2)
            self.assertEqual(result[0]['status'],'failed')
            self.assertEqual(snapshot(root),before)
            self.assertFalse((root/WORK).exists())

    def test_hard_kill_mid_publish_same_cli_recovers_missing_yaml_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=dataset(Path(tmp)/'data')
            project=Path(__file__).resolve().parents[1]
            code='''import os,sys
from pathlib import Path
from split_yolo_inplace import split_in_place,WORK
root=Path(sys.argv[1])
original=Path.rename
def killed(path,target):
    if path.parent == root/WORK/'prepared' and path.name == 'labels':
        os._exit(91)
    return original(path,target)
Path.rename=killed
split_in_place(root/'data.yaml',workers=2)
'''
            child=subprocess.run([sys.executable,'-X','utf8','-c',code,str(root)],cwd=project,
                                 capture_output=True,encoding='utf-8')
            self.assertEqual(child.returncode,91,child.stderr)
            self.assertFalse((root/'data.yaml').exists())
            self.assertTrue((root/WORK/'transaction.json').exists())
            # Run the public CLI, not an internal recovery-only shortcut.
            retry=subprocess.run([sys.executable,'-X','utf8',str(project/'split_yolo.py'),
                                  '--in-place','--datasets',str(root),'--workers','2'],
                                 capture_output=True,encoding='utf-8')
            self.assertEqual(retry.returncode,0,retry.stderr)
            self.assertTrue((root/'data.yaml').exists())
            self.assertFalse((root/WORK).exists())
            self.assertEqual(len(list(root.rglob('*.jpg'))),20)

    def test_build_interruption_leaves_originals_and_next_attempt_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=dataset(Path(tmp)/'data')
            before=snapshot(root)
            with patch('split_yolo_inplace.split_dataset',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.quiet([root],workers=2)
            self.assertEqual(snapshot(root),before)
            self.assertFalse((root/WORK).exists())
            self.assertEqual(self.quiet([root],workers=2)[0]['status'],'ok')

    def test_external_images_rejected_without_modifying_either_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            root=dataset(parent/'local')
            external=dataset(parent/'external')
            config=yaml.safe_load((root/'data.yaml').read_text())
            config['path']=str(external)
            (root/'data.yaml').write_text(yaml.safe_dump(config))
            before_local,before_external=snapshot(root),snapshot(external)
            result=self.quiet([root],workers=2)
            self.assertEqual(result[0]['status'],'failed')
            self.assertEqual(snapshot(root),before_local)
            self.assertEqual(snapshot(external),before_external)

    def test_custom_yaml_updated_and_batch_failure_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            bad=dataset(parent/'bad')
            good=dataset(parent/'good')
            (bad/'labels/1.txt').unlink()
            (good/'data.yaml').rename(good/'data.yml')
            result=self.quiet([bad,good],workers=2)
            self.assertEqual([r['status'] for r in result],['failed','ok'])
            self.assertEqual((good/'data.yml').read_bytes(),(good/'data.yaml').read_bytes())
            self.assertFalse((bad/WORK).exists())

    def test_unsafe_journal_and_nested_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            root=dataset(parent/'data')
            work=root/WORK
            work.mkdir()
            (work/'transaction.json').write_text(json.dumps({'version':1,'root':str(root),
                'phase':'switching','old':['../outside'],'new':[]}))
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                recover(root)
            self.assertTrue((root/'data.yaml').exists())
            child=root/'child'
            child.mkdir()
            with self.assertRaisesRegex(ValueError,'non-nested'):
                self.quiet([root,child],workers=2)


if __name__=='__main__':
    unittest.main()
