import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from split_yolo import assign_splits, run_batch, target_counts


def dataset(root, layout='flat', count=20, stale=False):
    root.mkdir()
    names = {0:'cat', 1:'dog'}
    for i in range(count):
        old = 'train' if i % 2 == 0 else 'val'
        if layout == 'flat':
            relative = Path('images')/f'{i}.jpg'
        elif layout == 'inside':
            relative = Path('images')/old/f'{i}.jpg'
        elif layout in ('outside','outside_complete'):
            relative = Path(('train','val','test')[i%3] if layout=='outside_complete' else old)/'images'/f'{i}.jpg'
        else:
            relative = Path('images')/('train','val','test')[i%3]/f'{i}.jpg'
        image = root/relative
        label = root/Path(*['labels' if part == 'images' else part for part in relative.parts]).with_suffix('.txt')
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        image.write_bytes(f'image{i}'.encode())
        label.write_bytes(b'' if i%7==0 else f'{i%2} 0.50000 0.5 0.2 0.3\n'.encode())
    config = {'names':names,'train':'images/train','val':'images/val','test':'images/test'}
    if stale:
        config['path'] = str(root.parent/'old_machine_path')
    (root/'data.yaml').write_text(yaml.safe_dump(config))
    return root


class SplitTests(unittest.TestCase):
    def run_quiet(self,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return run_batch(*args,**kwargs)

    def test_batch_flat_partial_complete_and_roboflow_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources=[dataset(root/name,layout,stale=(layout=='flat'))
                     for name,layout in [('a','flat'),('b','inside'),('c','outside'),('d','complete')]]
            results=self.run_quiet(sources,root/'out',workers=2)
            self.assertEqual([r['status'] for r in results],['ok']*4)
            for source,result in zip(sources,results):
                report=result['report']
                output=Path(report['output'])
                self.assertEqual(report['total_images'],20)
                self.assertEqual([report['splits'][s]['images'] for s in ('train','val','test')],[16,2,2])
                self.assertFalse(report['source_splits_preserved'])
                self.assertEqual(report['names'],{0:'cat',1:'dog'})
                config=yaml.safe_load((output/'data.yaml').read_text())
                self.assertEqual(config['test'],'images/test')
                self.assertEqual(len(list(output.glob('images/*/*.jpg'))),20)
                originals={p.read_bytes():p for p in source.rglob('*.jpg')}
                for image in output.glob('images/*/*.jpg'):
                    original=originals[image.read_bytes()]
                    self.assertTrue(os.path.samefile(image,original))
                    original_label=Path(*['labels' if p=='images' else p for p in original.parts]).with_suffix('.txt')
                    target_label=output/'labels'/image.parent.name/(image.stem+'.txt')
                    self.assertEqual(original_label.read_bytes(),target_label.read_bytes())
                self.assertFalse((output/'INCOMPLETE.txt').exists())
                self.assertIn('各类别统计',(output/'SPLIT_REPORT.md').read_text(encoding='utf-8'))
            self.assertFalse((root/'out/INCOMPLETE.txt').exists())
            self.assertEqual(len(json.loads((root/'out/report.json').read_text(encoding='utf-8'))),4)
            self.assertTrue((root/'out/BATCH_REPORT.md').is_file())

    def test_seed_reproducible_and_old_splits_are_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dataset(root/'a','complete',count=30)
            first=self.run_quiet([source],root/'one',seed=42,workers=2)[0]['report']
            second=self.run_quiet([source],root/'two',seed=42,workers=1)[0]['report']
            third=self.run_quiet([source],root/'three',seed=17,workers=2)[0]['report']
            def distribution(report):
                return {p.name:p.parent.name for p in Path(report['output']).glob('images/*/*.jpg')}
            self.assertEqual(distribution(first),distribution(second))
            self.assertNotEqual(distribution(first),distribution(third))
            self.assertEqual([first['splits'][s]['images'] for s in ('train','val','test')],[24,3,3])

    def test_shared_hardlinks_keep_all_paths_in_one_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dataset(root/'a',count=20)
            original=source/'images/0.jpg'
            twin=source/'images/duplicate.jpg'
            os.link(original,twin)
            (source/'labels/duplicate.txt').write_bytes((source/'labels/0.txt').read_bytes())
            result=self.run_quiet([source],root/'out',workers=2)[0]['report']
            twins=[p for p in Path(result['output']).glob('images/*/*.jpg') if os.path.samefile(p,original)]
            self.assertEqual(len(twins),2)
            self.assertEqual(len({p.parent.name for p in twins}),1)
            self.assertEqual(result['total_images'],21)
            self.assertEqual(result['independent_file_groups'],20)

    def test_failure_is_reported_other_datasets_continue_and_sources_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bad=dataset(root/'bad')
            good=dataset(root/'good')
            (bad/'labels/1.txt').unlink()
            results=self.run_quiet([bad,good],root/'out',workers=2)
            self.assertEqual([r['status'] for r in results],['failed','ok'])
            self.assertEqual(len(list(good.glob('images/*.jpg'))),20)
            self.assertTrue((root/'out/INCOMPLETE.txt').exists())
            self.assertIn('失败',(root/'out/BATCH_REPORT.md').read_text(encoding='utf-8'))

    def test_bad_yaml_is_reported_without_stopping_valid_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bad=dataset(root/'bad')
            good=dataset(root/'good')
            (bad/'data.yaml').write_text('names: {}\n')
            results=self.run_quiet([bad,good],root/'out',workers=1)
            self.assertEqual([r['status'] for r in results],['failed','ok'])

    def test_copy_mode_txt_lists_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dataset(root/'a')
            (source/'train.txt').write_text('\n'.join(f'images/{i}.jpg' for i in range(20)))
            (source/'data.yaml').write_text(yaml.safe_dump({'names':['cat','dog'],'train':'train.txt'}))
            result=self.run_quiet([source/'data.yaml'],root/'out',copy_images=True,workers=2)[0]['report']
            self.assertEqual(result['total_images'],20)  # Local directory and txt do not duplicate entries.
            self.assertEqual(result['image_mode'],'copy')
            target=next(Path(result['output']).glob('images/*/*.jpg'))
            original=next(p for p in source.glob('images/*.jpg') if p.read_bytes()==target.read_bytes())
            self.assertFalse(os.path.samefile(original,target))
            with self.assertRaisesRegex(ValueError,'new output'):
                self.run_quiet([source],root/'out')
            with self.assertRaisesRegex(ValueError,'separate'):
                self.run_quiet([source],source/'output')

    def test_too_few_groups_and_invalid_ratios(self):
        for ratios in [(8,1,0),(1,float('nan'),1),(1e308,1e308,1e308)]:
            with self.subTest(ratios=ratios),self.assertRaises(ValueError):
                target_counts(20,ratios)
        self.assertEqual(target_counts(3,(8,1,1)),{'train':1,'val':1,'test':1})
        with self.assertRaisesRegex(ValueError,'independent'):
            assign_splits([{'identity':'same'} for _ in range(5)],(8,1,1),42)

    def test_actual_cli_outputs_batch_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dataset(root/'a')
            script=Path(__file__).resolve().parents[1]/'split_yolo.py'
            result=subprocess.run([sys.executable,'-X','utf8',str(script),'--datasets',str(source),
                                   '--output',str(root/'out'),'--workers','2'],capture_output=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('1 OK, 0 failed',result.stdout)
            self.assertTrue((root/'out/BATCH_REPORT.md').exists())

    def test_no_yaml_layouts_preserve_numeric_ids_without_fabricating_names(self):
        for layout in ('flat','inside','outside_complete'):
            with self.subTest(layout=layout),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                source=dataset(root/'merged6_obstacle_only',layout,count=30)
                (source/'data.yaml').unlink()
                for path in source.rglob('*.txt'):
                    row=path.read_text()
                    if row:
                        fields=row.split()
                        fields[0]='7' if fields[0]=='0' else '27'
                        path.write_text(' '.join(fields)+'\n')
                original_labels=sorted(p.read_bytes() for p in source.rglob('*.txt'))
                result=self.run_quiet([source],root/'out',workers=2)[0]
                self.assertEqual(result['status'],'ok',result)
                report=result['report']
                self.assertEqual(report['observed_class_ids'],[7,27])
                self.assertEqual(report['names'],{})
                self.assertFalse(report['class_names_available'])
                output=Path(report['output'])
                self.assertFalse((output/'data.yaml').exists())
                self.assertEqual(sorted(p.read_bytes() for p in output.glob('labels/*/*.txt')),original_labels)
                self.assertIn('| 27 | —（未提供名称） |',(output/'SPLIT_REPORT.md').read_text(encoding='utf-8'))
                self.assertEqual([report['splits'][s]['images'] for s in ('train','val','test')],[24,3,3])

    def test_no_yaml_invalid_rows_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dataset(root/'no_yaml','outside_complete')
            (source/'data.yaml').unlink()
            (source/'train/labels/3.txt').write_text('999 0.5 0.5 -0.2 0.3\n')
            results=self.run_quiet([source],root/'out',workers=2)
            self.assertEqual(results[0]['status'],'failed')
            self.assertIn('Invalid normalized box',results[0]['error'])


if __name__=='__main__':
    unittest.main()
