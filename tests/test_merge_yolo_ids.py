import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import yaml

from merge_yolo_ids import merge, parse_reserved_ids


def dataset(root, names, rows, split_list=False):
    root.mkdir()
    for split in ('train', 'val', 'test'):
        (root/'images'/split).mkdir(parents=True)
        (root/'labels'/split).mkdir(parents=True)
        (root/'images'/split/'same_name.jpg').write_bytes((root.name+split).encode())
        (root/'labels'/split/'same_name.txt').write_text(rows)
    config = {'names': names, 'train': 'images/train', 'val': 'images/val', 'test': 'images/test'}
    if split_list:
        (root/'train.txt').write_text('./images/train/same_name.jpg\n')
        config['train'] = ['train.txt']
    (root/'data.yaml').write_text(yaml.safe_dump(config))
    return root


class MergeIDsTests(unittest.TestCase):
    def call_merge(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return merge(*args, **kwargs)

    def test_real_83_class_case_reserved_padding_hardlinks_and_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dataset(root/'a', {i:f'a{i}' for i in range(27)}, '26 0.5 0.5 0.2 0.3\n')
            bnames = {i:f'reserved_{i}' if i < 27 else f'b{i}' for i in range(83)}
            b = dataset(root/'b', bnames, '27 0.5 0.5 0.2 0.3\n82 0.4 0.4 0.2 0.2\n')
            report = self.call_merge([a, b], root/'out', reserved_ids={2:set(range(27))})
            config = yaml.safe_load((root/'out/data.yaml').read_text())
            self.assertEqual(config['nc'], 83)
            self.assertEqual(config['names'][0], 'a0')
            self.assertEqual(config['names'][27], 'b27')
            self.assertEqual(report['ignored_reserved_ids']['d2'], list(range(27)))
            self.assertFalse((root/'out/INCOMPLETE.txt').exists())
            for split in ('train','val','test'):
                images = list((root/'out/images'/split).iterdir())
                self.assertEqual(len(images), 2)
                for image in images:
                    source = a if image.name.startswith('d1_') else b
                    self.assertTrue(os.path.samefile(image, source/'images'/split/'same_name.jpg'))
                    label = root/'out/labels'/split/(image.stem+'.txt')
                    self.assertEqual(label.read_bytes(), (source/'labels'/split/'same_name.txt').read_bytes())
            self.assertEqual(report['class_image_counts']['train'][82], 1)
            self.assertEqual(report['class_box_counts']['test'][27], 1)

    def test_general_three_sources_sparse_names_lists_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dataset(root/'a', ['cat'], '', split_list=True)
            b = dataset(root/'b', {1:'dog'}, '1 0.5 0.5 0.2 0.2\n')
            c = dataset(root/'c', {2:'bird'}, '2 0.5 0.5 0.2 0.2\n')
            report = self.call_merge([a/'data.yaml', b, c], root/'out', copy_images=True)
            self.assertEqual(report['names'], {0:'cat',1:'dog',2:'bird'})
            self.assertEqual(report['splits']['train']['negative_images'], 1)
            self.assertEqual(len(list((root/'out/images/train').iterdir())),3)
            self.assertFalse(any(os.path.samefile(p,b/'images/train/same_name.jpg') for p in (root/'out/images/train').iterdir()))

    def test_conflicting_real_ids_and_gaps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dataset(root/'a', {0:'cat'}, '')
            b = dataset(root/'b', {0:'dog'}, '')
            with self.assertRaisesRegex(ValueError, 'Conflicting ID'):
                self.call_merge([a,b],root/'out')
            (b/'data.yaml').write_text(yaml.safe_dump({'names':{2:'dog'}}))
            with self.assertRaisesRegex(ValueError, 'continuous from 0'):
                self.call_merge([a,b],root/'out')
            self.assertFalse((root/'out').exists())

    def test_used_reserved_unknown_or_missing_labels_rejected(self):
        for row in ('0 0.5 0.5 0.2 0.2\n','9 0.5 0.5 0.2 0.2\n',None):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a = dataset(root/'a',{0:'cat'},'')
                b = dataset(root/'b',{0:'reserved_0',1:'dog'},row or '')
                if row is None:
                    (b/'labels/train/same_name.txt').unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):
                    self.call_merge([a,b],root/'out', reserved_ids={2:{0}})
                self.assertFalse((root/'out').exists())

    def test_shared_hardlink_across_splits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dataset(root/'a',{0:'cat'},'')
            b = dataset(root/'b',{1:'dog'},'')
            image=b/'images/test/same_name.jpg'
            image.unlink()
            os.link(a/'images/train/same_name.jpg',image)
            with self.assertRaisesRegex(ValueError,'different splits'):
                self.call_merge([a,b],root/'out')

    def test_real_reserved_names_are_never_automatically_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'reserved',1:'reserved_1'},'0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n')
            b=dataset(root/'b',{2:'dog'},'2 0.5 0.5 0.2 0.2\n')
            report=self.call_merge([a,b],root/'out')
            self.assertEqual(report['names'],{0:'reserved',1:'reserved_1',2:'dog'})
            self.assertEqual(report['ignored_reserved_ids'],{'d1':[],'d2':[]})
            self.assertEqual(report['class_box_counts']['train'][1],1)

    def test_padding_name_conflict_requires_explicit_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'cat'},'0 0.5 0.5 0.2 0.2\n')
            b=dataset(root/'b',{0:'reserved_0',1:'dog'},'1 0.5 0.5 0.2 0.2\n')
            with self.assertRaisesRegex(ValueError,'Conflicting ID'):
                self.call_merge([a,b],root/'out')
            self.assertFalse((root/'out').exists())
            result=self.call_merge([a,b],root/'out',reserved_ids={2:{0,99}})
            self.assertEqual(result['ignored_reserved_ids']['d2'],[0])
            self.assertEqual(result['absent_reserved_ids']['d2'],[99])

    def test_requested_range_intersects_yaml_classes_0_to_28(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{i:f'a{i}' for i in range(29)},'26 0.5 0.5 0.2 0.2\n')
            b=dataset(root/'b',{i:f'b{i}' for i in range(83)},'27 0.5 0.5 0.2 0.2\n82 0.5 0.5 0.2 0.2\n')
            reserved=parse_reserved_ids(['1:27-82','2:0-26'],2)
            result=self.call_merge([a,b],root/'out',reserved_ids=reserved)
            self.assertEqual(result['names'],{**{i:f'a{i}' for i in range(27)},**{i:f'b{i}' for i in range(27,83)}})
            self.assertEqual(result['ignored_reserved_ids']['d1'],[27,28])
            self.assertEqual(result['absent_reserved_ids']['d1'],list(range(29,83)))
            config=yaml.safe_load((root/'out/data.yaml').read_text())
            self.assertEqual(config['nc'],83)
            labels=[p.read_text() for p in (root/'out/labels/train').iterdir()]
            self.assertEqual(set(labels),{'26 0.5 0.5 0.2 0.2\n','27 0.5 0.5 0.2 0.2\n82 0.5 0.5 0.2 0.2\n'})

    def test_range_still_rejects_labels_using_ignored_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'cat',1:'old'},'1 0.5 0.5 0.2 0.2\n')
            b=dataset(root/'b',{1:'dog'},'1 0.5 0.5 0.2 0.2\n')
            with self.assertRaisesRegex(ValueError,'unknown/reserved ID 1'):
                self.call_merge([a,b],root/'out',reserved_ids={1:set(range(1,83))})
            self.assertFalse((root/'out').exists())

    def test_reserved_arguments_are_scoped_to_source(self):
        self.assertEqual(parse_reserved_ids(['2:0-2','2:4','3:0,6-7'],3),{2:{0,1,2,4},3:{0,6,7}})
        for spec in ('0:1','4:0','2:9-1','2:-1','2:cat'):
            with self.subTest(spec=spec),self.assertRaises(ValueError):
                parse_reserved_ids([spec],3)

    def test_same_real_id_same_name_allowed_and_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a=dataset(root/'a',{0:'cat'},'0 0.5 0.5 0.2 0.2\n')
            b=dataset(root/'b',{0:'cat'},'0 0.5 0.5 0.2 0.2\n')
            self.call_merge([a,b],root/'out')
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['class_image_counts']['train']['0'],2)
            with self.assertRaisesRegex(ValueError,'new output directory'):
                self.call_merge([a,b],root/'out')

    def test_single_source_subset_compact_ids_coordinates_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{i:f'class{i}' for i in range(27)},
                      '0 0.50000 0.5 0.2 0.3\n1 0.5 0.5 0.1 0.1\n25 0.4 0.4 0.2 0.2\n26 0.6 0.6 0.1 0.1\n')
            before=(a/'labels/train/same_name.txt').read_bytes()
            with self.assertRaisesRegex(ValueError,'compact-ids'):
                self.call_merge([a],root/'out',include_ids={1:{0,25,26}})
            report=self.call_merge([a],root/'out',include_ids={1:{0,25,26}},compact_ids=True)
            self.assertEqual(report['names'],{0:'class0',1:'class25',2:'class26'})
            self.assertEqual(report['id_mapping'],{0:0,25:1,26:2})
            self.assertFalse(report['class_ids_preserved'])
            label=next((root/'out/labels/train').iterdir())
            self.assertEqual(label.read_text(),'0 0.50000 0.5 0.2 0.3\n1 0.4 0.4 0.2 0.2\n2 0.6 0.6 0.1 0.1\n')
            self.assertEqual(report['class_box_counts']['train'],{0:1,1:1,2:1})
            self.assertEqual((a/'labels/train/same_name.txt').read_bytes(),before)
            self.assertTrue(os.path.samefile(next((root/'out/images/train').iterdir()),a/'images/train/same_name.jpg'))
            self.assertIn('| 25 | 1 | class25 |',(root/'out/MERGE_REPORT.md').read_text(encoding='utf-8'))

    def test_filter_scope_does_not_drop_other_sources_or_renumber_global_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{i:f'a{i}' for i in range(27)},'25 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n')
            b=dataset(root/'b',{i:f'b{i}' for i in range(1,25)},'1 0.5 0.5 0.2 0.2\n')
            report=self.call_merge([a,b],root/'out',include_ids={1:{0,25,26}})
            self.assertEqual(report['names'][1],'b1')
            self.assertEqual(report['names'][25],'a25')
            self.assertEqual(report['names'][26],'a26')
            self.assertTrue(report['class_ids_preserved'])
            self.assertEqual(report['splits']['train']['images'],2)
            labels={p.read_text() for p in (root/'out/labels/train').iterdir()}
            self.assertEqual(labels,{'25 0.5 0.5 0.2 0.2\n','1 0.5 0.5 0.2 0.2\n'})

    def test_filter_excludes_nonselected_images_and_negatives_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'cat',1:'dog'},'0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n')
            for name,row in [('other','1 0.5 0.5 0.1 0.1\n'),('negative','')]:
                (a/f'images/train/{name}.jpg').write_bytes(name.encode())
                (a/f'labels/train/{name}.txt').write_text(row)
            report=self.call_merge([a],root/'out',include_ids={1:{0}})
            self.assertEqual(report['filtering']['d1']['train'],{
                'scanned_images':3,'kept_images':1,'excluded_without_selected_class':1,
                'excluded_negatives':1,'removed_boxes':2})
            report=self.call_merge([a],root/'with_negatives',include_ids={1:{0}},keep_negatives=True)
            self.assertEqual(report['splits']['train'],{'images':2,'negative_images':1})
            self.assertEqual(report['class_image_counts']['train'][0],1)

    def test_unknown_includes_reserved_overlap_and_empty_split_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'cat',1:'dog'},'1 0.5 0.5 0.1 0.1\n')
            for options,message in [({'include_ids':{1:{99}}},'must exist'),
                                    ({'include_ids':{2:{0}}},'source index'),
                                    ({'include_ids':{1:{0}},'reserved_ids':{1:{0}}},'overlap'),
                                    ({'include_ids':{1:{0}}},'non-empty train and val')]:
                with self.subTest(options=options),self.assertRaisesRegex(ValueError,message):
                    self.call_merge([a],root/'out',**options)
                self.assertFalse((root/'out').exists())

    def test_actual_cli_single_source_with_include_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dataset(root/'a',{0:'cat',25:'dog',26:'bird'},'25 0.5 0.5 0.2 0.2\n')
            script=Path(__file__).resolve().parents[1]/'merge_yolo_ids.py'
            result=subprocess.run([sys.executable,'-X','utf8',str(script),'--datasets',str(a),'--include-ids','1:0,25,26',
                                   '--compact-ids','--output',str(root/'out')],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            config=yaml.safe_load((root/'out/data.yaml').read_text())
            self.assertEqual(config['names'],{0:'cat',1:'dog',2:'bird'})
            self.assertEqual(next((root/'out/labels/train').iterdir()).read_text(),'1 0.5 0.5 0.2 0.2\n')


if __name__ == '__main__':
    unittest.main()
