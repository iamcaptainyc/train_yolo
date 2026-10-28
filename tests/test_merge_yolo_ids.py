import contextlib
import io
import json
import os
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
            with self.assertRaisesRegex(ValueError,'not declared'):
                self.call_merge([a,b],root/'out',reserved_ids={2:{99}})

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


if __name__ == '__main__':
    unittest.main()
