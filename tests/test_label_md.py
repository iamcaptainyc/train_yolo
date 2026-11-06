import argparse
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import yaml

from label_md import generate, animal_class


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class Detector:
    names = {0:'animal', 1:'person', 2:'vehicle'}

    def predict(self, source, **kwargs):
        return [SimpleNamespace(path=p, boxes=[] if Path(p).stem=='empty' else [
            SimpleNamespace(cls=Scalar(0), conf=Scalar(.9), xywhn=[Scalar([.5,.5,.4,.6])]),
            SimpleNamespace(cls=Scalar(1), conf=Scalar(.9), xywhn=[Scalar([.5,.5,.2,.2])])
        ]) for p in source]


class LabelMDTests(unittest.TestCase):
    def test_folder_labels_resplit_filter_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'source'/'validation'/'Wild Boar'
            folder.mkdir(parents=True)
            for name in [str(i) for i in range(10)]+['empty']:
                Image.new('RGB', (64,64)).save(folder/(name+'.jpg'))
            os.link(folder/'0.jpg', folder/'alias.jpg')
            tiger = root/'source'/'train'/'Tiger'
            tiger.mkdir(parents=True)
            for i in range(10):
                Image.new('RGB', (64,64), (100,0,0)).save(tiger/f'{i}.jpg')
            weights = root/'md.pt'
            weights.write_bytes(b'mock')
            args = argparse.Namespace(input=root/'source', output=root/'output', weights=weights,
                conf=.3, batch=4, imgsz=1280, seed=42, device='cpu', animal_class_id=None, copy_images=False)
            report = generate(args, Detector())
            self.assertEqual(report['split_images'], {'train':16,'val':2,'test':2})
            self.assertEqual(report['class_boxes'], {'tiger':10,'wild_boar':10})
            self.assertEqual(report['skipped_no_detection'], {'wild_boar':1})
            config = yaml.safe_load((args.output/'data.yaml').read_text())
            self.assertEqual(config['names'], {0:'tiger',1:'wild_boar'})
            self.assertEqual(len(list((args.output/'labels').rglob('*.txt'))), 20)
            self.assertFalse((args.output/'INCOMPLETE.txt').exists())
            image = next((args.output/'images').rglob('*.jpg'))
            self.assertTrue(any(os.path.samefile(image, p) for p in (root/'source').rglob('*.jpg')))
            ids = {p.read_text().split()[0] for p in (args.output/'labels').rglob('*.txt')}
            self.assertEqual(ids, {'0','1'})

    def test_ambiguous_model_requires_explicit_id(self):
        with self.assertRaises(ValueError):
            animal_class(SimpleNamespace(names={0:'0', 1:'1'}))
        self.assertEqual(animal_class(SimpleNamespace(names={0:'0',1:'1'}), 1), 1)


if __name__ == '__main__':
    unittest.main()
