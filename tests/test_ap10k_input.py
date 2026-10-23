from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ap10k_input import AP_NAMES, load_ap10k, scan_ap10k, rows_from_coco
from camping_subset import filter_lila_area, sample_global
from merge_yolo_datasets import merge
from merge_yolo_split import resplit
from test_merge_yolo_datasets import fixture
from test_camping_subset import record


def ap_fixture(root, species=('dog',), count=4):
    (root/'annotations').mkdir(parents=True)
    (root/'data').mkdir()
    cats=[{'id':i+1,'name':n} for i,n in enumerate(species)]
    for split in ('train','val','test'):
        images=[]; annotations=[]
        for i in range(count):
            image_id = ('train','val','test').index(split)*count+i
            name=f'{image_id:012d}.jpg'
            (root/'data'/name).write_bytes(f'AP unique {image_id}'.encode())
            images.append({'id':image_id,'file_name':name,'width':100,'height':100})
            for cid in range(1,len(cats)+1):
                annotations.append({'id':len(annotations),'image_id':image_id,'category_id':cid,
                                    'bbox':[10,20,40,50],'iscrowd':0})
        data={'images':images,'annotations':annotations,'categories':cats}
        for scheme in (1,2,3):
            (root/'annotations'/f'ap10k-{split}-split{scheme}.json').write_text(json.dumps(data))


class AP10KTests(unittest.TestCase):
    def test_one_scheme_only_and_parent_root_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'ap-10k';ap_fixture(root)
            d=load_ap10k(Path(tmp))
            self.assertEqual(sum(map(len,d['loaded'].values())),12)
            self.assertEqual(len(d['manifest']),3)
            self.assertTrue(all('split1' in v['path'] for v in d['manifest'].values()))
            self.assertEqual(len(AP_NAMES),54)
            image,ann=d['loaded']['train'][0]
            r=scan_ap10k((d,'train',image,ann))
            self.assertNotIn('error',r)
            self.assertEqual(r['rows'][0][1],('0.3000000000','0.4500000000','0.4000000000','0.5000000000'))

    def test_invalid_or_unsupported_box_excludes_entire_image(self):
        image={'width':100,'height':100}
        base={'category_id':1,'bbox':[10,20,30,40]}
        for ann in ({**base,'bbox':[-1,0,30,40]}, {**base,'iscrowd':1}, {**base,'bbox':[90,0,30,40]}):
            with self.assertRaises(ValueError): rows_from_coco(image,[ann],{1:'dog'},{1:'dog'})
        with self.assertRaisesRegex(ValueError,'WHOLE'):
            rows_from_coco(image,[base,{**base,'category_id':2}],{1:'dog',2:'panther'},{1:'dog',2:None})
        with self.assertRaises(ValueError): rows_from_coco(image,[],{1:'dog'},{1:'dog'})

    def test_lila_area_any_box_strict_threshold_no_dropped_small_boxes(self):
        r=record(0,['rabbit','mouse'],'lila')
        r['rows']=[('rabbit',('.5','.5','.5','.5')),('mouse',('.5','.5','.1','.1'))]
        equal=record(1,'rabbit','lila');equal['rows']=[('rabbit',('.5','.5','.25','.25'))]
        tiny=record(2,'rabbit','lila')
        ap=record(3,'rabbit','ap10k')
        negative=record(4,None,'lila')
        kept,rejected,stats=filter_lila_area([r,equal,tiny,ap,negative],1/16)
        self.assertEqual([x['relative'] for x in kept],['0','3','4'])
        self.assertEqual(len(r['rows']),2)
        self.assertEqual(len(rejected),2)
        self.assertIn('ANY',stats['multi_box_rule'])

    def test_ap_then_animals_then_lila_global_priority_before_split(self):
        rows=[record(i,'rabbit','lila',split='val') for i in range(20)]
        rows += [record(100+i,'rabbit','animals',split='test') for i in range(4)]
        rows += [record(200+i,'rabbit','ap10k',split='train') for i in range(3)]
        selected,_,stats=sample_global(rows,8,0,100,42)
        self.assertEqual(Counter(r['dataset'] for r in selected),{'ap10k':3,'animals':4,'lila':1})
        self.assertEqual(stats['source_priority'],['ap10k','animals','lila'])
        self.assertEqual([r['split'] for r in rows[:20]],['val']*20)

    def test_full_pool_transitive_group_survives_removed_bridge(self):
        rows=[record(i,'rabbit') for i in range(6)]
        for i,r in enumerate(rows):
            r['image_stamp']={'device':1,'inode':i+10}
            r['group_keys']=[('single',i)]
            r['image_rel']=Path('images/train/animals')/f'{i}.jpg'
            r['label_rel']=Path('labels/train/animals')/f'{i}.txt'
        rows[0]['group_keys']=[('camera','a')]
        rows[1]['group_keys']=[('camera','a'),('sequence','b')]
        rows[2]['group_keys']=[('sequence','b')]
        resplit(rows,.05,.05,42,groups_only=True)
        selected=[rows[i] for i in (0,2,3,4,5)]
        result=resplit(selected,.05,.05,42)
        self.assertEqual(selected[0]['split'],selected[1]['split'])
        self.assertEqual(result['group_overlap'],0)

    def test_three_source_end_to_end_native_audit_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ap_fixture(root/'ap-10k')
            fixture(root/'lila',['dog'],'lila');fixture(root/'animals',['dog'],'animals')
            args={'workers':2,'min_class_images':0,'max_class_images':0,'ap10k':root/'ap-10k',
                  'subset_options':{'max_images':9,'negative_fraction':0,'class_target':1200,'lila_min_area':1/16}}
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
                    patch('PIL.Image.open',side_effect=AssertionError('Decoded image')), \
                    patch('hashlib.file_digest',side_effect=AssertionError('Hashed image')):
                report=merge(root/'lila',root/'animals',root/'out',**args)
            self.assertEqual(report['audit']['images_verified'],9)
            self.assertEqual(report['sampling']['source_images'],{'ap10k':9})
            self.assertEqual(report['audit']['group_overlap'],0)
            self.assertEqual(report['compact']['lila_area_filter']['images_excluded'],6)
            for image in (root/'out/images').rglob('*.jpg'):
                self.assertTrue(any(os.path.samefile(image,p) for p in (root/'ap-10k/data').glob('*.jpg')))
                label=root/'out/labels'/image.relative_to(root/'out/images').with_suffix('.txt')
                self.assertEqual(label.read_text().split()[1:],['0.3000000000','0.4500000000','0.4000000000','0.5000000000'])
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                again=merge(root/'lila',root/'animals',root/'out',**{**args,'workers':1})
            self.assertEqual(again['audit']['resumed_output_files'],9)
            text=(root/'out/SUBSET_REPORT.md').read_text(encoding='utf-8')
            self.assertIn('AP-10K',text)
            self.assertIn('任意一个正框达标',text)


if __name__=='__main__':unittest.main()
