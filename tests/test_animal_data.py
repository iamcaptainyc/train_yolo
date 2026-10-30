from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from animal_index import build,connect
from animal_sampling import compile_policy,export,plan


def fixture(root,classes,counts,name='animals',**extra):
    root.mkdir(parents=True)
    images,annotations=[],[]
    for cid,(label,total) in enumerate(zip(classes,counts)):
        for j in range(total):
            iid=len(images); filename=f'{iid}.jpg'
            (root/filename).write_bytes(f'{name} {iid}'.encode())
            images.append({'id':iid,'file_name':filename,'width':100,'height':100})
            annotations.append({'id':iid,'image_id':iid,'category_id':cid,'bbox':[10,20,40,50]})
    path=root/'annotations.json'
    path.write_text(json.dumps({'images':images,'annotations':annotations,
                               'categories':[{'id':i,'name':n} for i,n in enumerate(classes)]}))
    return {'name':name,'family':name,'kind':'coco','images':str(root),'annotations':[str(path)],**extra}


def catalog():
    return {'nodes':{'fox':{},'red_fox':{'parent':'fox'},'arctic_fox':{'parent':'fox'},'rabbit':{},'bear':{}},'source_names':{}}


class AnimalDataTests(unittest.TestCase):
    def test_lila_area_override_any_box_strict_boundary_and_negatives(self):
        from animal_data import area_ratio
        import argparse
        self.assertEqual(area_ratio('1/16'), .0625)
        for value in ('nan', '1/0', '-0.1', '1', '2'):
            with self.assertRaises(argparse.ArgumentTypeError): area_ratio(value)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            spec=fixture(root/'source',['rabbit'],[4],name='lila')
            spec['empty_images_are_negative']=True
            annotation=Path(spec['annotations'][0])
            data=json.loads(annotation.read_text())
            data['annotations'][0]['bbox']=[0,0,25,25]  # exactly 1/16: exclude
            data['annotations'][1]['bbox']=[0,0,10,10]  # small box retained with large one
            data['annotations'].append({'id':10,'image_id':1,'category_id':0,'bbox':[0,0,50,50]})
            data['annotations']=[a for a in data['annotations'] if a['image_id']!=3]
            annotation.write_text(json.dumps(data))
            index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'rabbit':{'members':['rabbit']}},'negative_fraction':.2,'lila_min_area':.0625}
            chosen,report,_=plan(index,catalog(),policy,10)
            self.assertEqual({r['relative'] for r in chosen},{'1','2','3'})
            self.assertEqual(len(next(r for r in chosen if r['relative']=='1')['rows']),2)
            self.assertEqual(report['lila_min_area'],.0625)
            (root/'catalog.yaml').write_text(yaml.safe_dump(catalog()))
            (root/'policy.yaml').write_text(yaml.safe_dump(policy))
            for n,flag in enumerate(('--lila-min-area','--lila-ratio','--lila_ratio')):
                output=root/f'preview{n}'
                result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'animal_data.py'),
                    'preview','--index',str(index),'--catalog',str(root/'catalog.yaml'),
                    '--policy',str(root/'policy.yaml'),'--output',str(output),flag,'0'],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                cli_report=json.loads((output/'report.json').read_text(encoding='utf-8'))
                self.assertEqual(cli_report['lila_min_area'],0)
                self.assertEqual(cli_report['selected_images'],4)
            policy['lila_min_area']=0
            chosen,_,_=plan(index,catalog(),policy,10)
            self.assertEqual(len(chosen),4)
            policy['lila_min_area']=1
            with self.assertRaises(ValueError): plan(index,catalog(),policy,10)

    def test_raw_index_keeps_unmapped_classes_update_and_atomic_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=fixture(root/'source',['red fox','rabbit','unknown'],[8,8,3])
            path=root/'index.sqlite'
            self.assertEqual(build(path,[spec])[0]['images'],19)
            self.assertEqual(build(path,[spec])[0]['status'],'unchanged')
            db=connect(path)
            self.assertEqual(db.execute('SELECT count(*) FROM categories').fetchone()[0],3);db.close()
            annotation=Path(spec['annotations'][0]); content=json.loads(annotation.read_text())
            content['annotations'][0]['image_id']=99999
            annotation.write_text(json.dumps(content))
            with self.assertRaises(ValueError): build(path,[spec])
            db=connect(path);self.assertEqual(db.execute('SELECT count(*) FROM images').fetchone()[0],19);db.close()

    def test_recursive_balance_not_flat_leaf_balance_and_reusable_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=fixture(root/'source',['red fox','arctic fox','rabbit'],[20,20,20])
            path=root/'index.sqlite';build(path,[spec])
            policy={'groups':{'small':{'members':['fox','rabbit']}},'negative_fraction':0,'seed':2}
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            chosen,report,_=plan(path,catalog(),policy,12)
            fine=Counter(concept for r in chosen for _,_,concept,_ in r['origins'])
            self.assertEqual(fine,{'rabbit':6,'red_fox':3,'arctic_fox':3})
            another={'groups':{'foxes':{'members':['fox']},'rabbits':{'members':['rabbit']}},'negative_fraction':0}
            selected,_,_=plan(path,catalog(),another,10)
            self.assertEqual(Counter(n for r in selected for n,_ in r['rows']),{'foxes':5,'rabbits':5})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_source_priority_inside_each_fine_class_and_rare_classes_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=fixture(root/'a',['red fox'],[20],name='ap10k')
            b=fixture(root/'b',['rabbit','red fox'],[2,20],name='lila')
            path=root/'index.sqlite';build(path,[a,b])
            policy={'groups':{'small':{'members':['fox','rabbit']}},'negative_fraction':0}
            chosen,_,_=plan(path,catalog(),policy,10)
            self.assertEqual(Counter(r['dataset'] for r in chosen),{'ap10k':8,'lila':2})

    def test_output_links_geometry_reports_and_group_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=fixture(root/'source',['red fox','rabbit'],[20,20])
            data=json.loads(Path(spec['annotations'][0]).read_text())
            for im in data['images']: im['camera_id']=im['id']//2
            Path(spec['annotations'][0]).write_text(json.dumps(data))
            index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'small':{'members':['fox','rabbit']}},'negative_fraction':0}
            report=export(index,catalog(),policy,20,root/'out',27)
            self.assertEqual(report['selected_images'],20)
            self.assertEqual(report['split']['group_overlap'],0)
            self.assertEqual(report['audited_boxes'],20)
            self.assertFalse((root/'out/INCOMPLETE.txt').exists())
            config=yaml.safe_load((root/'out/data.yaml').read_text())
            self.assertEqual(config['names'][27],'small')
            stamps={p.stat().st_ino:p for p in (root/'source').glob('*.jpg')}
            for image in (root/'out/images').rglob('*.jpg'):
                self.assertTrue(os.path.samefile(image,stamps[image.stat().st_ino]))
            for label in (root/'out/labels').rglob('*.txt'):
                self.assertEqual(label.read_text(),'27 0.300000000000 0.450000000000 0.400000000000 0.500000000000\n')
            self.assertTrue(report['provenance'])
            self.assertEqual(sum(report['split_image_counts'].values()),20)
            markdown=(root/'out/REPORT.md').read_text(encoding='utf-8')
            self.assertIn('类别数：**1**',markdown)
            self.assertIn('| **总计** | **20** |',markdown)
            self.assertNotIn('原始细类与来源',markdown)
            self.assertIn('原始细类与来源',(root/'out/REPORT_APPENDIX.md').read_text(encoding='utf-8'))
            # Refresh old exports using selection metadata, without reading image bytes.
            del report['split_image_counts'];del report['negative_images']
            (root/'out/report.json').write_text(json.dumps(report),encoding='utf-8')
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'animal_data.py'),
                'report','--dataset',str(root/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(markdown,(root/'out/REPORT.md').read_text(encoding='utf-8'))

    def test_catalog_cycles_overlaps_and_unknown_members_rejected(self):
        for cat,policy in [({'nodes':{'a':{'parent':'a'}}},{'groups':{'x':{'members':['a']}}}),
            (catalog(),{'groups':{'x':{'members':['fox']},'y':{'members':['red_fox']}}}),
            (catalog(),{'groups':{'x':{'members':['missing']}}})]:
            with self.assertRaises(ValueError):compile_policy(cat,policy)

    def test_unmapped_mixed_image_excluded_and_invalid_records_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit','unknown'],[10,3])
            p=Path(spec['annotations'][0]);data=json.loads(p.read_text())
            data['annotations'].append({'id':100,'image_id':0,'category_id':1,'bbox':[1,1,10,10]})
            data['annotations'][1]['bbox']=[-5,0,10,10]
            p.write_text(json.dumps(data));index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'small':{'members':['rabbit']}},'negative_fraction':0}
            chosen,report,_=plan(index,catalog(),policy,100)
            self.assertEqual(len(chosen),8)
            self.assertEqual(len(report['index_errors']),1)
            self.assertTrue(report['excluded'])

    def test_existing_source_changes_prevent_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit'],[10]);index=root/'index.sqlite';build(index,[spec])
            Path(spec['annotations'][0]).write_text(Path(spec['annotations'][0]).read_text()+' ')
            policy={'groups':{'small':{'members':['rabbit']}},'negative_fraction':0}
            with self.assertRaisesRegex(ValueError,'Source changed'):
                export(index,catalog(),policy,8,root/'out')
            self.assertFalse((root/'out').exists())

    def test_actual_cli_manifest_index_preview_and_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit'],[12])
            from PIL import Image
            for image in (root/'s').glob('*.jpg'):
                Image.new('RGB',(100,100),(30,70,90)).save(image)
            (root/'sources.yaml').write_text(yaml.safe_dump({'sources':[spec]}))
            (root/'catalog.json').write_text(json.dumps(catalog()))
            (root/'policy.yaml').write_text(yaml.safe_dump({'groups':{'small':{'members':['rabbit']}},'negative_fraction':0}))
            script=Path(__file__).resolve().parents[1]/'animal_data.py'
            def run(args):
                result=subprocess.run([sys.executable,'-X','utf8',str(script),*map(str,args)],capture_output=True,encoding='utf-8')
                self.assertEqual(result.returncode,0,result.stderr)
            run(['index','--index',root/'index.sqlite','--sources',root/'sources.yaml'])
            for command in ('preview','sample'):
                run([command,'--index',root/'index.sqlite','--catalog',root/'catalog.json','--policy',root/'policy.yaml',
                     '--max-images','10','--output',root/command])
            self.assertFalse((root/'preview/images').exists())
            self.assertTrue((root/'sample/data.yaml').exists())
            report=json.loads((root/'sample/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['image_repair']['counts'],{'healthy':10})

    def test_inline_repair_hardlinks_deletion_and_final_counts(self):
        from PIL import Image
        from unittest.mock import patch
        from repair_images import inspect_image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit'],[5])
            for image in (root/'s').glob('*.jpg'):
                Image.new('RGB',(100,100),(30,70,90)).save(image)
            damaged=root/'s/0.jpg'
            damaged.write_bytes(damaged.read_bytes()[:-2])
            alias=root/'alias.jpg';os.link(damaged,alias)
            (root/'s/1.jpg').write_bytes(b'not an image')
            index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'rabbit':{'members':['rabbit']}},'negative_fraction':0}
            with patch('repair_images.inspect_image',wraps=inspect_image) as check:
                report=export(index,catalog(),policy,5,root/'out',repair_images=True)
                self.assertEqual(check.call_count,5)
            self.assertEqual(report['selected_images'],4)
            self.assertEqual(report['image_repair']['counts'],{'healthy':3,'repaired':1,'deleted':1})
            self.assertEqual(sum(report['split_image_counts'].values()),4)
            self.assertEqual(sum(report['split']['class_image_counts']['rabbit'].values()),4)
            self.assertEqual(report['provenance'][0]['selected_images'],4)
            self.assertEqual(len(list((root/'out/labels').rglob('*.txt'))),4)
            self.assertEqual(len((root/'out/selection.jsonl').read_text().splitlines()),4)
            self.assertTrue(os.path.samefile(alias,damaged))
            self.assertTrue(alias.read_bytes().endswith(b'\xff\xd9'))
            self.assertTrue((root/'s/1.jpg').exists())
            # A second extraction must accept the stamps changed by our own repair.
            export(index,catalog(),policy,5,root/'again',repair_images=True)
            # Standalone repair also refreshes the new compact animal report.
            from repair_dataset_report import refresh_dataset_reports
            victim=next((root/'out/images').rglob('*.jpg'))
            victim.write_bytes(b'broken')
            operations=root/'standalone/operations'
            self.assertEqual(inspect_image(victim,operations=operations)['status'],'deleted')
            refresh_dataset_reports([root/'out'],operations)
            refreshed=json.loads((root/'out/report.json').read_text(encoding='utf-8'))
            self.assertEqual(refreshed['selected_images'],3)
            self.assertEqual(sum(refreshed['split_image_counts'].values()),3)
            self.assertEqual(refreshed['provenance'][0]['selected_images'],3)
            self.assertEqual(len((root/'out/selection.jsonl').read_text().splitlines()),3)

    def test_shipped_19_policy_compiles_and_preserves_required_exclusions(self):
        root=Path(__file__).resolve().parents[1]
        cat=json.loads((root/'configs/animal_catalog.json').read_text(encoding='utf-8'))
        policy=yaml.safe_load((root/'configs/camping19.yaml').read_text())
        result=compile_policy(cat,policy)
        self.assertEqual(len(policy['groups']),19)
        for excluded in ('camel','pig','squirrel','turtle','panda'):
            self.assertNotIn(excluded,result)
        self.assertEqual(result['eagle'][0],'eagle')
        self.assertEqual(result['owl'][0],'bird')
        self.assertEqual(result['fox'][0],'small_mammal')
        for name in ('mongoose','crab_eating_mongoose','weasel','yellow_bellied_weasel','yellow_throated_marten',
                     'civet','large_indian_civet','spotted_linsang','beaver','mole','mink','opossum','tree_shrew'):
            self.assertEqual(result[name][0],'small_mammal',name)

    def test_size_tag_balances_member_families_despite_different_leaf_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            classes=['mongoose_a','mongoose_b','mongoose_c','weasel_a','civet_a']
            cat={'nodes':{n:{'tags':['small']} for n in ('mongoose','weasel','civet')}}
            for name in classes:cat['nodes'][name]={'parent':name.split('_')[0]}
            spec=fixture(root/'s',classes,[20]*5);index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'small':{'member_tag':'small'}},'negative_fraction':0}
            chosen,_,_=plan(index,cat,policy,12)
            counts=Counter(concept.split('_')[0] for r in chosen for _,_,concept,_ in r['origins'])
            self.assertEqual(counts,{'mongoose':4,'weasel':4,'civet':4})

    def test_refresh_preserves_seeded_selection_across_source_id_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=fixture(root/'a',['red fox','rabbit'],[20,20],name='ap10k')
            b=fixture(root/'b',['red fox','rabbit'],[20,20],name='animals')
            index=root/'index.sqlite';build(index,[a,b])
            policy={'groups':{'small':{'members':['fox','rabbit']}},'negative_fraction':0}
            before,_,_=plan(index,catalog(),policy,10)
            build(index,[a],refresh=True)
            after,_,_=plan(index,catalog(),policy,10)
            self.assertEqual({str(r['image']) for r in before},{str(r['image']) for r in after})

    def test_raw_geometry_evidence_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit'],[10]);index=root/'index.sqlite';build(index,[spec])
            db=connect(index)
            db.execute('UPDATE boxes SET coords=?',(json.dumps(['.4','.4','.2','.2']),));db.commit();db.close()
            with self.assertRaisesRegex(ValueError,'evidence'):
                export(index,catalog(),{'groups':{'small':{'members':['rabbit']}},'negative_fraction':0},10,root/'out')

    def test_multi_class_image_counts_once_and_keeps_every_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=fixture(root/'s',['rabbit','bear'],[3,3]);p=Path(spec['annotations'][0])
            data=json.loads(p.read_text())
            for im in data['images']:
                data['annotations'].append({'id':100+im['id'],'image_id':im['id'],'category_id':1 if im['id']<3 else 0,'bbox':[40,40,10,10]})
            p.write_text(json.dumps(data));index=root/'index.sqlite';build(index,[spec])
            policy={'groups':{'small':{'members':['rabbit']},'large':{'members':['bear']}},'negative_fraction':0}
            chosen,report,_=plan(index,catalog(),policy,1)
            self.assertEqual(len(chosen),1)
            self.assertEqual(report['selected_images'],1)
            self.assertEqual(len(chosen[0]['rows']),2)
            self.assertEqual({n for n,_ in chosen[0]['rows']},{'small','large'})

    def test_yolo_adapter_preserves_original_coordinate_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'yolo';source.mkdir()
            for split in ('train','val','test'):
                (source/'images'/split).mkdir(parents=True);(source/'labels'/split).mkdir(parents=True)
                (source/'images'/split/'a.jpg').write_bytes(split.encode())
                (source/'labels'/split/'a.txt').write_text('7 0.50000 0.5 0.2 0.3\n')
            (source/'data.yaml').write_text(yaml.safe_dump({'names':{7:'rabbit'},'train':'images/train','val':'images/val','test':'images/test'}))
            spec={'name':'yolo','kind':'yolo','yaml':str(source/'data.yaml')};index=root/'index.sqlite';build(index,[spec])
            export(index,catalog(),{'groups':{'small':{'members':['rabbit']}},'negative_fraction':0},3,root/'out')
            for label in (root/'out/labels').rglob('*.txt'):
                self.assertEqual(label.read_text(),'0 0.50000 0.5 0.2 0.3\n')
            self.assertEqual(len((root/'out/selection.jsonl').read_text().splitlines()),3)


if __name__=='__main__':unittest.main()
