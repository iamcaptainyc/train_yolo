from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from camping_subset import policy, target_name, remap, sample, deduplicate
import merge_yolo_datasets as merger
from test_merge_yolo_datasets import fixture


def record(i, name, source='animals', split='train', component=None):
    names = name if isinstance(name, list) else [name] if name else []
    rows = [(n, ('0.50000000', '5e-1', '0.20000000', '0.20000000')) for n in names]
    return {'dataset': source, 'split': split, 'relative': str(i), 'component': i if component is None else component,
            'rows': rows, 'source_rows': rows[:], 'target_names': names[:], 'refined_names': names[:],
            'row_origins': [[(source, '', n, 'source_yolo_only')] for n in names]}


class SubsetTests(unittest.TestCase):
    def test_68_classes_and_safe_boundaries(self):
        rules = policy()
        meta = merger.class_metadata()
        self.assertEqual(len(rules['classes']), 68)
        self.assertIsNone(target_name('hyena', {}, rules))
        self.assertIsNone(target_name('spotted_hyena', {'spotted_hyena': {'parent': 'hyena'}}, rules))
        for name in ('rabbit', 'mouse', 'cat', 'dog', 'wolf', 'sheep', 'goat', 'cattle', 'bear', 'polar_bear'):
            self.assertEqual(target_name(name, meta, rules), name)
        for name, expected in [('sun_bear','bear'),('asiatic_black_bear','bear'),('butterfly','insect'),('goldfish','fish'),('raven','crow')]:
            self.assertEqual(target_name(name, meta, rules), expected)
        for name in ('rodent', 'mustelid', 'tick', 'wild_cat', 'sloth_bear'):
            self.assertIsNone(target_name(name, meta, rules))
        meta['red_muntjac'] = {'parent': 'deer'}
        self.assertEqual(target_name('red_muntjac', meta, rules), 'deer')

    def test_user_merges_and_removed_classes_all_sources(self):
        rules, meta = policy(), merger.class_metadata()
        expected = {'cheetah':'leopard','jaguar':'leopard','snow_leopard':'leopard',
                    'clouded_leopard':'leopard','cougar':'lion','red_panda':'raccoon',
                    'brown_bear':'bear','black_bear':'bear','american_black_bear':'bear',
                    'pigeon':'bird'}
        for source in ('ap10k','animals','lila'):
            rows=[record(i,n,source) for i,n in enumerate(expected)]
            kept,rejected,_=remap(rows,meta)
            self.assertFalse(rejected)
            self.assertEqual([r['rows'][0][0] for r in kept],list(expected.values()))
            for r in kept:
                self.assertEqual(r['rows'][0][1],r['source_rows'][0][1])
        for name in ('snail','shark','seal','sea_lion','harbor_seal','bat','pangolin'):
            kept,rejected,_=remap([record(0,['rabbit',name])],meta)
            self.assertFalse(kept)
            self.assertEqual(len(rejected),1)
        self.assertEqual(target_name('polar_bear',meta,rules),'polar_bear')
        self.assertEqual(target_name('panda',meta,rules),'panda')

    def test_out_of_scope_box_excludes_whole_image_not_background(self):
        rows = [record(0, ['rabbit','tick']), record(1,'rabbit'), record(2,None)]
        kept, rejected, info = remap(rows, merger.class_metadata())
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(kept[0]['rows'][0][1], ('0.50000000','5e-1','0.20000000','0.20000000'))
        self.assertIn('WHOLE', rejected[0]['error'])

    def test_animals_first_for_each_class_then_lila_fill(self):
        rows = [record(i,'rabbit','lila') for i in range(20)] + [record(100+i,'rabbit') for i in range(3)]
        kept, rejected, info = sample(rows, 8, 0, 100, 42, .05, .05)
        self.assertEqual(len(kept), 8)
        self.assertEqual(Counter(r['dataset'] for r in kept), {'animals':3, 'lila':5})
        # With enough Animals candidates, no LILA rabbit image enters the subset.
        kept, _, _ = sample(rows, 2, 0, 100, 42, .05, .05)
        self.assertTrue(all(r['dataset']=='animals' for r in kept))

    def test_strict_total_negatives_multiclass_and_reproducibility(self):
        rows = []
        for split in ('train','val','test'):
            for i in range(100):
                rows.append(record(len(rows), ['rabbit','mouse'], split=split))
                rows.append(record(len(rows), None, 'lila', split))
        kept, _, stats = sample(rows, 100, .05, 1000, 17, .05, .05)
        self.assertEqual(len(kept), 100)
        self.assertEqual(stats['negative_images'], 5)
        again, _, _ = sample(rows, 100, .05, 1000, 17, .05, .05)
        self.assertEqual([r['relative'] for r in kept], [r['relative'] for r in again])
        self.assertEqual(stats['selected_class_images']['rabbit'],95)
        with self.assertRaises(ValueError):
            sample(rows,30001,.05,1200,42,.05,.05)

    def test_rare_class_is_not_folded_and_balanced_counts(self):
        rows = [record(i,'rabbit') for i in range(100)] + [record(100+i,'sheep') for i in range(3)]
        kept, _, _ = sample(rows, 10, 0, 1000, 42, .05, .05)
        self.assertEqual(Counter(n for r in kept for n,_ in r['rows']), {'rabbit':7,'sheep':3})

    def test_lila_group_round_robin_diversity(self):
        rows = [record(i,'rabbit','lila',component=0) for i in range(50)] + [record(100+i,'rabbit','lila',component=1) for i in range(50)]
        kept, _, _ = sample(rows, 10, 0, 1000, 42, .05, .05)
        self.assertEqual(Counter(r['component'] for r in kept), {0:5,1:5})

    def test_duplicate_identity_prefers_animals_and_retains_group_assignment(self):
        lila, animals = record(0, 'rabbit', 'lila'), record(1, 'rabbit')
        for r in (lila, animals):
            r.update(image_stamp={'device':1,'inode':99}, component=17, image=Path('same.jpg'))
        kept, rejected = deduplicate([lila, animals])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['dataset'], 'animals')
        self.assertEqual(kept[0]['component'], 17)
        self.assertEqual(len(rejected), 1)

    def test_alias_duplicate_box_keeps_all_original_targets_for_audit(self):
        r=record(0,['butterfly','insect'])
        kept, _, _=remap([r],merger.class_metadata())
        self.assertEqual(len(kept[0]['rows']),1)
        self.assertEqual(kept[0]['target_names'],['insect','insect'])
        self.assertEqual(len(kept[0]['source_rows']),2)

    def test_full_pipeline_hardlinks_remap_cap_grouping_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root/'lila', ['black_bear','rabbit','butterfly'], 'lila')
            fixture(root/'animals', ['brown bear','bull','moths and butterflies'], 'animals')
            args={'workers':2,'min_class_images':0,'max_class_images':0,
                  'subset_options':{'max_images':9,'negative_fraction':0,'class_target':1200}}
            with patch('PIL.Image.open',side_effect=AssertionError('decoded image')), patch('hashlib.file_digest',side_effect=AssertionError('hashed image')):
                report=merger.merge(root/'lila',root/'animals',root/'out',**args)
            self.assertLessEqual(report['audit']['images_verified'],9)
            self.assertEqual(report['audit']['group_overlap'],0)
            self.assertTrue({'bear','cattle','insect','rabbit'} <= set(report['classes']))
            self.assertNotIn('butterfly',report['classes'])
            self.assertNotIn('below_min_images_folded_to_parent', json.dumps(report['refinement']))
            self.assertTrue((root/'out/SUBSET_REPORT.md').is_file())
            self.assertNotIn('balanced whole-image sampling', (root/'out/excluded_images.jsonl').read_text())
            self.assertEqual(sum(x['images'] for x in report['stats'].values()),report['sampling']['total_selected'])
            for image in (root/'out/images').rglob('*.jpg'):
                source=next(p for p in root.rglob('*.jpg') if root/'out' not in p.parents and os.path.samefile(p,image))
                label=root/'out/labels'/image.relative_to(root/'out/images').with_suffix('.txt')
                self.assertTrue(all(line.split()[1:]==['0.50000000','5e-1','0.20000000','0.20000000'] for line in label.read_text().splitlines()))
            resumed=merger.merge(root/'lila',root/'animals',root/'out',**{**args,'workers':1})
            self.assertEqual(resumed['audit']['resumed_output_files'], report['audit']['images_verified'])
            with self.assertRaisesRegex(ValueError,'policy changed'):
                merger.merge(root/'lila',root/'animals',root/'out',**{**args,'subset_options':{**args['subset_options'],'max_images':8}})


if __name__=='__main__': unittest.main()
