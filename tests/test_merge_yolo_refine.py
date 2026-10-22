import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import merge_yolo_datasets as merger
from merge_yolo_refine import image_counts, soft_cap, refine_large, RULE_FILE
from merge_yolo_split import fold_rare
from test_merge_yolo_datasets import fixture


class RefineTests(unittest.TestCase):
    def make_lila(self, root, categories, assignments):
        """Each assignment is a list of original category IDs in the same image."""
        slug = 'ena24'
        images, annotations = [], []
        for i, cats in enumerate(assignments):
            split = ('train', 'val', 'test')[i % 3]
            filename = f'frame_{i}.jpg'
            stem = hashlib.sha256((slug + '/' + filename).encode()).hexdigest()
            ip = root / 'lila/images' / split / slug / (stem + '.jpg')
            lp = root / 'lila/labels' / split / slug / (stem + '.txt')
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            ip.write_bytes(f'unique-lila-{i}'.encode())
            lines = []
            for k, cid in enumerate(cats):
                # Nonoverlapping boxes; source order can differ from class order.
                x = 10 + k * 30
                annotations.append({'image_id': i, 'category_id': cid, 'bbox': [x, 20, 20, 20]})
                lines.append(f'0 {(x+10)/100:.8f} 0.30000000 0.20000000 0.20000000')
            lp.write_text('\n'.join(lines) + '\n')
            images.append({'id': i, 'file_name': filename, 'width': 100, 'height': 100, 'location': f'camera-{i}'})
        (root / 'lila/data.yaml').write_text(yaml.safe_dump({'path': str(root / 'lila'),
             'names': {0: 'squirrel'}, **{s: 'images/' + s for s in merger.SPLITS}}))
        ann = root / 'source/ena24/selection/selected_annotations.json'
        ann.parent.mkdir(parents=True)
        ann.write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}))
        fixture(root / 'animals', ['snake'], 'animals')
        return ann

    def run_merge(self, root, **kw):
        return merger.merge(root / 'lila', root / 'animals', root / 'out', lila_source=root / 'source',
                            min_class_images=kw.pop('minimum', 2), max_class_images=kw.pop('maximum', 3),
                            cap_class_images=kw.pop('cap', 0), **kw)

    def test_original_box_matching_image_threshold_and_one_to_many_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cats = [{'id': 1, 'name': 'Eastern Gray Squirrel'}, {'id': 2, 'name': 'Eastern Fox Squirrel'}]
            self.make_lila(root, cats, [[1, 2], [1], [1], [2], [2], [2]])
            with patch('PIL.Image.open', side_effect=AssertionError('decoder')), patch('hashlib.file_digest', side_effect=AssertionError('image hash')):
                result = self.run_merge(root)
            self.assertIn('eastern_gray_squirrel', result['classes'])
            self.assertIn('eastern_fox_squirrel', result['classes'])
            self.assertNotIn('squirrel', result['classes'])
            mapping = result['source_class_mapping']['lila'][0]
            self.assertEqual(set(mapping['targets']), {'eastern_gray_squirrel', 'eastern_fox_squirrel'})
            self.assertEqual(result['refinement']['before_class_images']['squirrel'], 6)
            self.assertEqual(result['refinement']['after_class_images']['eastern_fox_squirrel'], 4)
            self.assertEqual(result['audit']['boxes_verified'], 13)
            repeated = self.run_merge(root, workers=2)
            self.assertEqual(repeated['audit']['resumed_output_files'], 12)

    def test_default_does_not_sample_species_above_refinement_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}], [[1]] * 12)
            # Intentionally omit cap_class_images to exercise the public default.
            result = merger.merge(root / 'lila', root / 'animals', root / 'out',
                                  lila_source=root / 'source', min_class_images=1, max_class_images=3)
            self.assertEqual(result['sampling']['maximum_images'], 0)
            self.assertEqual(result['sampling']['removed_images'], 0)
            self.assertEqual(result['excluded_images'], 0)
            self.assertEqual(result['audit']['images_verified'], 18)
            self.assertEqual(result['refinement']['still_above_threshold']['eastern_gray_squirrel'], 12)
            self.assertIn('未启用数量抽样', (root / 'out/MERGE_REPORT.md').read_text(encoding='utf-8'))

    def test_small_child_folds_back_but_large_child_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}, {'id': 2, 'name': 'Eastern Fox Squirrel'}],
                           [[1], [1], [1], [1], [1], [2]])
            result = self.run_merge(root)
            self.assertIn('eastern_gray_squirrel', result['classes'])
            self.assertIn('squirrel', result['classes'])
            self.assertNotIn('eastern_fox_squirrel', result['classes'])
            self.assertIn({'source': 'eastern_fox_squirrel', 'target': 'squirrel', 'original_images': 1}, result['rare_class_policy']['changes'])

    def test_missing_dimensions_and_unmatched_boxes_keep_coarse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}], [[1]] * 6)
            data = json.loads(path.read_text())
            data['images'][0].pop('width')
            data['annotations'][1]['bbox'] = [11, 20, 20, 20]
            path.write_text(json.dumps(data))
            result = self.run_merge(root, minimum=0)
            self.assertEqual(result['refinement']['after_class_images']['squirrel'], 2)
            self.assertEqual(result['refinement']['after_class_images']['eastern_gray_squirrel'], 4)
            self.assertGreater(result['refinement']['issues']['ambiguous_or_unmatched_box_kept_coarse'], 0)

    def test_coincident_original_species_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}, {'id': 2, 'name': 'Eastern Fox Squirrel'}], [[1]] * 6)
            data = json.loads(path.read_text())
            data['annotations'].append({**data['annotations'][0], 'category_id': 2})
            path.write_text(json.dumps(data))
            result = self.run_merge(root, minimum=0)
            self.assertEqual(result['refinement']['after_class_images']['squirrel'], 1)

    def test_no_refinement_at_exact_image_threshold_despite_many_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}], [[1, 1, 1]] * 6)
            result = self.run_merge(root, maximum=6, minimum=0)
            self.assertNotIn('eastern_gray_squirrel', result['classes'])
            self.assertEqual(result['refinement']['before_class_images']['squirrel'], 6)

    def test_cap_counts_images_keeps_rare_boxes_and_split_membership(self):
        coords = ('0.5', '0.5', '0.2', '0.2')
        records = [{'rows': [('bird', coords)] * 4 + ([('snake', coords)] if i < 2 else []),
                    'split': ('train', 'val', 'test')[i % 3], 'relative': str(i), 'component': i % 6}
                   for i in range(30)]
        kept, excluded, report = soft_cap(records, 9, 2, 42)
        self.assertEqual(image_counts(kept)['bird'], 9)
        self.assertEqual(image_counts(kept)['snake'], 2)
        self.assertEqual(len(excluded), 21)
        self.assertEqual({r['split'] for r in kept}, {'train', 'val', 'test'})
        self.assertTrue(all(len(r['rows']) in (4, 5) for r in kept))
        again, _, _ = soft_cap(records, 9, 2, 42)
        self.assertEqual([r['relative'] for r in kept], [r['relative'] for r in again])

    def test_soft_cap_reports_unavoidable_excess_without_erasing_rare_labels(self):
        coords = ('0.5', '0.5', '0.2', '0.2')
        records = [{'rows': [('bird', coords), ('rare_' + str(i), coords)],
                    'split': 'train', 'relative': str(i)} for i in range(5)]
        kept, excluded, report = soft_cap(records, 2, 1, 42)
        self.assertEqual(len(kept), 5)
        self.assertEqual(excluded, [])
        self.assertEqual(report['remaining_over_limit']['bird'], 5)

    def test_capped_merge_resume_and_final_report_use_retained_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_lila(root, [{'id': 1, 'name': 'Eastern Gray Squirrel'}], [[1]] * 12)
            result = self.run_merge(root, minimum=1, cap=5)
            self.assertLessEqual(result['refinement']['after_class_images']['eastern_gray_squirrel'], 5)
            self.assertEqual(sum(s['images'] for s in result['stats'].values()) + result['excluded_images'], 18)
            self.assertEqual(result['resplit']['missing_classes'], result['missing_classes'])
            again = self.run_merge(root, minimum=1, cap=5)
            self.assertEqual(again['audit']['resumed_output_files'], result['audit']['images_verified'])

    def test_legacy_marten_bird_mapping_corrected_only_with_matched_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slug, filename = 'swg-camera-traps', 'x.jpg'
            stem = hashlib.sha256((slug + '/' + filename).encode()).hexdigest()
            path = root / slug / 'selection/selected_annotations.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'images': [{'id': 1, 'file_name': filename, 'width': 100, 'height': 100}],
                                        'categories': [{'id': 32, 'name': 'yellow_throated_marten'}],
                                        'annotations': [{'image_id': 1, 'category_id': 32, 'bbox': [10, 20, 20, 20]}]}))
            coords = ('0.20000000', '0.30000000', '0.20000000', '0.20000000')
            records = [{'dataset': 'lila', 'relative': slug + '/' + stem + '.jpg',
                        'rows': [('bird', coords)], 'duplicate_boxes_removed': 0},
                       {'dataset': 'animals', 'relative': 'x.jpg', 'rows': [('bird', coords)], 'duplicate_boxes_removed': 0}]
            datasets = [{'alias': 'lila', 'yaml': root / 'input/data.yaml', 'mapping': {0: 'bird'}}]
            metadata = merger.class_metadata()
            refine_large(records, datasets, metadata, 1, root)
            self.assertEqual(records[0]['rows'][0][0], 'yellow_throated_marten')
            self.assertEqual(records[1]['rows'][0][0], 'bird')
            fold_rare(records, datasets, metadata, 100)
            self.assertEqual(records[0]['rows'][0][0], 'mustelid')

    def test_all_refinement_rules_have_english_names_and_known_parents(self):
        metadata = merger.class_metadata()
        for rule in json.loads(RULE_FILE.read_text(encoding='utf-8'))['rules']:
            self.assertEqual(merger.normalized(rule['target']), rule['target'])
            self.assertIn(rule['parent'], metadata)
            self.assertNotIn(rule['parent'], {'wild_boar', 'pig', 'wolf'})


if __name__ == '__main__':
    unittest.main()
