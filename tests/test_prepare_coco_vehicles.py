import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from prepare_coco_vehicles import NAMES, export_views, load_label_schema
from download_coco import cooccurrence_counts
from merge_yolo_ids import merge
from vehicle_pose import angle_class


class Predictor:
    digest = 'local-test-predictor'

    def __init__(self, angles):
        self.angles = list(angles)
        self.seen = 0

    def predict(self, crops):
        self.seen += len(crops)
        angles, self.angles = self.angles[:len(crops)], self.angles[len(crops):]
        return angles


def make_cache(root):
    meta = root/'_metadata'
    meta.mkdir(parents=True)
    categories = [{'id': 1, 'name': 'person'}, {'id': 3, 'name': 'car'}, {'id': 6, 'name': 'bus'},
                  {'id': 8, 'name': 'truck'}, {'id': 18, 'name': 'dog'}]
    for split, ids in (('train', (1, 3)), ('val', (2,))):
        images = [{'id': i, 'file_name': f'{i:012d}.jpg', 'width': 100, 'height': 80} for i in ids]
        annotations = []
        for image in images:
            target = root/'images'/split/image['file_name']
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (100, 80), 'white').save(target)
            classes = (18, 1, 3, 3, 8, 6) if image['id'] == 1 else (3, 1) if image['id'] == 2 else (18, 8)
            for index, cid in enumerate(classes):
                annotations.append({'id': image['id']*10+index, 'image_id': image['id'],
                                    'category_id': cid, 'bbox': [10, 20, 30, 40]})
        (meta/f'instances_{split}2017_subset.json').write_text(
            json.dumps({'images': images, 'annotations': annotations, 'categories': categories}), encoding='utf-8')


class VehicleExportTests(unittest.TestCase):
    def test_fixed_46_ids_rewrite_existing_output_without_repredicting_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, output = root/'cache', root/'final'
            make_cache(cache)
            path = cache/'_metadata/instances_train2017_subset.json'
            data = json.loads(path.read_text())
            additions = [(100, 'cow', 1), (101, 'horse', 1), (102, 'zebra', 1), (103, 'giraffe', 3)]
            for cid, name, image_id in additions:
                data['categories'].append({'id': cid, 'name': name})
                data['annotations'].append({'id': 1000+cid, 'image_id': image_id,
                                            'category_id': cid, 'bbox': [10, 20, 30, 40]})
            path.write_text(json.dumps(data))
            export_views(cache, output, Predictor([0, 90, 180]))
            self.assertTrue((output/'images/train/000000000003.jpg').is_file())

            config_path = Path(__file__).resolve().parents[1]/'configs/coco_sentry46.yaml'
            schema = load_label_schema(config_path)
            cached = Predictor([])
            report = export_views(cache, output, cached, label_schema=schema)
            self.assertEqual(cached.seen, 0)
            config = yaml.safe_load((output/'data.yaml').read_text())
            self.assertEqual(config['nc'], 46)
            self.assertEqual(config['names'], schema['names'])
            label = output/'labels/train/000000000001.txt'
            rows = label.read_text().splitlines()
            self.assertEqual([int(row.split()[0]) for row in rows], [33, 0, 1, 3, 26, 25, 30, 36, 36])
            self.assertTrue(all(row.split(maxsplit=1)[1] == '0.25 0.5 0.3 0.5' for row in rows))
            self.assertTrue(os.path.samefile(cache/'images/train/000000000001.jpg',
                                            output/'images/train/000000000001.jpg'))
            self.assertFalse((output/'images/train/000000000003.jpg').exists())
            self.assertFalse((output/'labels/train/000000000003.txt').exists())
            self.assertTrue((cache/'images/train/000000000003.jpg').is_file())
            self.assertEqual(report['splits']['train']['excluded_images'], 1)
            self.assertEqual(report['splits']['train']['excluded_images_by_class'], {'giraffe': 1})
            self.assertEqual(report['splits']['train']['boxes_by_class']['horse'], 2)
            self.assertEqual(report['splits']['train']['images_by_class']['horse'], 1)
            self.assertIn('| zebra | horse | 36 |', (output/'REPORT.md').read_text(encoding='utf-8'))
            self.assertEqual(cooccurrence_counts([{'cattle', 'person', 'truck'}], ('truck',), {'cattle'})['all_three'], 1)

            existing = root/'existing'
            for split in ('train', 'val'):
                (existing/'images'/split).mkdir(parents=True)
                (existing/'labels'/split).mkdir(parents=True)
                Image.new('RGB', (20, 20), 'blue').save(existing/'images'/split/'own.jpg')
                (existing/'labels'/split/'own.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            (existing/'data.yaml').write_text(yaml.safe_dump(
                {'names': schema['names'], 'train': 'images/train', 'val': 'images/val'}))
            merged = merge([existing, output], root/'merged')
            self.assertTrue(merged['class_ids_preserved'])
            self.assertEqual(merged['names'], schema['names'])
            self.assertEqual(merged['splits']['train']['images'], 2)
            self.assertEqual(merged['splits']['val']['images'], 2)

    def test_label_config_rejects_ambiguous_or_incomplete_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.yaml'
            names = dict(enumerate(NAMES))
            names.pop(max(names))
            path.write_text(yaml.safe_dump({'names': names}))
            with self.assertRaisesRegex(ValueError, 'explicit exclusion'):
                load_label_schema(path)
            path.write_text('names:\n  0: person\n  0: dog\n')
            with self.assertRaises(yaml.YAMLError):
                load_label_schema(path)
            path.write_text(yaml.safe_dump({'names': NAMES, 'aliases': {'cow': 'unknown'}}))
            with self.assertRaisesRegex(ValueError, 'Aliases'):
                load_label_schema(path)

    def test_circular_angles_and_inclusive_boundaries(self):
        examples = {0: 'car_head', 45: 'car_head', -45: 'car_head', 45.0001: 'car_side',
                    90: 'car_side', -90: 'car_side', 134.9999: 'car_side', 135: 'car_tail',
                    -135: 'car_tail', 180: 'car_tail', 225: 'car_tail', 270: 'car_side', 315: 'car_head',
                    360: 'car_head', 720: 'car_head'}
        for angle, name in examples.items():
            self.assertEqual(angle_class(angle), name)
        for angle in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                angle_class(angle)

    def test_resume_keeps_people_animals_bus_truck_and_every_car_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, output = root/'cache', root/'final'
            make_cache(cache)
            first = Predictor([0, 90, 180])
            original_link = os.link

            def interrupt(source, destination):
                if Path(destination).name == '000000000003.jpg':
                    raise KeyboardInterrupt
                return original_link(source, destination)

            with patch('prepare_coco_vehicles.os.link', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                export_views(cache, output, first, batch=2)
            self.assertEqual(first.seen, 2)
            self.assertTrue((output/'INCOMPLETE.txt').is_file())
            second = Predictor([180])
            report = export_views(cache, output, second, batch=2)
            self.assertEqual(second.seen, 1)
            self.assertEqual(report['pose_counts_this_run'], {'cached_car_boxes': 2, 'predicted_car_boxes': 1})
            self.assertEqual([report['splits'][s]['images'] for s in ('train', 'val')], [2, 1])
            config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf-8'))
            self.assertEqual(config['nc'], 16)
            self.assertNotIn('car', config['names'].values())
            rows = (output/'labels/train/000000000001.txt').read_text().splitlines()
            self.assertEqual([config['names'][int(row.split()[0])] for row in rows],
                             ['dog', 'person', 'car_head', 'car_side', 'truck', 'bus'])
            self.assertTrue(all(row.split(maxsplit=1)[1] == '0.25 0.5 0.3 0.5' for row in rows))
            self.assertTrue(os.path.samefile(cache/'images/train/000000000001.jpg', output/'images/train/000000000001.jpg'))
            self.assertFalse((output/'INCOMPLETE.txt').exists())
            with (output/'car_angles.csv').open(newline='', encoding='utf-8') as stream:
                angles = list(csv.DictReader(stream))
            self.assertEqual([row['class'] for row in angles], ['car_head', 'car_side', 'car_tail'])
            stats = json.loads((output/'cooccurrence.json').read_text())
            self.assertEqual(stats['total']['all_three'], 1)
            self.assertEqual(stats['total']['person_vehicle'], 2)
            third = Predictor([])
            export_views(cache, output, third)
            self.assertEqual(third.seen, 0)
            # A changed image gets new predictions while the unchanged image reuses its angles.
            Image.new('RGB', (100, 80), 'black').save(cache/'images/train/000000000001.jpg')
            changed = Predictor([135, 45])
            export_views(cache, output, changed)
            self.assertEqual(changed.seen, 2)
            label = (output/'labels/train/000000000001.txt').read_text().splitlines()
            self.assertEqual([NAMES[int(row.split()[0])] for row in label][2:4], ['car_tail', 'car_head'])

    def test_bad_car_image_excluded_as_whole_and_model_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, output = root/'cache', root/'final'
            make_cache(cache)
            Image.new('RGB', (30, 20), 'white').save(cache/'images/train/000000000001.jpg')
            (cache/'report.json').write_text(json.dumps({'train': {'missing_images': 2}, 'val': {'missing_images': 0}}))
            report = export_views(cache, output, Predictor([180]))
            self.assertEqual(report['splits']['train']['images'], 1)
            self.assertEqual(len(report['issues']), 1)
            self.assertFalse((output/'labels/train/000000000001.txt').exists())
            self.assertNotIn('000000000001.jpg', (output/'train.txt').read_text())
            self.assertIn('2 张图片未下载', (output/'REPORT.md').read_text(encoding='utf-8'))
            incompatible = Predictor([])
            incompatible.digest = 'different-model'
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                export_views(cache, output, incompatible)


if __name__ == '__main__':
    unittest.main()
