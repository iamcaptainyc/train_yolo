import csv
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
from concurrent.futures import CancelledError
from threading import Barrier, Event, Lock
import unittest
from unittest.mock import Mock, patch

from download_lila import (build_plan, cloud_url, command_for, load_catalog, read_catalog,
                           dataset_inputs, execute, parse_dataset_page, relative_prefix,
                           resolve_dataset, run_datasets, run_plan, select_dataset)


def entry(slug='wcs-camera-traps', name='WCS Camera Traps'):
    row = {'name': name, 'short_name': slug}
    roots = {'aws': 'http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife',
             'azure': 'https://lilawildlife.blob.core.windows.net/lila-wildlife',
             'gcp': 'https://storage.googleapis.com/public-datasets-lila'}
    for provider, root in roots.items():
        row['image_base_url_'+provider] = root+'/wcs-unzipped'
        row['metadata_url_'+provider] = root+'/wcs/metadata.json.zip'
        row['bbox_url_'+provider] = root+'/wcs/bboxes.json.zip'
    return row


class CatalogTests(unittest.TestCase):
    def test_csv_and_dataset_matching(self):
        row = entry()
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        rows = read_catalog('\ufeff'+stream.getvalue())
        self.assertEqual(select_dataset(rows, 'WCS'), row)
        self.assertEqual(select_dataset(rows, 'wcs-camera-traps'), row)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            select_dataset([row, entry('another-wcs', 'Another WCS')], 'wcs')

    def test_invalid_or_duplicate_slugs_rejected(self):
        for text in ['name,short_name\nBad,../outside\n',
                     'name,short_name\nA,duplicate\nB,DUPLICATE\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_catalog(text)

    def test_cached_and_custom_catalog_need_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp)/'catalog.csv'
            cached.write_text('name,short_name\nWCS,wcs\n')
            with patch('download_lila.urlopen') as network:
                self.assertEqual(load_catalog(None, cached)[0]['short_name'], 'wcs')
                self.assertEqual(load_catalog(cached, Path(tmp)/'missing', refresh=True)[0]['name'], 'WCS')
            network.assert_not_called()


class PageUrlTests(unittest.TestCase):
    def test_ena24_page_slug_need_not_match_catalog_name(self):
        row = entry('ena24', 'ENA24')
        for key in row:
            if '_url_' in key:
                row[key] = row[key].replace('wcs-unzipped', 'ena24/images').replace('/wcs/', '/ena24/')
        row['bbox_url_aws'] = row['bbox_url_gcp'] = row['bbox_url_azure'] = ''
        html = '''<h1>ENA24-detection</h1>
            <p>s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/ena24/images (AWS)</p>
            <a href="https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json">Metadata</a>
            <a href="https://storage.googleapis.com/public-datasets-lila/ena24/non_human.json">Metadata (non-human only)</a>'''
        result = parse_dataset_page([row], 'https://lila.science/datasets/ena24detection', html)
        self.assertEqual(result['short_name'], 'ena24')
        self.assertTrue(result['metadata_url_aws'].endswith('/ena24/ena24.json'))
        self.assertEqual(result['bbox_url_aws'], '')

    def test_page_updates_old_catalog_bboxes_and_prefers_species_classes(self):
        html = '''<h1>WCS Camera Traps</h1>
            <p>gs://public-datasets-lila/wcs-unzipped (GCP)</p>
            <a href="https://storage.googleapis.com/public-datasets-lila/wcs/new_no_classes.zip">Bounding box annotations</a>
            <a href="https://storage.googleapis.com/public-datasets-lila/wcs/new_with_classes.zip">Bounding box annotations</a>
            <a href="https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/animals/0011/0009.jpg">Example image</a>'''
        result = parse_dataset_page([entry()], 'https://lila.science/datasets/wcscameratraps', html)
        self.assertTrue(result['bbox_url_aws'].endswith('/wcs/new_with_classes.zip'))
        self.assertTrue(result['image_base_url_aws'].endswith('/wcs-unzipped'))

    def test_split_json_does_not_replace_annotation_metadata(self):
        html = '''<h1>WCS</h1><p>gs://public-datasets-lila/wcs-unzipped</p>
            <a href="https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip">wcs_camera_traps.json.zip</a>
            <a href="https://storage.googleapis.com/public-datasets-lila/wcs/wcs_splits.json">wcs_splits.json</a>'''
        result = parse_dataset_page([entry()], 'https://lila.science/datasets/wcscameratraps', html)
        self.assertTrue(result['metadata_url_aws'].endswith('/wcs/wcs_camera_traps.json.zip'))

    def test_unlisted_dataset_resolved_from_actual_page_sources(self):
        html = '''<h1>New Camera Dataset</h1><p>gs://public-datasets-lila/new-camera/images</p>
            <a href="https://storage.googleapis.com/public-datasets-lila/new-camera/metadata.json.zip">Metadata</a>'''
        result = parse_dataset_page([], 'https://lila.science/datasets/new-camera', html)
        self.assertEqual(result['short_name'], 'new-camera')
        self.assertEqual(result['image_base_url_aws'],
                         's3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/new-camera/images')

    def test_resolves_url_ignoring_page_query_and_fragment(self):
        response = io.BytesIO(b'<h1>WCS</h1><p>gs://public-datasets-lila/wcs-unzipped</p>')
        with patch('download_lila.urlopen', return_value=response) as fetch:
            result = resolve_dataset([entry()], 'https://lila.science/datasets/wcscameratraps/?x=1#download')
        self.assertEqual(result['short_name'], 'wcs-camera-traps')
        self.assertEqual(fetch.call_args.args[0], 'https://lila.science/datasets/wcscameratraps/')

    def test_unrelated_page_and_non_lila_urls_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no supported public image folder'):
            parse_dataset_page([], 'https://lila.science/datasets/unknown', '<h1>Not found</h1>')
        with patch('download_lila.urlopen') as fetch, self.assertRaises(ValueError):
            resolve_dataset([], 'https://example.com/datasets/ena24detection')
        fetch.assert_not_called()


class PlanTests(unittest.TestCase):
    def test_cloud_url_conversion(self):
        row = entry()
        self.assertEqual(cloud_url(row['image_base_url_aws'], 'aws'),
                         's3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped')
        self.assertEqual(cloud_url(row['image_base_url_gcp'], 'gcp'),
                         'gs://public-datasets-lila/wcs-unzipped')
        self.assertEqual(cloud_url(row['image_base_url_azure'], 'azure'), row['image_base_url_azure'])
        self.assertEqual(cloud_url(row['metadata_url_azure'], 'gcp'),
                         'gs://public-datasets-lila/wcs/metadata.json.zip')
        with self.assertRaises(ValueError):
            cloud_url('https://example.com/dataset', 'aws')

    def test_prefix_preserves_metadata_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan([entry()], 'aws', Path(tmp), prefix='animals/0011')
            image = plan[0]
            self.assertTrue(image['source'].endswith('/wcs-unzipped/animals/0011'))
            self.assertEqual(Path(image['target']), Path(tmp)/'wcs-camera-traps/images/animals/0011')
            self.assertEqual(Path(plan[1]['target']).name, 'metadata.json.zip')

    def test_multiple_datasets_and_missing_bboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = entry(), entry('other', 'Other')
            second['bbox_url_aws'] = ''
            plan = build_plan([first, second], 'aws', Path(tmp))
            self.assertEqual(len(plan), 5)
            self.assertNotEqual(plan[0]['target'], plan[3]['target'])
            with self.assertRaisesRegex(ValueError, 'no bbox'):
                build_plan([second], 'aws', Path(tmp), kind='bbox')

    def test_bbox_override_uses_exact_selected_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan([entry()], 'aws', Path(tmp), kind='bbox',
                              bbox_url='s3://bucket/new-bboxes.zip')
            self.assertEqual(plan[0]['source'], 's3://bucket/new-bboxes.zip')
            self.assertEqual(Path(plan[0]['target']).name, 'new-bboxes.zip')

    def test_unsafe_prefixes_rejected(self):
        for prefix in ['../outside', '/absolute', 'a/../../b', 'C:/data', 'a/*', '%2e%2e']:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                relative_prefix(prefix)

    def test_three_native_tool_commands_do_not_delete_local_files(self):
        item = {'source': 'source with spaces', 'target': '/destination with spaces', 'kind': 'images'}
        aws = command_for(item, 'aws', 32)
        azure = command_for(item, 'azure', 32)
        gcp = command_for(item, 'gcp', 32)
        self.assertEqual(aws[:3], ['aws', 's3', 'sync'])
        self.assertIn('--no-sign-request', aws)
        self.assertIn('source with spaces/*', azure)
        self.assertIn('--overwrite=ifSourceNewer', azure)
        self.assertIn('GSUtil:parallel_thread_count=32', gcp)
        for command in (aws, azure, gcp):
            self.assertNotIn('--delete', command)
            self.assertNotIn('-d', command)


class BatchTests(unittest.TestCase):
    def test_text_and_json_url_lists_merge_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = Path(tmp)/'urls.txt'
            text.write_text('\ufeff# notes\n\n https://lila.science/datasets/ena24detection \n'
                            'https://lila.science/datasets/wcscameratraps#download\n', encoding='utf-8')
            first = 'https://lila.science/datasets/ena24detection'
            self.assertEqual(dataset_inputs([first], text),
                             [first, 'https://lila.science/datasets/wcscameratraps#download'])
            data = Path(tmp)/'urls.json'
            data.write_text(json.dumps([first, 'wcs', first]))
            self.assertEqual(dataset_inputs([], data), [first, 'wcs'])
            data.write_text(json.dumps({'urls': [first]}))
            with self.assertRaises(ValueError):
                dataset_inputs([], data)

    def test_serial_mode_keeps_dataset_order_and_isolates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan([entry('first'), entry('second')], 'aws', Path(tmp), kind='images')
            calls = []

            def run(items, provider, **kwargs):
                name = items[0]['dataset']
                calls.append(name)
                if name == 'first':
                    raise OSError('failed transfer')
                return [{**item, 'status': 'ok'} for item in items]

            with patch('download_lila.run_plan', side_effect=run):
                results = run_datasets(plan, 'aws', mode='serial')
            self.assertEqual(calls, ['first', 'second'])
            self.assertEqual([r['status'] for r in results], ['failed', 'ok'])

    def test_parallel_datasets_overlap_but_do_not_exceed_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan([entry(name) for name in ['first', 'second', 'third']],
                              'aws', Path(tmp), kind='images')
            barrier, lock = Barrier(2), Lock()
            active, peak, started = 0, 0, 0

            def run(items, provider, **kwargs):
                nonlocal active, peak, started
                with lock:
                    active += 1
                    peak = max(peak, active)
                    started += 1
                    ordinal = started
                try:
                    if ordinal <= 2:
                        barrier.wait(timeout=5)
                    return [{**item, 'status': 'ok'} for item in items]
                finally:
                    with lock:
                        active -= 1

            with patch('download_lila.run_plan', side_effect=run):
                results = run_datasets(plan, 'aws', mode='parallel', dataset_workers=2, workers=17)
            self.assertEqual(peak, 2)
            self.assertEqual([r['dataset'] for r in results], ['first', 'second', 'third'])

    def test_cancel_stops_active_native_process(self):
        stop = Event()
        process = Mock()

        def wait(timeout=None):
            if timeout == 0.5:
                stop.set()
                raise subprocess.TimeoutExpired('aws', timeout)
            return 0

        process.wait.side_effect = wait
        with patch('download_lila.subprocess.Popen', return_value=process), self.assertRaises(CancelledError):
            execute(['aws', 's3', 'sync'], {}, stop_event=stop)
        process.terminate.assert_called_once_with()


class ExecutionTests(unittest.TestCase):
    def test_completed_annotations_verified_and_reused_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan([entry()], 'aws', root, kind='metadata')
            state_dir = root/'.transfer-state'

            def transfer(command, env):
                Path(command[4]).write_bytes(b'complete annotation')
                return 0

            with patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', side_effect=transfer) as execute:
                first = run_plan(plan, 'aws', state_dir=state_dir)
                second = run_plan(plan, 'aws', state_dir=state_dir)
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(first[0]['status'], 'ok')
                self.assertEqual(second[0]['status'], 'skipped')
                # Existence alone is insufficient: detect a same-sized corrupt annotation.
                target = Path(plan[0]['target'])
                target.write_bytes(b'x'*target.stat().st_size)
                third = run_plan(plan, 'aws', state_dir=state_dir)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(third[0]['status'], 'ok')

    def test_interrupted_job_persists_and_image_jobs_always_resync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan([entry()], 'aws', root, kind='images')
            state_dir = root/'.transfer-state'
            with patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', side_effect=KeyboardInterrupt), \
                 self.assertRaises(KeyboardInterrupt):
                run_plan(plan, 'aws', state_dir=state_dir)
            checkpoint = next(state_dir.glob('*.json'))
            self.assertEqual(json.loads(checkpoint.read_text())['status'], 'interrupted')
            with patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', return_value=0) as execute:
                run_plan(plan, 'aws', state_dir=state_dir)
                run_plan(plan, 'aws', state_dir=state_dir)
                self.assertEqual(execute.call_count, 2)
            self.assertEqual(json.loads(checkpoint.read_text())['status'], 'ok')

    def test_retry_delays_are_capped_and_state_reports_final_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan([entry()], 'aws', root, kind='images')
            state_dir = root/'.transfer-state'
            with patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', return_value=1), patch('download_lila.time.sleep') as sleep:
                result = run_plan(plan, 'aws', retries=3, state_dir=state_dir,
                                  retry_delay=2, max_retry_delay=3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 3, 3])
            self.assertEqual(result[0]['status'], 'failed')
            saved = json.loads(next(state_dir.glob('*.json')).read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['attempt'], 4)

    def test_recovery_mode_can_recheck_completed_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan([entry()], 'aws', root, kind='metadata')

            def transfer(command, env):
                Path(command[4]).write_bytes(b'annotation')
                return 0

            with patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', side_effect=transfer) as execute:
                run_plan(plan, 'aws', state_dir=root/'state')
                run_plan(plan, 'aws', state_dir=root/'state', resume=False)
                self.assertEqual(execute.call_count, 2)

    def test_dry_run_needs_no_cloud_tool_and_creates_no_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'not-created'
            plan = build_plan([entry()], 'azure', root, kind='images')
            with patch('download_lila.shutil.which', return_value=None), patch('download_lila.execute') as execute:
                result = run_plan(plan, 'azure', dry_run=True)
            execute.assert_not_called()
            self.assertEqual(result[0]['status'], 'planned')
            self.assertFalse(root.exists())

    def test_retry_failure_isolated_and_aws_config_is_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_plan([entry()], 'aws', Path(tmp), kind='all')
            codes = iter([1, 1, 0, 0])

            def execute(command, env):
                self.assertNotEqual(env['AWS_CONFIG_FILE'], 'original-user-config')
                self.assertIn('max_concurrent_requests = 17', Path(env['AWS_CONFIG_FILE']).read_text())
                return next(codes)

            with patch.dict(os.environ, {'AWS_CONFIG_FILE': 'original-user-config'}), \
                 patch('download_lila.shutil.which', return_value='/bin/aws'), \
                 patch('download_lila.execute', side_effect=execute), patch('download_lila.time.sleep'):
                results = run_plan(plan, 'aws', workers=17, retries=1)
                self.assertEqual(os.environ['AWS_CONFIG_FILE'], 'original-user-config')
            self.assertEqual([r['status'] for r in results], ['failed', 'ok', 'ok'])


if __name__ == '__main__':
    unittest.main()
