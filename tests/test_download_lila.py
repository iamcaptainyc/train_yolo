import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_lila import (build_plan, cloud_url, command_for, load_catalog, read_catalog,
                           relative_prefix, run_plan, select_dataset)


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


class ExecutionTests(unittest.TestCase):
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
