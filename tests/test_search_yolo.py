from argparse import Namespace
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

import search_yolo
from search_worker import digest, execute, write_json


class FakeGPU:
    """Fake only the expensive process boundary; use real studies, files and reports."""
    def __init__(self):
        self.calls = []
        self.fail_test_once = False
        self.kill_train_once = False

    def launch(self, command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        phase = job['phase']
        self.calls.append(job)
        output = Path(job['result'])
        if phase == 'train' and self.kill_train_once:
            self.kill_train_once = False
            return Mock(wait=Mock(return_value=137))
        if phase == 'test' and self.fail_test_once:
            self.fail_test_once = False
            write_json(output, {'status':'failed', 'error':'temporary evaluation error'})
            return Mock(wait=Mock(return_value=1))
        if phase == 'prepare':
            target = Path(job['target'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(job['model'].encode())
            result = {'checkpoint':str(target), 'sha256':digest(target)}
        elif phase == 'train':
            best = Path(job['train_dir'])/'weights/best.pt'
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes((job['model']+str(job['trial'])).encode())
            result = {'best_checkpoint':str(best), 'checkpoint_sha256':digest(best), 'recorded_epochs':2}
        else:
            score = {'yolov8n.pt':.2, 'yolov8s.pt':.3, 'yolo11n.pt':.5, 'yolo11s.pt':.6}[job['model']]
            score += job['trial']*.01
            if phase == 'test':
                score = 1-score  # Test favors the opposite models; it must never select winners.
            result = {'split':phase, 'checkpoint_sha256':digest(job['checkpoint']),
                      'metrics':{'precision':score, 'recall':score, 'map50':score, 'map50_95':score}}
        write_json(output, {'status':'ok', **result})
        return Mock(wait=Mock(return_value=0))


@unittest.skipUnless(importlib.util.find_spec('optuna'), 'Install requirements-search.txt for real Optuna integration')
class SearchIntegrationTests(unittest.TestCase):
    def args(self, root, trials=1, test_policy='every-trial'):
        data = root/'dataset.yaml'
        data.write_text(yaml.safe_dump({'path':str(root), 'train':'images/train', 'val':'images/val',
                                       'test':'images/test', 'names':{0:'cat'}}))
        output = root/'search'
        output.mkdir(exist_ok=True)
        return Namespace(data=data, output=output, weights_dir=root/'weights', epochs=2, patience=1,
                         imgsz=640, batch=None, eval_batch=2, workers=0, device='cpu', seed=42,
                         amp=False, trials=trials, test_policy=test_policy)

    def run_fake(self, args, gpu):
        with patch('search_yolo.subprocess.Popen', side_effect=gpu.launch), \
                patch('training_guard.supervise', side_effect=lambda command: gpu.launch(command).wait()), \
                contextlib.redirect_stdout(io.StringIO()):
            search_yolo.run(args)

    def test_one_or_two_selected_models_and_resume_identity(self):
        for chosen in [('yolo11s.pt',), ('yolo11n.pt', 'yolo11s.pt')]:
            with self.subTest(chosen=chosen), tempfile.TemporaryDirectory() as tmp:
                args, gpu = self.args(Path(tmp)), FakeGPU()
                full_output = search_yolo.automatic_output(args)
                args.models = chosen
                self.assertNotEqual(full_output, search_yolo.automatic_output(args))
                self.run_fake(args, gpu)
                self.assertEqual([j['model'] for j in gpu.calls if j['phase'] == 'train'], list(chosen))
                report = json.loads((args.output/'summary.json').read_text())
                self.assertEqual({r['model'] for r in report['iterations']}, set(chosen))
                self.run_fake(args, gpu)
                self.assertEqual(len([j for j in gpu.calls if j['phase'] == 'train']), len(chosen))
                args.models = search_yolo.MODELS
                with self.assertRaisesRegex(ValueError, 'configuration/data changed'):
                    self.run_fake(args, gpu)

    def test_round_robin_resume_and_validation_only_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, gpu = self.args(root), FakeGPU()
            self.run_fake(args, gpu)
            args.trials = 2  # Total target, not two additional trials per model.
            self.run_fake(args, gpu)
            training = [j for j in gpu.calls if j['phase']=='train']
            self.assertEqual([j['model'] for j in training], list(search_yolo.MODELS)*2)
            self.assertEqual(len([j for j in gpu.calls if j['phase']=='prepare']), 4)
            for job in training:
                self.assertIn(job['batch'], [64, 128, 256])
                self.assertEqual(job['batch'], job['params']['batch'])
            self.assertEqual([job['batch'] for job in training[:4]], [64]*4)
            self.assertTrue(all(j['eval_batch'] == 2 for j in gpu.calls if j['phase'] in ('val', 'test')))
            with (args.output/'iterations.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['batch']) for row in rows], [job['batch'] for job in training])
            report = json.loads((args.output/'summary.json').read_text())
            self.assertEqual(len(report['iterations']), 8)
            self.assertEqual(report['winners']['n']['model'], 'yolo11n.pt')
            self.assertEqual(report['winners']['s']['model'], 'yolo11s.pt')
            self.assertEqual(report['winners']['n']['trial'], 1)
            human = (args.output/'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('本次搜索完成', human)
            self.assertIn('按 Size 比较架构', human)
            self.assertIn('### yolov8n.pt', human)
            self.assertIn('+1.00 个百分点', human)
            self.assertIn('| n | 测试集 | 49.00%', human)
            self.assertIn('[best_n.pt](best/best_n.pt)', human)

            self.assertEqual((args.output/'best/best_n.pt').read_bytes(), b'yolo11n.pt1')
            self.assertTrue((args.output/'rounds/round_0002.json').is_file())
            count = len(gpu.calls)
            self.run_fake(args, gpu)
            self.assertEqual(len(gpu.calls), count)  # Completed search performs no new GPU stages.

    def test_baseline_and_batch_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), trials=5, test_policy='final'), FakeGPU()
            self.run_fake(args, gpu)
            jobs = [j for j in gpu.calls if j['phase'] == 'train' and j['model'] == 'yolov8n.pt']
            self.assertEqual(jobs[0]['params'], {'batch': 64, 'optimizer': 'auto'})
            self.assertEqual([j['batch'] for j in jobs[1:4]], [64, 128, 256])
            for job in jobs[1:4]:
                self.assertEqual(job['params']['optimizer'], 'AdamW')
                self.assertEqual(job['params']['lr0'], .001)
            self.assertEqual(jobs[4]['params']['optimizer'], 'AdamW')
            self.assertTrue(1e-4 <= jobs[4]['params']['lr0'] <= 3e-3)
            self.assertIn(jobs[4]['batch'], [64, 128, 256])
            records = json.loads((args.output/'summary.json').read_text())['iterations']
            self.assertEqual(records[0]['params']['optimizer'], 'auto')
            self.assertEqual(records[4]['params']['optimizer'], 'AdamW')
            self.assertEqual(len([j for j in gpu.calls if j['phase'] == 'test']), 2)

    def test_automatic_name_stable_for_resume_but_changes_with_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            original = search_yolo.automatic_output(args)
            args.trials = 20
            args.workers = 16
            self.assertEqual(search_yolo.automatic_output(args), original)
            args.epochs += 1
            self.assertNotEqual(search_yolo.automatic_output(args), original)
            args.epochs -= 1
            args.data.write_text(args.data.read_text().replace('cat', 'dog'))
            self.assertNotEqual(search_yolo.automatic_output(args), original)

    def test_explicit_batch_disables_search_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp)), FakeGPU()
            args.batch = 32
            self.run_fake(args, gpu)
            args.trials = 2
            self.run_fake(args, gpu)
            training = [j for j in gpu.calls if j['phase'] == 'train']
            self.assertEqual(len(training), 8)
            self.assertTrue(all(j['batch'] == j['params']['batch'] == 32 for j in training))

    def test_test_stage_failure_retries_without_training_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp)), FakeGPU()
            gpu.fail_test_once = True
            with self.assertRaisesRegex(ValueError, 'test stages failed'):
                self.run_fake(args, gpu)
            count = len([j for j in gpu.calls if j['phase']=='train'])
            self.run_fake(args, gpu)
            self.assertEqual(len([j for j in gpu.calls if j['phase']=='train']), count)
            self.assertEqual(len([j for j in gpu.calls if j['phase']=='test']), 5)

    def test_sigkill_stops_batch_and_preserves_running_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp)), FakeGPU()
            gpu.kill_train_once = True
            with self.assertRaises(search_yolo.JobError):
                self.run_fake(args, gpu)
            before = json.loads((args.output/'v8n/t000/config.json').read_text())
            self.run_fake(args, gpu)
            after = json.loads((args.output/'v8n/t000/config.json').read_text())
            self.assertEqual(before, after)
            self.assertEqual(len(json.loads((args.output/'summary.json').read_text())['iterations']), 4)

    def test_final_policy_evaluates_only_two_selected_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), test_policy='final'), FakeGPU()
            self.run_fake(args, gpu)
            self.assertEqual([j['model'] for j in gpu.calls if j['phase']=='test'], ['yolo11n.pt','yolo11s.pt'])

    def test_final_evaluation_failure_is_reported_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), test_policy='final'), FakeGPU()
            gpu.fail_test_once = True
            with self.assertRaises(search_yolo.JobError):
                self.run_fake(args, gpu)
            summary = json.loads((args.output/'summary.json').read_text())
            self.assertEqual(summary['winners']['n']['test']['status'], 'failed')
            human = (args.output/'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('尚未确认全部完成', human)
            self.assertIn('失败，需重试', human)
            self.assertNotIn('**状态：本次搜索完成。**', human)

            self.run_fake(args, gpu)
            self.assertEqual(len([j for j in gpu.calls if j['phase']=='train']), 4)

    def test_changed_experiment_cannot_mix_with_old_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp)), FakeGPU()
            self.run_fake(args, gpu)
            args.epochs += 1
            count = len(gpu.calls)
            with self.assertRaisesRegex(ValueError, 'configuration/data changed'):
                self.run_fake(args, gpu)
            self.assertEqual(len(gpu.calls), count)


class HumanReportTests(unittest.TestCase):
    def test_missing_and_zero_metrics_are_distinct(self):
        from search_report import metric, delta
        self.assertEqual(metric({}, 'test', 'map50_95'), '—')
        self.assertEqual(metric({'test': {'status': 'ok', 'metrics': {'map50_95': 0}}}, 'test', 'map50_95'), '0.00%')
        self.assertEqual(delta({}, None), '—（无有效基线）')


class BatchSpaceTests(unittest.TestCase):
    def test_each_batch_choice_is_sampled_as_a_category(self):
        for batch in (64, 128, 256):
            with self.subTest(batch=batch):
                trial = Mock()
                trial.suggest_categorical.side_effect = [batch, 'AdamW']
                params = search_yolo.sample_params(trial)
                self.assertEqual(params['batch'], batch)
                self.assertEqual(trial.suggest_categorical.call_args_list[0].args, ('batch', [64, 128, 256]))


class WorkerTests(unittest.TestCase):
    def test_incomplete_initial_checkpoint_is_not_accepted_by_size_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'yolov8n.pt'
            target.write_bytes(b'x'*100_001)
            sdk = SimpleNamespace(YOLO=Mock(side_effect=RuntimeError('truncated checkpoint')))
            modules = {'ultralytics':sdk, 'ultralytics.utils':SimpleNamespace(),
                       'ultralytics.utils.downloads':SimpleNamespace(attempt_download_asset=Mock())}
            with patch.dict('sys.modules', modules), self.assertRaisesRegex(ValueError, 'cannot be loaded'):
                execute({'phase':'prepare','target':str(target),'source':str(Path(tmp)/'missing.pt')})
            self.assertEqual(target.stat().st_size, 100_001)

    def test_test_evaluation_records_metrics_and_uses_correct_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'best.pt'
            checkpoint.write_bytes(b'checkpoint')
            metrics = SimpleNamespace(box=SimpleNamespace(mp=.6, mr=.5, map50=.7, map=.4),
                                      speed={'inference':1.2}, summary=lambda:[{'Class':'cat','mAP50-95':.4}])
            model = Mock()
            model.val.return_value = metrics
            sdk = SimpleNamespace(YOLO=Mock(return_value=model), __version__='fake')
            job = {'phase':'test','checkpoint':str(checkpoint),'data':'data.yaml','imgsz':640,
                   'eval_batch':2,'workers':0,'device':'cpu','trial_dir':tmp}
            with patch.dict('sys.modules', {'ultralytics':sdk,'torch':SimpleNamespace(__version__='fake')}):
                result = execute(job)
            self.assertEqual(result['metrics']['map50_95'], .4)
            self.assertEqual(model.val.call_args.kwargs['split'], 'test')
            self.assertEqual(model.val.call_args.kwargs['conf'], .001)
            self.assertEqual(result['per_class'][0]['Class'], 'cat')

    def test_json_accepts_scientific_scalar_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metric.json'
            write_json(path, {'metric':SimpleNamespace(item=lambda:.25)})
            self.assertEqual(json.loads(path.read_text())['metric'], .25)

    def test_explicit_optimizer_and_learning_rate_reach_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'training'
            model = Mock()
            def train(**kwargs):
                (directory/'weights').mkdir(parents=True)
                (directory/'weights/best.pt').write_bytes(b'best')
                (directory/'weights/last.pt').write_bytes(b'last')
            model.train.side_effect = train
            sdk = SimpleNamespace(YOLO=Mock(return_value=model), __version__='fake')
            job = {'phase':'train','train_dir':str(directory),'epochs':2,'data':'data.yaml','imgsz':640,
                   'batch':4,'workers':0,'device':'cpu','seed':42,'patience':1,'amp':False,
                   'initial_checkpoint':'initial.pt','params':{'batch':256,'optimizer':'AdamW','lr0':.002}}
            with patch.dict('sys.modules', {'ultralytics':sdk, 'torch':SimpleNamespace(__version__='fake')}):
                result = execute(job)
            self.assertEqual(model.train.call_args.kwargs['optimizer'], 'AdamW')
            self.assertEqual(model.train.call_args.kwargs['lr0'], .002)
            self.assertEqual(model.train.call_args.kwargs['batch'], 256)
            self.assertFalse(model.train.call_args.kwargs['cache'])
            self.assertEqual(result['checkpoint_sha256'], digest(directory/'weights/best.pt'))

    def test_recovery_of_finished_checkpoint_skips_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'weights').mkdir()
            for name in ('best.pt','last.pt'):
                (directory/'weights'/name).write_bytes(b'finished')
            model = Mock(ckpt={'epoch':-1})
            sdk = SimpleNamespace(YOLO=Mock(return_value=model), __version__='fake')
            with patch.dict('sys.modules', {'ultralytics':sdk, 'torch':SimpleNamespace(__version__='fake')}):
                execute({'phase':'train','train_dir':str(directory),'epochs':2})
            model.train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
