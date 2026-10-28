"""Real Optuna/SQLite decisions; simulate only GPU training and validation."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import optuna
import yaml

import search_yolo
from search_pruning import DeferredEarlyStopping, EpochReporter, MilestonePruner, PrunedRun, validation_trainer
from search_worker import digest, execute, write_json


class EpochGPU:
    def __init__(self):
        self.calls = []
        self.kill_after_prune = False
        self.fail_test = False

    def launch(self, command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        self.calls.append(job)
        phase = job['phase']
        if phase == 'prepare':
            target = Path(job['target'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(job['model'].encode())
            result = {'checkpoint': str(target), 'sha256': digest(target)}
        elif phase == 'train':
            best = Path(job['train_dir'])/'weights/best.pt'
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(f"{job['stage']} {job['trial']}".encode())
            last = best.with_name('last.pt')
            last.write_bytes(b'checkpoint')
            # t1 is bad; baseline, t2, t3 survive. Baseline seeds real SHA rungs.
            score = {0: .5, 1: .1, 2: .7, 3: .8}.get(job['trial'], .6)
            reporter = EpochReporter(job)
            try:
                for epoch in range(job['epochs']):
                    reporter(SimpleNamespace(epoch=epoch, metrics={'metrics/mAP50-95(B)':score},
                                             last=last, best=best))
                result = {'best_checkpoint': str(best), 'checkpoint_sha256': digest(best),
                          'recorded_epochs': job['epochs']}
            except PrunedRun as exc:
                if self.kill_after_prune:
                    self.kill_after_prune = False
                    return Mock(wait=Mock(return_value=137))
                result = exc.result
            finally:
                reporter.close()
        else:
            if phase == 'test' and self.fail_test:
                self.fail_test = False
                write_json(Path(job['result']), {'status':'failed', 'error':'test unavailable'})
                return Mock(wait=Mock(return_value=1))
            score = {0:.5, 1:.1, 2:.7, 3:.8}.get(job['trial'], .6)
            if job.get('stage') == 'screen':
                score = min(.99, score + .1)
            if phase == 'test':
                score = 1-score  # Test must not choose the winner.
            result = {'split':phase, 'checkpoint_sha256':digest(job['checkpoint']),
                      'metrics':dict.fromkeys(('precision','recall','map50','map50_95'), score)}
        write_json(Path(job['result']), {'status':'ok', **result})
        return Mock(wait=Mock(return_value=0))


class ModeTests(unittest.TestCase):
    def args(self, root, mode):
        data = root/'data.yaml'
        data.write_text(yaml.safe_dump({'train':'images/train', 'val':'images/val', 'test':'images/test',
                                       'names':{0:'cat'}}))
        output = root/'run'
        output.mkdir()
        return Namespace(mode=mode, data=data, output=output, weights_dir=root/'weights',
                         models=['yolo11n.pt'], trials=4, epochs=5, patience=20, imgsz=640,
                         batch=64, eval_batch=2, workers=0, device='cpu', seed=42, amp=False,
                         test_policy='final', search_imgsz=448, search_epochs=3,
                         prune_epochs=[1, 2], promote=2, dry_run=False)

    def run_fake(self, args, gpu, backfill=False):
        with patch('search_yolo.subprocess.Popen', side_effect=gpu.launch), \
                patch('training_guard.supervise', side_effect=lambda cmd: gpu.launch(cmd).wait()), \
                contextlib.redirect_stdout(io.StringIO()):
            (search_yolo.test_only if backfill else search_yolo.run)(args)

    def test_full_size_pruning_real_sqlite_resume_and_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), 'prune'), EpochGPU()
            self.run_fake(args, gpu)
            records = json.loads((args.output/'summary.json').read_text())['iterations']
            self.assertEqual([r['state'] for r in records], ['COMPLETE','PRUNED','COMPLETE','COMPLETE'])
            self.assertEqual(records[1]['train']['pruned_epoch'], 1)
            self.assertEqual(records[0]['train']['recorded_epochs'], 5)
            self.assertTrue(all(j['imgsz']==640 for j in gpu.calls if j['phase']=='train'))
            self.assertEqual([j['trial'] for j in gpu.calls if j['phase']=='test'], [3])
            count = len(gpu.calls)
            self.run_fake(args, gpu)
            self.assertEqual(len(gpu.calls), count)
            self.run_fake(args, gpu, backfill=True)
            self.assertEqual([j['trial'] for j in gpu.calls[count:] if j['phase']=='test'], [0, 2])
            report = (args.output/'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('提前淘汰（第 1 轮）', report)
            self.assertIn('分阶段预算与剪枝记录', report)

    def test_progressive_promotes_parameters_not_weights_and_separates_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), 'progressive'), EpochGPU()
            self.run_fake(args, gpu)
            training = [j for j in gpu.calls if j['phase']=='train']
            small = [j for j in training if j['stage']=='screen']
            full = [j for j in training if j['stage']=='full']
            self.assertEqual(len(small), 4)
            self.assertEqual(len(full), 3)  # baseline + top two surviving candidates
            self.assertTrue(all(j['imgsz']==448 and j['eval_imgsz']==640 for j in small))
            self.assertTrue(all(j['imgsz']==640 and j['epochs']==5 for j in full))
            self.assertEqual([j['parent']['trial'] for j in full], [0, 3, 2])
            self.assertTrue(all(j['initial_checkpoint']==small[0]['initial_checkpoint'] for j in full))
            for job in full:
                self.assertEqual(job['params'], small[job['parent']['trial']]['params'])
            self.assertNotEqual(small[0]['pruning']['storage'], full[0]['pruning']['storage'])
            self.assertTrue(all(j['stage']=='full' for j in gpu.calls if j['phase']=='test'))
            summary = json.loads((args.output/'summary.json').read_text())
            self.assertTrue(all(r['config']['stage']=='full' for r in summary['iterations']))
            self.assertEqual(summary['winners']['n']['trial'], 2)
            count = len(gpu.calls)
            self.run_fake(args, gpu)
            self.assertEqual(len(gpu.calls), count)
            args.trials += 1
            with self.assertRaisesRegex(ValueError, 'configuration/data changed'):
                self.run_fake(args, gpu)

    def test_killed_after_prune_marker_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), 'prune'), EpochGPU()
            gpu.kill_after_prune = True
            with self.assertRaises(search_yolo.JobError):
                self.run_fake(args, gpu)
            count = len(gpu.calls)
            self.run_fake(args, gpu)
            self.assertNotIn(1, [j['trial'] for j in gpu.calls[count:] if j['phase']=='train'])
            records = json.loads((args.output/'summary.json').read_text())['iterations']
            self.assertEqual(records[1]['state'], 'PRUNED')

    def test_failed_final_test_only_retries_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, gpu = self.args(Path(tmp), 'progressive'), EpochGPU()
            gpu.fail_test = True
            with self.assertRaises(search_yolo.JobError):
                self.run_fake(args, gpu)
            count = len(gpu.calls)
            self.run_fake(args, gpu)
            self.assertEqual([j['phase'] for j in gpu.calls[count:]], ['test'])

    def test_identity_isolates_modes_milestones_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp), 'prune')
            original = search_yolo.automatic_output(args)
            args.prune_epochs = [2, 3]
            self.assertNotEqual(original, search_yolo.automatic_output(args))
            args.prune_epochs = [1, 2]
            args.mode = 'progressive'
            progressive = search_yolo.automatic_output(args)
            self.assertNotEqual(original, progressive)
            args.search_imgsz = 384
            self.assertNotEqual(progressive, search_yolo.automatic_output(args))


class PrunerTests(unittest.TestCase):
    def reporter_fixture(self, root):
        storage = optuna.storages.RDBStorage(url='sqlite:///'+(root/'study.sqlite3').as_posix())
        study = optuna.create_study(storage=storage, study_name='test', direction='maximize',
                                   pruner=MilestonePruner([2,4]))
        baseline = study.ask()
        for epoch in range(1,5):
            baseline.report(.8,epoch)
            baseline.should_prune()
        study.tell(baseline,.8)
        trial = study.ask()
        job = {'phase':'train','trial':trial.number,'trial_dir':str(root),'train_dir':str(root/'training'),
               'pruning':{'storage':str(storage.url),'study':'test','milestones':[2,4],'baseline':False}}
        return storage, study, job

    def test_final_validation_is_not_an_extra_epoch_or_pruning_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage, study, job = self.reporter_fixture(Path(tmp))
            reporter = EpochReporter(job)
            try:
                trainer = SimpleNamespace(epoch=0, metrics={'metrics/mAP50-95(B)':.1},
                                          last='last.pt',best='best.pt',validator=SimpleNamespace(training=True))
                reporter(trainer)
                trainer.epoch = 1  # Modern final_eval reports a synthetic epoch+1.
                trainer.validator.training = False
                reporter(trainer)
                frozen = study.get_trials()[-1]
                self.assertEqual(frozen.intermediate_values, {1:.1})
                self.assertFalse((Path(tmp)/'pruned.json').exists())
            finally:
                reporter.close()
                storage.remove_session()
                storage.engine.dispose()

    def test_recover_saved_epoch_not_yet_reported_and_recover_prune_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage, study, job = self.reporter_fixture(root)
            train = Path(job['train_dir'])
            train.mkdir()
            (train/'results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.1\n2,0.1\n')
            (train/'weights').mkdir()
            (train/'weights/last.pt').write_bytes(b'checkpoint')
            reporter = EpochReporter(job)
            try:
                with self.assertRaises(PrunedRun):
                    reporter.recover(1)
                self.assertEqual(study.get_trials()[-1].intermediate_values, {1:.1,2:.1})
            finally:
                reporter.close()
            # Simulate death after database decision but before writing the marker.
            (root/'pruned.json').unlink()
            (train/'results.csv').unlink()
            reporter = EpochReporter(job)
            try:
                with self.assertRaises(PrunedRun):
                    reporter.recover(1)
                self.assertTrue((root/'pruned.json').is_file())
            finally:
                reporter.close()
                storage.remove_session()
                storage.engine.dispose()

    def test_baseline_exemption_and_nonfinite_metric_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage, study, job = self.reporter_fixture(root)
            job['pruning']['baseline'] = True
            reporter = EpochReporter(job)
            try:
                trainer = SimpleNamespace(epoch=1, metrics={'metrics/mAP50-95(B)':.1}, last='last',best='best')
                reporter(trainer)  # Would lose to the existing stronger trial, but exempt.
                self.assertFalse((root/'pruned.json').exists())
                trainer.epoch = 2
                trainer.metrics['metrics/mAP50-95(B)'] = float('nan')
                with self.assertRaisesRegex(ValueError,'Invalid epoch'):
                    reporter(trainer)
            finally:
                reporter.close()
                storage.remove_session()
                storage.engine.dispose()

    def test_worker_returns_pruned_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage, study, job = self.reporter_fixture(root)
            job.update(epochs=5, data='data.yaml', imgsz=448, eval_imgsz=640, eval_batch=2,
                       batch=64, workers=0, device='cpu', seed=42, patience=20, amp=False,
                       initial_checkpoint='official.pt', params={'batch':64,'optimizer':'AdamW','lr0':.001})
            callbacks = {}
            model = Mock()
            model.add_callback.side_effect = lambda name, callback: callbacks.update({name:callback})
            weights = Path(job['train_dir'])/'weights'
            def train(**kwargs):
                weights.mkdir(parents=True)
                (weights/'last.pt').write_bytes(b'resumable')
                (weights/'best.pt').write_bytes(b'best')
                for epoch in range(5):
                    callbacks['on_fit_epoch_end'](SimpleNamespace(epoch=epoch,
                        metrics={'metrics/mAP50-95(B)':.1},last=weights/'last.pt',best=weights/'best.pt'))
            model.train.side_effect = train
            sdk = SimpleNamespace(YOLO=Mock(return_value=model),__version__='fake')
            try:
                with patch.dict('sys.modules', {'ultralytics':sdk,'torch':SimpleNamespace(__version__='fake')}), \
                        patch('search_pruning.validation_trainer',return_value=object), \
                        patch('training_diagnostics.install_phase_logging'):
                    result = execute(job)
                    self.assertEqual(result['status'],'pruned')
                    self.assertEqual(result['pruned_epoch'],2)
                    self.assertEqual(model.train.call_args.kwargs['epochs'],5)
                    self.assertEqual((weights/'last.pt').read_bytes(),b'resumable')
                    self.assertEqual(execute(job),result)
                    self.assertEqual(sdk.YOLO.call_count,1)
            finally:
                storage.remove_session()
                storage.engine.dispose()

    def test_worker_resume_reports_saved_epoch_without_restarting_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage, study, job = self.reporter_fixture(root)
            job.update(epochs=5, device='cpu', workers=0, eval_imgsz=640, eval_batch=2)
            job['pruning']['baseline'] = True
            directory = Path(job['train_dir'])
            weights = directory/'weights'
            weights.mkdir(parents=True)
            for name in ('last.pt','best.pt'):
                (weights/name).write_bytes(b'checkpoint')
            (directory/'results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.5\n')
            model = Mock(ckpt={'epoch':0})
            sdk = SimpleNamespace(YOLO=Mock(return_value=model),__version__='fake')
            try:
                with patch.dict('sys.modules',{'ultralytics':sdk,'torch':SimpleNamespace(__version__='fake')}), \
                        patch('search_pruning.validation_trainer',return_value=object), \
                        patch('training_diagnostics.install_phase_logging'):
                    execute(job)
                self.assertEqual(sdk.YOLO.call_args.args,(str(weights/'last.pt'),))
                self.assertEqual(model.train.call_args.kwargs,
                                 {'resume':True,'device':'cpu','workers':0,'trainer':object})
                self.assertEqual(study.get_trials()[-1].intermediate_values,{1:.5})
            finally:
                storage.remove_session()
                storage.engine.dispose()

    def test_early_stop_waits_for_last_checkpoint_without_resetting_history(self):
        stopper = Mock(possible_stop=True, best_epoch=3)
        stopper.return_value = True
        gated = DeferredEarlyStopping(stopper,80)
        self.assertFalse(gated(30,.2))
        self.assertFalse(stopper.possible_stop)
        self.assertEqual(gated.best_epoch,3)
        gated.best_epoch = 5
        self.assertEqual(stopper.best_epoch,5)
        self.assertTrue(gated(80,.1))
        self.assertEqual(stopper.call_count,2)

    def test_real_halving_uses_exact_milestones_and_stores_actual_epochs(self):
        study = optuna.create_study(direction='maximize', pruner=MilestonePruner([30,80]))
        baseline = study.ask()
        for e in range(1,81):
            baseline.report(.5, e)
            self.assertFalse(baseline.should_prune())
        study.tell(baseline, .5)
        weak = study.ask()
        for e in range(1,30):
            weak.report(.1, e)
            self.assertFalse(weak.should_prune())
        weak.report(.1, 30)
        self.assertTrue(weak.should_prune())
        study.tell(weak, state=optuna.trial.TrialState.PRUNED)
        late = study.ask()
        late.report(.9, 30)
        self.assertFalse(late.should_prune())
        for e in range(31,80):
            late.report(.01, e)
            self.assertFalse(late.should_prune())
        late.report(.01, 80)
        self.assertTrue(late.should_prune())
        self.assertIn(80, study.get_trials()[-1].intermediate_values)

    def test_fixed_validation_dimensions_reach_loader_and_validator(self):
        class Trainer:
            def __init__(self):
                self.args = SimpleNamespace(imgsz=448)
            def get_dataloader(self, path, batch, rank, mode):
                return self.args.imgsz, batch, mode
            def get_validator(self):
                return SimpleNamespace(args=SimpleNamespace(imgsz=self.args.imgsz))
        with patch.dict('sys.modules', {'ultralytics.models.yolo.detect': SimpleNamespace(DetectionTrainer=Trainer)}):
            trainer = validation_trainer({'eval_imgsz':640, 'eval_batch':8})()
            self.assertEqual(trainer.get_dataloader('train',64,mode='train'), (448,64,'train'))
            self.assertEqual(trainer.get_dataloader('val',128,mode='val'), (640,8,'val'))
            self.assertEqual(trainer.args.imgsz,448)
            validator = trainer.get_validator()
            self.assertEqual((validator.args.imgsz,validator.args.batch), (640,8))


class CLITests(unittest.TestCase):
    def test_dry_run_defaults_and_invalid_milestones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data.yaml'
            data.write_text('train: images/train\nval: images/val\ntest: images/test\nnames: [cat]\n')
            output = root/'preview'
            for mode in ('standard','prune','progressive'):
                args = ['tune.py','--data',str(data),'--output',str(output),'--mode',mode,'--dry-run']
                with patch('sys.argv',args), contextlib.redirect_stdout(io.StringIO()) as printed:
                    search_yolo.main()
                result = json.loads(printed.getvalue())
                self.assertEqual(result['mode'],mode)
                self.assertEqual(result['max_epochs_per_trial'],100 if mode=='standard' else 250)
                self.assertEqual(result['batch_choices'],[64,128,256] if mode=='standard' else [64])
                self.assertFalse(output.exists())
            args = ['tune.py','--data',str(data),'--mode','prune','--prune-epochs','80','30','--dry-run']
            with patch('sys.argv',args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                search_yolo.main()


if __name__ == '__main__':
    unittest.main()
