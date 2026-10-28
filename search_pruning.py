"""Optuna epoch reporting and successive halving at explicit epoch milestones."""
from copy import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import optuna


class MilestonePruner(optuna.pruners.BasePruner):
    """Keep real epoch steps in storage; map milestones onto SHA resource rungs.

    E.g. epochs 30/80 map to resources 1/2. SHA still owns ranking, tie
    handling and persistent rung state; it cannot prune between milestones.
    """
    def __init__(self, milestones):
        self.milestones = tuple(milestones)
        self.halving = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)

    def prune(self, study, trial):
        if trial.last_step not in self.milestones:
            return False
        key = f'prune_at_epoch_{trial.last_step}'
        if key in trial.user_attrs:
            return trial.user_attrs[key]
        mapped = copy(trial)
        mapped.intermediate_values = {
            2**i: trial.intermediate_values[epoch]
            for i, epoch in enumerate(self.milestones) if epoch in trial.intermediate_values
        }
        decision = self.halving.prune(study, mapped)
        optuna.trial.Trial(study, trial._trial_id).set_user_attr(key, decision)
        return decision


class PrunedRun(Exception):
    def __init__(self, result):
        self.result = result
        super().__init__(result['reason'])


class EpochReporter:
    """Runs after an epoch checkpoint was saved; decisions survive worker death."""
    def __init__(self, job):
        self.job = job
        spec = job['pruning']
        self.storage = optuna.storages.RDBStorage(
            url=spec['storage'], engine_kwargs={'connect_args': {'timeout': 60}})
        self.study = optuna.load_study(study_name=spec['study'], storage=self.storage,
                                      pruner=MilestonePruner(spec['milestones']))
        frozen = next(t for t in self.study.get_trials() if t.number == job['trial'])
        if frozen.state != optuna.trial.TrialState.RUNNING:
            self.close()
            raise ValueError('Epoch reporter requires a RUNNING Optuna trial')
        if frozen.intermediate_values and not (Path(job['train_dir'])/'weights/last.pt').is_file():
            self.close()
            raise ValueError('Reported epochs exist but last.pt is missing; cannot safely restart this trial')
        self.trial = optuna.trial.Trial(self.study, frozen._trial_id)
        self.seen = set(frozen.intermediate_values)
        self.directory = Path(job['trial_dir'])

    def recover(self, checkpoint_epoch):
        """Replay a saved epoch missed by the callback if the worker died after save."""
        weights = Path(self.job['train_dir'])/'weights'
        if not self.job['pruning']['baseline']:
            for epoch in self.job['pruning']['milestones']:
                if self.trial.user_attrs.get(f'prune_at_epoch_{epoch}'):
                    frozen = next(t for t in self.study.get_trials() if t.number == self.job['trial'])
                    self.stop(epoch, frozen.intermediate_values[epoch], weights/'last.pt', weights/'best.pt')
        csv_path = Path(self.job['train_dir'])/'results.csv'
        if not csv_path.is_file():
            return
        with csv_path.open(encoding='utf-8-sig') as stream:
            rows = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(stream)]
        for row in rows:
            epoch = int(row['epoch'])
            if checkpoint_epoch >= 0 and epoch > checkpoint_epoch + 1:
                continue
            if epoch in self.seen:
                continue
            self(SimpleNamespace(epoch=epoch-1,
                                 metrics={'metrics/mAP50-95(B)':float(row['metrics/mAP50-95(B)'])},
                                 last=weights/'last.pt', best=weights/'best.pt'))

    def close(self):
        self.storage.remove_session()
        self.storage.engine.dispose()

    def __call__(self, trainer):
        if getattr(getattr(trainer, 'validator', None), 'training', True) is False:
            return  # final_eval validates best.pt, sometimes at a synthetic epoch+1.
        epoch = trainer.epoch + 1
        # final_eval() also invokes this callback; do not report/prune twice.
        if epoch in self.seen:
            return
        raw = trainer.metrics.get('metrics/mAP50-95(B)')
        if raw is None:
            raise ValueError('Epoch validation did not return metrics/mAP50-95(B)')
        score = float(raw)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Invalid epoch validation mAP50-95: {score}')
        self.trial.report(score, step=epoch)
        self.seen.add(epoch)
        should_prune = self.trial.should_prune()
        exempt = self.job['pruning']['baseline']
        event = {'epoch': epoch, 'val_map50_95': score,
                 'milestone': epoch in self.job['pruning']['milestones'],
                 'decision': 'baseline_exempt' if exempt and should_prune else
                             ('prune' if should_prune else 'continue')}
        with (self.directory/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        if should_prune and not exempt:
            self.stop(epoch, score, trainer.last, trainer.best)

    def stop(self, epoch, score, last, best):
        result = {'status': 'pruned', 'pruned_epoch': epoch, 'recorded_epochs': epoch,
                  'val_map50_95': score, 'last_checkpoint': str(last), 'best_checkpoint': str(best),
                  'reason': f'Successive halving stopped this candidate at epoch {epoch}'}
        from search_worker import write_json
        # Persist before raising so recovery cannot accidentally revive a pruned trial.
        write_json(self.directory/'pruned.json', result)
        print(f'[Pruner] {result["reason"]}; val mAP50-95={score:.5f}', flush=True)
        raise PrunedRun(result)


class DeferredEarlyStopping:
    """Preserve early-stop history, but let scheduled screening finish first."""
    def __init__(self, stopper, minimum_epoch):
        self.stopper, self.minimum_epoch = stopper, minimum_epoch

    def __getattr__(self, name):
        return getattr(self.stopper, name)

    def __setattr__(self, name, value):
        if name in ('stopper', 'minimum_epoch'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.stopper, name, value)

    def __call__(self, epoch, fitness):
        stop = self.stopper(epoch, fitness)
        if epoch < self.minimum_epoch:
            self.stopper.possible_stop = False
            return False
        return stop


def validation_trainer(job):
    """Use the same validation size and batch for small/full training and resume."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SearchTrainer(DetectionTrainer):
        def _setup_train(self):
            super()._setup_train()
            self.stopper = DeferredEarlyStopping(self.stopper, max(job['pruning']['milestones'], default=0))

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'val':
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            old = self.args.imgsz
            try:
                self.args.imgsz = job['eval_imgsz']
                return super().get_dataloader(dataset_path, job['eval_batch'], rank, mode)
            finally:
                self.args.imgsz = old

        def get_validator(self):
            validator = super().get_validator()
            validator.args.imgsz = job['eval_imgsz']
            validator.args.batch = job['eval_batch']
            validator.args.conf = 0.001
            validator.args.iou = 0.7
            validator.args.max_det = 300
            return validator

    return SearchTrainer
