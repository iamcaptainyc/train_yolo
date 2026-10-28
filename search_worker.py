"""One isolated YOLO preparation, training or evaluation stage for search_yolo.py."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    def scalar(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        raise TypeError(f'Unsupported JSON value: {type(obj).__name__}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=scalar), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
        self.encoding = getattr(terminal, 'encoding', 'utf-8')

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def execute(job):
    reporter = None
    if job['phase'] == 'train' and job.get('pruning'):
        from search_pruning import EpochReporter, PrunedRun
        marker = Path(job['trial_dir'])/'pruned.json'
        if marker.is_file():
            return json.loads(marker.read_text(encoding='utf-8'))
        reporter = EpochReporter(job)
        try:
            return execute_stage(job, reporter)
        except PrunedRun as exc:
            return exc.result
        finally:
            reporter.close()
    return execute_stage(job)


def execute_stage(job, reporter=None):
    phase = job['phase']
    if phase == 'prepare':
        from ultralytics.utils.downloads import attempt_download_asset
        from ultralytics import YOLO
        target = Path(job['target'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            source = Path(job['source'])
            if source.is_file():
                temporary = target.with_suffix('.tmp')
                shutil.copy2(source, temporary)
                temporary.replace(target)
            else:
                attempt_download_asset(str(target), unzip=False, retry=3)
        if not target.is_file() or target.stat().st_size < 100_000:
            raise ValueError(f'Invalid initial checkpoint: {target}')
        try:
            detector = YOLO(str(target))
            if detector.task != 'detect':
                raise ValueError('Checkpoint is not an object detector')
        except Exception as exc:
            raise ValueError(f'Initial checkpoint cannot be loaded: {target}. '
                             'It may be incomplete; replace it with a complete trusted checkpoint and rerun.') from exc
        return {'checkpoint': str(target), 'sha256': digest(target)}

    import torch
    import ultralytics
    from ultralytics import YOLO
    environment = {'python': sys.version, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__}
    if phase == 'train':
        directory = Path(job['train_dir'])
        last, best = directory/'weights/last.pt', directory/'weights/best.pt'
        started = time.monotonic()
        train_options = {}
        if reporter is not None:
            from search_pruning import validation_trainer
            train_options['trainer'] = validation_trainer(job)
        if last.is_file():
            model = YOLO(str(last))
            if reporter is not None:
                model.add_callback('on_fit_epoch_end', reporter)
            from training_diagnostics import install_phase_logging
            install_phase_logging(model)
            checkpoint = model.ckpt or {}
            if reporter is not None:
                reporter.recover(checkpoint.get('epoch', -1))
            # A completed run may have been killed just before stage-result JSON was saved.
            finished = checkpoint.get('epoch', 0) < 0 or checkpoint.get('epoch', -1) >= job['epochs']-1
            if not (finished and best.is_file()):
                model.train(resume=True, device=job['device'], workers=job['workers'], **train_options)
        else:
            model = YOLO(job['initial_checkpoint'])
            if reporter is not None:
                model.add_callback('on_fit_epoch_end', reporter)
            from training_diagnostics import install_phase_logging
            install_phase_logging(model)
            model.train(data=job['data'], epochs=job['epochs'], imgsz=job['imgsz'],
                        workers=job['workers'], device=job['device'], seed=job['seed'], deterministic=True,
                        patience=job['patience'], amp=job['amp'], cache=False, save=True, val=True,
                        project=str(directory.parent), name=directory.name, exist_ok=True,
                        close_mosaic=min(10, max(0, job['epochs']//5)), plots=True,
                        **train_options,
                        **{'batch': job['batch'], **job['params']})
        if not best.is_file():
            raise ValueError(f'Training did not produce {best}')
        rows = []
        results_csv = directory/'results.csv'
        if results_csv.is_file():
            with results_csv.open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
        return {'best_checkpoint': str(best), 'checkpoint_sha256': digest(best), 'last_checkpoint': str(last),
                'train_dir': str(directory), 'recorded_epochs': len(rows),
                'elapsed_seconds_this_attempt': time.monotonic()-started, 'environment': environment}

    if phase not in ('val', 'test'):
        raise ValueError(f'Unknown phase: {phase}')
    model = YOLO(job['checkpoint'])
    from training_diagnostics import install_phase_logging
    install_phase_logging(model)
    metrics = model.val(data=job['data'], split=phase, imgsz=job.get('eval_imgsz', job['imgsz']), batch=job['eval_batch'],
                        device=job['device'], workers=job['workers'],
                        project=str(Path(job['trial_dir'])/'evaluation'), name=phase, exist_ok=True,
                        conf=0.001, iou=0.7, max_det=300, plots=True, save_json=False)
    scores = {'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
              'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map)}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError(f'Invalid {phase} metrics: {scores}')
    # DetMetrics.summary() reports only classes with ground-truth annotations.
    per_class = metrics.summary() if hasattr(metrics, 'summary') else []
    return {'split': phase, 'metrics': scores, 'per_class': per_class,
            'speed_ms_per_image': {k: float(v) for k, v in metrics.speed.items()},
            'checkpoint_sha256': digest(job['checkpoint']), 'environment': environment}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    args = p.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    output = Path(job['result'])
    logfile = Path(job['log'])
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8', buffering=1) as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        print(f'\n--- Starting stage {job["phase"]} ---', flush=True)
        try:
            result = execute(job)
            write_json(output, {'status': 'ok', **result})
        except KeyboardInterrupt:
            raise SystemExit(130)
        except Exception as exc:
            traceback.print_exc()
            write_json(output, {'status': 'failed', 'error': str(exc), 'error_type': type(exc).__name__,
                                'errno': getattr(exc, 'errno', None)})
            raise SystemExit(1)


if __name__ == '__main__':
    main()
