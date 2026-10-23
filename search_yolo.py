"""Round-robin Optuna search of YOLOv8n/s and YOLO11n/s, with resumable isolated stages."""
from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import re
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from search_worker import digest, write_json
from search_report import write_report

MODELS = ('yolov8n.pt', 'yolov8s.pt', 'yolo11n.pt', 'yolo11s.pt')
BASELINE = {'batch': 64, 'optimizer': 'auto'}
SEARCH_ANCHOR = {'lr0': 0.001, 'weight_decay': 0.0005, 'mosaic': 1.0}
PARAM_FIELDS = ['batch', 'optimizer', 'lr0', 'weight_decay', 'mosaic']
SPACE_VERSION = 3
BATCH_CHOICES = [64, 128, 256]


def selected_models(args):
    requested = getattr(args, 'models', MODELS)
    # Canonical order makes equivalent selections share a resumable experiment.
    return tuple(model for model in MODELS if model in requested)


def trial_directory(root, model, number):
    return root/Path(model).stem.replace('yolov', 'v').replace('yolo', 'v')/f't{number:03d}'


def experiment_identity(args):
    return json.loads(json.dumps({
        'data': dataset_identity(args.data), 'weights_dir': str(args.weights_dir),
        'models': selected_models(args), 'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': args.batch,
        'patience': args.patience, 'seed': args.seed, 'amp': args.amp,
        'space_version': SPACE_VERSION, 'test_policy': args.test_policy,
        'metric': 'validation mAP50-95'}))


def automatic_output(args):
    identity = json.dumps(experiment_identity(args), sort_keys=True).encode()
    token = hashlib.sha256(identity).hexdigest()[:10]
    label = re.sub(r'[^a-z0-9]+', '-', args.data.parent.name.lower()).strip('-')[:20] or 'data'
    return Path('runs/tune')/f'{label}-{token}'


class JobError(RuntimeError):
    def __init__(self, phase, code, result):
        self.phase, self.code, self.result = phase, code, result
        super().__init__(f'{phase} failed (exit={code}): {result.get("error", "No result; inspect stage log")}')

    @property
    def resource_failure(self):
        return self.code in (137, -9) or self.result.get('errno') in (
            errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EIO, errno.EACCES) or (
            'out of memory' in self.result.get('error', '').lower()) or self.result.get('error_type') in (
                'MemoryError', 'OutOfMemoryError')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def sample_params(trial, fixed_batch=None):
    # Compact space: optimizer/schedule/color augmentation retain stable choices.
    return {'batch': trial.suggest_categorical('batch', [fixed_batch] if fixed_batch is not None else BATCH_CHOICES),
            'optimizer': 'AdamW',
            'lr0': trial.suggest_float('lr0', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'mosaic': trial.suggest_float('mosaic', 0.0, 1.0)}


def run_job(job):
    result_path = Path(job['result'])
    old = read_json(result_path, {})
    if old.get('status') == 'ok':
        if job['phase'] == 'train':
            checkpoint = Path(old['best_checkpoint'])
            if not checkpoint.is_file() or digest(checkpoint) != old['checkpoint_sha256']:
                raise ValueError(f'Completed checkpoint missing/changed: {checkpoint}')
        if job['phase'] in ('val', 'test') and old.get('checkpoint_sha256') != digest(job['checkpoint']):
            raise ValueError(f'Evaluation checkpoint changed: {job["checkpoint"]}')
        return old
    job_path = result_path.with_name(job['phase']+'_job.json')
    write_json(job_path, job)
    command = [sys.executable, str(Path(__file__).with_name('search_worker.py')), '--job', str(job_path)]
    print(f'[{job["phase"]}] {job.get("model", "weights")} -> {result_path.parent}', flush=True)
    if sys.platform == 'linux':
        from training_guard import supervise
        code = supervise(command)
    else:
        child = subprocess.Popen(command)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    result = read_json(result_path, {})
    if code in (130, 143, -2, -15):
        raise KeyboardInterrupt
    if code or result.get('status') != 'ok':
        raise JobError(job['phase'], code, result)
    return result


def choose_train_dir(directory):
    previous = read_json(directory/'train_job.json', {})
    if previous:
        run = Path(previous['train_dir'])
        if (run/'weights/last.pt').is_file():
            return run
    run, index = directory/'training', 0
    while run.exists():
        index += 1
        run = directory/f'training_attempt_{index:03d}'
    return run


def evaluate(config, training, phase, args):
    directory = Path(config['trial_dir'])
    return run_job({**config, 'phase': phase, 'checkpoint': training['best_checkpoint'],
                    'eval_batch': args.eval_batch, 'device': args.device, 'workers': args.workers,
                    'result': str(directory/f'{phase}_result.json'), 'log': str(directory/f'{phase}.log')})


def train_and_validate(config, args):
    directory = Path(config['trial_dir'])
    training = run_job({**config, 'phase': 'train', 'train_dir': str(choose_train_dir(directory)),
                        'device': args.device, 'workers': args.workers,
                        'result': str(directory/'train_result.json'), 'log': str(directory/'train.log')})
    validation = evaluate(config, training, 'val', args)
    return training, validation


def collect_records(root, studies):
    records = []
    for model in studies:
        for trial in studies[model].get_trials(deepcopy=False):
            if trial.state.name == 'WAITING':
                continue
            directory = trial_directory(root, model, trial.number)
            records.append({'model': model, 'size': Path(model).stem[-1], 'trial': trial.number,
                            'state': trial.state.name, 'objective': trial.value,
                            'params': read_json(directory/'config.json', {}).get('params', trial.params), 'trial_dir': str(directory),
                            'config': read_json(directory/'config.json', {}),
                            'train': read_json(directory/'train_result.json', {}),
                            'val': read_json(directory/'val_result.json', {}),
                            'test': read_json(directory/'test_result.json', {}),
                            'status': read_json(directory/'status.json', {})})
    return sorted(records, key=lambda r: (r['trial'], MODELS.index(r['model'])))


def select_winners(records):
    winners = {}
    for size in ('n', 's'):
        candidates = [r for r in records if r['size'] == size and r['state'] == 'COMPLETE'
                      and r['val'].get('status') == 'ok' and r['train'].get('status') == 'ok']
        if candidates:
            # No test metric is read for model selection or tie-breaking.
            winners[size] = max(candidates, key=lambda r: r['val']['metrics']['map50_95'])
    return winners


def refresh_reports(root, studies):
    records = collect_records(root, studies)
    winners = select_winners(records)
    write_json(root/'summary.json', {'selection_metric': 'validation mAP50-95',
                                     'winners': winners, 'iterations': records})
    fields = ['round', 'model', 'size', 'trial', 'state', 'val_map50_95', 'val_map50', 'val_precision',
              'val_recall', 'test_map50_95', 'test_map50', 'test_precision', 'test_recall',
              'test_status', 'recorded_epochs', 'train_seconds_this_attempt', 'test_inference_ms',
              'best_checkpoint', 'error', *PARAM_FIELDS]
    temporary = root/'iterations.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {'round': r['trial']+1, 'model': r['model'], 'size': r['size'], 'trial': r['trial'],
                   'state': r['state'], 'test_status': r['test'].get('status', 'pending'),
                   'recorded_epochs': r['train'].get('recorded_epochs', ''),
                   'train_seconds_this_attempt': r['train'].get('elapsed_seconds_this_attempt', ''),
                   'test_inference_ms': r['test'].get('speed_ms_per_image', {}).get('inference', ''),
                   'best_checkpoint': r['train'].get('best_checkpoint', ''),
                   'error': r['status'].get('error', ''), **r['params']}
            for split in ('val', 'test'):
                for key, value in r[split].get('metrics', {}).items():
                    row[split+'_'+key] = value
            writer.writerow(row)
    temporary.replace(root/'iterations.csv')
    for number in sorted({r['trial'] for r in records}):
        write_json(root/'rounds'/f'round_{number+1:04d}.json', [r for r in records if r['trial'] == number])
    best_dir = root/'best'
    best_dir.mkdir(exist_ok=True)
    for size, winner in winners.items():
        source = Path(winner['train']['best_checkpoint'])
        target = best_dir/f'best_{size}.pt'
        expected = winner['train']['checkpoint_sha256']
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f'Selected checkpoint missing or changed: {source}')
        if not target.is_file() or digest(target) != expected:
            temporary = target.with_suffix('.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(target)
    write_json(best_dir/'selection.json', {'selection_metric': 'validation mAP50-95', 'models': winners})
    write_report(root, records, winners, tuple(studies))
    return records, winners


def dataset_identity(data):
    import yaml
    config = yaml.safe_load(data.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or not all(config.get(k) for k in ('train', 'val', 'test', 'names')):
        raise ValueError('Dataset YAML must define non-empty train, val, test and names')
    if any(config[a] == config[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Train, validation and test entries must be distinct')
    report = data.parent/'report.json'
    return {'path': str(data), 'sha256': digest(data),
            'converter_report_sha256': digest(report) if report.is_file() else None}


def run(args):
    import optuna
    root = args.output.resolve()
    identity = experiment_identity(args)
    manifest = root/'run.json'
    existing = read_json(manifest)
    if existing is not None and existing != identity:
        raise ValueError('Search configuration/data changed; use a new --output directory. '
                         'Only trial target and resource options (device/workers/eval-batch) may change on resume.')
    if existing is None and any(p.name != '.search.lock' for p in root.iterdir()):
        raise ValueError('Output is not an existing compatible search or an empty directory')
    write_json(manifest, identity)
    initial = {}
    for model in selected_models(args):
        result = run_job({'phase': 'prepare', 'model': model,
                          'source': str(args.weights_dir/model), 'target': str(root/'initial_weights'/model),
                          'result': str(root/'initial_weights'/(model+'.json')),
                          'log': str(root/'initial_weights'/(model+'.log'))})
        if digest(result['checkpoint']) != result['sha256']:
            raise ValueError(f'Initial checkpoint changed: {model}')
        initial[model] = result['checkpoint']
    storage = optuna.storages.RDBStorage(url='sqlite:///'+(root/'optuna.sqlite3').as_posix(),
                                        engine_kwargs={'connect_args': {'timeout': 60}})
    try:
        return search_studies(args, root, initial, storage)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def search_studies(args, root, initial, storage):
    import optuna
    from optuna.trial import TrialState
    studies = {}
    for model in selected_models(args):
        index = MODELS.index(model)
        study = optuna.create_study(study_name=Path(model).stem, direction='maximize',
                                   storage=storage, load_if_exists=True,
                                   sampler=optuna.samplers.TPESampler(seed=args.seed+index, n_startup_trials=3),
                                   pruner=optuna.pruners.NopPruner())
        if not study.get_trials():
            study.enqueue_trial({})  # Official auto-optimizer baseline; no lr override.
            for batch in ([args.batch] if args.batch is not None else BATCH_CHOICES):
                study.enqueue_trial({**SEARCH_ANCHOR, 'batch': batch})
        studies[model] = study
    refresh_reports(root, studies)

    # Retry evaluations of already-completed training without repeating the training stage.
    if args.test_policy == 'every-trial':
        try:
            for record in collect_records(root, studies):
                if record['state'] == 'COMPLETE':
                    evaluate(record['config'], record['train'], 'test', args)
        finally:
            refresh_reports(root, studies)

    while True:
        did_work = False
        for model, study in studies.items():
            trials = study.get_trials(deepcopy=False)
            running = [t for t in trials if t.state == TrialState.RUNNING]
            attempted = [t for t in trials if t.state != TrialState.WAITING]
            if running:
                candidate = running[0]
                directory = trial_directory(root, model, candidate.number)
                config = read_json(directory/'config.json')
                if not config:
                    study.tell(candidate.number, state=TrialState.FAIL)
                    write_json(directory/'status.json', {'state': 'FAIL', 'error': 'Interrupted before trial config was committed'})
                    did_work = True
                    continue
                number = candidate.number
            elif len(attempted) < args.trials:
                queued = [t.number for t in trials if t.state == TrialState.WAITING]
                next_number = min(queued) if queued else max((t.number for t in trials), default=-1)+1
                # Seed per candidate rather than per process, so restarting the coordinator
                # cannot replay the same random startup suggestions on every invocation.
                study.sampler = optuna.samplers.TPESampler(
                    seed=(args.seed+MODELS.index(model)*1000003+next_number*1009) % (2**32),
                    n_startup_trials=3)
                candidate = study.ask()
                number = candidate.number
                directory = trial_directory(root, model, number)
                config = {**{k: getattr(args, k) for k in ('epochs', 'imgsz', 'batch', 'patience', 'seed', 'amp')},
                          'data': str(args.data), 'model': model, 'trial': number, 'trial_dir': str(directory),
                          'initial_checkpoint': initial[model], 'params': ({**BASELINE, 'batch': args.batch or BASELINE['batch']} if number == 0
                                     else sample_params(candidate, args.batch))}
                config['batch'] = config['params']['batch']
                write_json(directory/'config.json', config)
            else:
                continue
            did_work = True
            print(f'\nRound {number+1}/{args.trials}: {model}; params={config["params"]}', flush=True)
            try:
                write_json(directory/'status.json', {'state': 'RUNNING', 'started_at': time.time()})
                training, validation = train_and_validate(config, args)
                study.tell(number, validation['metrics']['map50_95'])
                write_json(directory/'status.json', {'state': 'COMPLETE', 'finished_at': time.time()})
                refresh_reports(root, studies)
                if args.test_policy == 'every-trial':
                    evaluate(config, training, 'test', args)
            except KeyboardInterrupt:
                write_json(directory/'status.json', {'state': 'INTERRUPTED', 'error': 'Rerun the same command to continue'})
                raise
            except JobError as exc:
                # Preserve RUNNING for recoverable resource failures, so last.pt can resume.
                state = next(t.state for t in study.get_trials() if t.number == number)
                if state == TrialState.RUNNING and not exc.resource_failure:
                    study.tell(number, state=TrialState.FAIL)
                write_json(directory/'status.json', {'state': 'FAILED_STAGE', 'error': str(exc)})
                print(str(exc), file=sys.stderr, flush=True)
                if exc.resource_failure:
                    raise
            finally:
                refresh_reports(root, studies)
        if not did_work:
            break
    records, winners = refresh_reports(root, studies)
    if args.test_policy == 'final':
        try:
            for record in winners.values():
                evaluate(record['config'], record['train'], 'test', args)
        finally:
            records, winners = refresh_reports(root, studies)
    print(f'\nResults: {root/"iterations.csv"}', flush=True)
    for size, record in winners.items():
        print(f'Best {size}: {record["model"]} trial {record["trial"]}; '
              f'val mAP50-95={record["val"]["metrics"]["map50_95"]:.5f}; '
              f'test={record["test"].get("metrics")}; weights={root/"best"/f"best_{size}.pt"}', flush=True)
    if set(winners) != {Path(model).stem[-1] for model in studies} or any(r['state'] != 'COMPLETE' for r in records):
        raise ValueError('Some trials failed or a selected model size has no complete candidate; inspect iterations.csv and stage logs')
    if args.test_policy == 'every-trial' and any(r['test'].get('status') != 'ok' for r in records if r['state'] == 'COMPLETE'):
        raise ValueError('Some test stages failed; rerun the same command to retry evaluation only')
    write_report(root, records, winners, tuple(studies), complete=True)
    print(f'Human-readable report: {root / "REPORT.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS),
                   help='Train only the selected models (default: all four)')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, default=None, help='Optional; otherwise derive a short resumable experiment name')
    p.add_argument('--weights-dir', type=Path, default=Path('weights'))
    p.add_argument('--trials', type=int, default=8, help='Target total trial attempts per model, including previous attempts')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=None,
                   help='Fix training batch instead of searching the default choices 64/128/256')
    p.add_argument('--eval-batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='0', help='One GPU index, or cpu; this search is serial')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--test-policy', choices=['every-trial', 'final'], default='final')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    args.data = args.data.expanduser().resolve()
    args.weights_dir = args.weights_dir.expanduser().resolve()
    if any(v < 1 for v in (args.trials, args.epochs, args.imgsz, args.eval_batch)) or (args.batch is not None and args.batch < 1) or args.workers < 0 or args.patience < 0:
        p.error('Trials/epochs/imgsz/batches must be positive; workers/patience must be non-negative')
    if not (args.device == 'cpu' or args.device.isdecimal()):
        p.error('--device must be one GPU index or cpu')
    try:
        dataset_identity(args.data)
        args.output = (args.output or automatic_output(args)).expanduser().resolve()
        if args.dry_run:
            print(json.dumps({'models_in_order': selected_models(args), 'rounds': args.trials,
                              'total_trials': len(selected_models(args))*args.trials, 'max_epochs_per_trial': args.epochs,
                              'batch_choices': [args.batch] if args.batch is not None else BATCH_CHOICES,
                              'eval_batch': args.eval_batch, 'workers': args.workers,
                              'selection_metric': 'validation mAP50-95', 'test_policy': args.test_policy,
                              'output': str(args.output), 'baseline': {**BASELINE, 'batch': args.batch if args.batch is not None else BASELINE['batch']}}, indent=2))
            return
        from filelock import FileLock, Timeout
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(args.output/'.search.lock'), timeout=0):
                run(args)
        except Timeout as exc:
            raise ValueError('Another search is using this output directory') from exc
    except KeyboardInterrupt:
        p.exit(130, '\nSearch interrupted. Completed stages and checkpoints are retained; rerun the same command.\n')
    except ImportError as exc:
        p.exit(1, f'Missing dependency: {exc}. Install: pip install -r requirements-search.txt\n')
    except Exception as exc:
        p.exit(1, f'Search stopped ({type(exc).__name__}): {exc}\nArtifacts are retained in {args.output}\n')


if __name__ == '__main__':
    main()
