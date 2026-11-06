"""Shared detection evaluation results, logs and lossless confusion-matrix exports."""
import contextlib
import logging
from pathlib import Path
import sys
import traceback

from search_worker import Tee, write_json


def save_confusion_matrix(matrix, path, split, checkpoint):
    import numpy as np
    raw = np.asarray(matrix.matrix, dtype=float)
    names = matrix.names
    labels = list(names) if isinstance(names, list) else [names[i] for i in sorted(names)]
    if raw.shape != (len(labels)+1, len(labels)+1) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Invalid detection confusion matrix dimensions/counts')
    labels.append('background')
    normalized = raw/(raw.sum(axis=0, keepdims=True)+1e-9)
    write_json(path, {'split':split, 'checkpoint':str(checkpoint),
        'labels':labels, 'class_ids':list(range(len(labels)-1))+[None],
        'row_axis':'predicted', 'column_axis':'true',
        'background_index':len(labels)-1, 'normalization':'per true-class column',
        'matrix':raw.tolist(), 'normalized_matrix':normalized.tolist(),
        'thresholds':'Ultralytics confusion-matrix defaults for this installed version; independent of AP IoU thresholds'})


def evaluate_logged(model, directory, split, checkpoint, **options):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/f'{split}_result.json'
    matrix_path = directory/f'{split}_confusion_matrix.json'
    # A previous success must not survive a failed retry.
    matrix_path.unlink(missing_ok=True)
    captured = {}
    def capture(validator):
        captured['matrix'] = validator.confusion_matrix
        captured['save_dir'] = str(validator.save_dir)
    model.add_callback('on_val_end', capture)
    logger = logging.getLogger('ultralytics')
    try:
        with (directory/f'{split}.log').open('a', encoding='utf-8', buffering=1) as log:
            handler = logging.StreamHandler(log)
            logger.addHandler(handler)
            try:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    print(f'Evaluating split={split}, checkpoint={checkpoint}', flush=True)
                    metrics = model.val(split=split, plots=True, **options)
                    cm = captured.get('matrix', getattr(metrics, 'confusion_matrix', None))
                    if cm is None:
                        raise ValueError('Evaluation did not provide a confusion matrix')
                    save_confusion_matrix(cm, matrix_path, split, checkpoint)
                    scores = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                                  map50=float(metrics.box.map50), map50_95=float(metrics.box.map))
                    import math
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
                        raise ValueError(f'Invalid metrics: {scores}')
                    result = {'status':'ok', 'split':split, 'checkpoint':str(checkpoint), 'metrics':scores,
                        'per_class':metrics.summary() if hasattr(metrics, 'summary') else [],
                        'speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},
                        'confusion_matrix_json':str(matrix_path),
                        'save_dir':captured.get('save_dir', str(getattr(metrics, 'save_dir', '')))}
                    write_json(result_path, result)
                    return result
            except BaseException as exc:
                traceback.print_exc(file=log)
                write_json(result_path, {'status':'failed', 'split':split, 'error':str(exc)})
                raise
            finally:
                logger.removeHandler(handler)
    finally:
        callbacks = getattr(model, 'callbacks', {})
        if isinstance(callbacks, dict) and capture in callbacks.get('on_val_end', []):
            callbacks['on_val_end'].remove(capture)
