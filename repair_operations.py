"""Durable journals for problem images only; preserve inode identity during repair."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def label_for(path):
    for parent in path.parents:
        if parent.name == 'images':
            root = parent.parent
            label = root/'labels'/path.relative_to(parent).with_suffix('.txt')
            if not label.resolve().is_relative_to(root.resolve()) or label.is_symlink():
                raise ValueError(f'Unsafe label path: {label}')
            return label
    return None


def journal_for(directory, path):
    return directory/(hashlib.sha256(str(path.absolute()).encode()).hexdigest()+'.json')


def finish(journal):
    op = json.loads(journal.read_text(encoding='utf-8'))
    if op.get('done'):
        if op.get('stage'):
            Path(op['stage']).unlink(missing_ok=True)
        return op['result']
    path = Path(op['path'])
    if op['action'] == 'delete':
        if path.exists():
            stat = path.stat()
            if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns] != op['identity']:
                raise ValueError(f'Image changed before deletion: {path}')
            # Remove the label first: a crash leaves an image to finish deleting,
            # rather than an orphan label that could be mistaken for valid data.
        label = Path(op['label']) if op.get('label') else None
        if label is not None:
            if label.exists() and label.read_bytes().hex() != op['label_bytes']:
                raise ValueError(f'Label changed before deletion: {label}')
            label.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    else:
        stage = Path(op['stage'])
        if hashlib.sha256(stage.read_bytes()).hexdigest() != op['stage_sha256']:
            raise ValueError(f'Repair recovery payload changed: {stage}')
        with path.open('r+b') as target, stage.open('rb') as source:
            stat = os.fstat(target.fileno())
            if [stat.st_dev, stat.st_ino] != op['identity'][:2]:
                raise ValueError(f'Image inode changed before repair: {path}')
            shutil.copyfileobj(source, target)
            target.truncate()
            target.flush()
            os.fsync(target.fileno())
    op['done'] = True
    atomic_json(journal, op)
    if op.get('stage'):
        Path(op['stage']).unlink(missing_ok=True)
    return op['result']


def mutate(path, directory, result, stage=None):
    directory.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    op = {'path': str(path), 'identity': [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
          'action': 'repair' if stage else 'delete', 'result': result, 'done': False}
    if stage:
        op.update(stage=str(stage), stage_sha256=hashlib.sha256(stage.read_bytes()).hexdigest())
    else:
        label = label_for(path)
        op.update(label=str(label) if label else None,
                  label_bytes=label.read_bytes().hex() if label and label.exists() else '')
        result['label_deleted'] = bool(label and label.exists())
        result['label_path'] = str(label) if label else None
    journal = journal_for(directory, path)
    atomic_json(journal, op)
    return finish(journal)


def recover(directory):
    from filelock import FileLock
    import tempfile
    for journal in sorted(directory.glob('*.json')):
        op = json.loads(journal.read_text(encoding='utf-8'))
        stripe = (op['identity'][0] ^ op['identity'][1]) % 64
        with FileLock(str(Path(tempfile.gettempdir())/f'yolo-image-repair-{stripe}.lock')):
            finish(journal)


def deleted_paths(directory):
    return [op['path'] for p in directory.glob('*.json')
            if (op := json.loads(p.read_text(encoding='utf-8')))['action'] == 'delete']
