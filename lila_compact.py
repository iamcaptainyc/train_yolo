"""Compact batch checkpoints: no persistent rows, labels, dimensions or hashes for good images."""

from collections import Counter
from contextlib import closing
from PIL import __version__ as PIL_VERSION
from lila_parallel import bounded_map
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid

from lila_state import (
    RunState,
    atomic_text,
    canonical,
    fingerprint,
    label_text,
    record_key,
    relative_paths,
    signature,
)


class CompactState(RunState):
    compact = True
    batch_size = 128

    def __init__(self, output, force=False):
        self.output = Path(output).resolve()
        self.directory = self.output / '.lila-state'
        self.force = force
        self.lock = threading.RLock()
        self.stats = Counter()
        self.run_id = uuid.uuid4().hex
        # The inherited error export uses SQLite, but only in RAM in compact mode.
        self.db = sqlite3.connect(
            ':memory:', isolation_level=None, check_same_thread=False
        )
        self.db.execute(
            'CREATE TABLE errors (stage TEXT,key TEXT,payload TEXT,PRIMARY KEY(stage,key))'
        )
        self.db.execute(
            'CREATE TABLE outputs (key TEXT PRIMARY KEY,image TEXT,label TEXT,signature TEXT,stamp TEXT,mode TEXT)'
        )
        for stage in ('validation', 'write', 'audit'):
            path = self.output / (stage + '_errors.jsonl')
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    key = row['dataset'] + '/' + row['file_name']
                    self.db.execute(
                        'INSERT OR REPLACE INTO errors VALUES (?,?,?)',
                        (stage, key, canonical(row)),
                    )
        path = self.directory / 'progress.json'
        self.progress = (
            json.loads(path.read_text(encoding='utf-8'))
            if path.exists()
            else {'schema': 1, 'batch_size': self.batch_size, 'stages': {}}
        )
        if (
            self.progress.get('schema') != 1
            or self.progress.get('batch_size') != self.batch_size
        ):
            raise ValueError('Unsupported compact checkpoint version')
        self.phases = {}
        self.paths = {}

    def configure(self, config):
        # Preserve the full-mode configuration guard without storing an output registry.
        path = self.directory / 'run.json'
        old = (
            json.loads(path.read_text(encoding='utf-8'))
            if path.exists()
            else {'config': {}}
        )
        changed = [
            k for k, v in config.items() if k in old['config'] and old['config'][k] != v
        ]
        if changed and (
            'source' in changed or any((self.output / 'images').rglob('*.*'))
        ):
            raise ValueError(
                f'Conversion configuration changed ({changed}); use a new output directory'
            )
        self.config = {**old['config'], **config}
        self.status('RUNNING')
        # These are explicitly owned artifacts from the previous detailed mode.
        for relative in (
            'box_audit.jsonl',
            'split_manifest.jsonl',
        ):
            (self.output / relative).unlink(missing_ok=True)

    def _stamp(self, records, stage):
        rows = []
        for r in records:
            try:
                info = fingerprint(r['source'])
                # Hardlinks change ctime; mtime, size and file identity remain stable.
                info.pop('ctime_ns')
            except OSError:
                info = None
            row = [record_key(r), info, r['image'], r['boxes']]
            if stage == 'audit':
                files = []
                for relative in relative_paths(r):
                    try:
                        info = fingerprint(self.safe_path(relative))
                        info.pop('ctime_ns')
                    except OSError:
                        info = None
                    files.append(info)
                row += [
                    files,
                    r['digest'],
                    r['labels'],
                    r['split'],
                    r['source_annotations'],
                ]
            rows.append(row)
        return signature(['compact-batch-v1', PIL_VERSION, self.config, rows])

    def prepare_batches(self, records, stage='validation'):
        legacy = None
        legacy_path = self.directory / 'cache.sqlite3'
        if stage == 'validation' and legacy_path.exists() and not self.force:
            try:
                legacy = sqlite3.connect(str(legacy_path))
                legacy.execute('SELECT 1 FROM cache LIMIT 1')
            except sqlite3.DatabaseError:
                if legacy:
                    legacy.close()
                legacy = None
        old = self.progress['stages'].get(stage, {})
        phase = self.phases[stage] = {'batches': [], 'lookup': {}}
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            index = str(start // self.batch_size)
            stamp = self._stamp(batch, stage)
            item = {
                'records': batch,
                'stamp': stamp,
                'index': index,
                'remaining': len(batch),
                'passed': not self.force and old.get(index) == stamp,
            }
            if legacy and not item['passed']:
                inherited = True
                for record in batch:
                    row = legacy.execute(
                        'SELECT stamp,payload FROM cache WHERE stage=? AND key=? AND status=?',
                        ('image', str(record['source']), 'ok'),
                    ).fetchone()
                    try:
                        previous = json.loads(row[0]) if row else {}
                        payload = json.loads(row[1]) if row else {}
                        matches = (
                            previous.get('file') == fingerprint(record['source'])
                            and previous.get('version')
                            == 'decode-exif-verify-sha256-v1:' + PIL_VERSION
                            and all(
                                record['image'].get(dimension) is None
                                or record['image'][dimension] == payload.get(dimension)
                                for dimension in ('width', 'height')
                            )
                        )
                    except OSError:
                        matches = False
                    known_error = self.db.execute(
                        'SELECT 1 FROM errors WHERE stage=? AND key=?',
                        ('validation', record_key(record)),
                    ).fetchone()
                    if not matches and not known_error:
                        inherited = False
                        break
                item['passed'] = inherited
            phase['batches'].append(item)
            for r in batch:
                phase['lookup'][record_key(r)] = item
                self.paths[str(r['source'])] = r
        self.progress['stages'][stage] = {
            item['index']: item['stamp'] for item in phase['batches'] if item['passed']
        }
        if legacy:
            legacy.close()
        atomic_text(self.directory / 'progress.json', canonical(self.progress))
        if stage == 'validation':
            for suffix in ('', '-wal', '-shm'):
                (self.directory / ('cache.sqlite3' + suffix)).unlink(missing_ok=True)

    def _passed(self, record, stage):
        item = self.phases[stage]['lookup'][record_key(record)]
        with self.lock:
            error = self.db.execute(
                'SELECT 1 FROM errors WHERE stage=? AND key=?',
                (stage, record_key(record)),
            ).fetchone()
        return item['passed'] and not error

    def validation_done(self, record):
        self._done(record, 'validation')

    def audit_done(self, record):
        self._done(record, 'audit')

    def _done(self, record, stage):
        with self.lock:
            item = self.phases[stage]['lookup'][record_key(record)]
            item['remaining'] -= 1
            if (
                item['remaining'] == 0
                and self.progress['stages'][stage].get(item['index']) != item['stamp']
                and self._stamp(item['records'], stage) == item['stamp']
            ):
                self.progress['stages'][stage][item['index']] = item['stamp']
                atomic_text(self.directory / 'progress.json', canonical(self.progress))

    def map_tasks(self, function, records, workers, progress=None):
        # Finish one batch before starting another: interruption repeats at most one batch.
        for start in range(0, len(records), self.batch_size):
            with closing(
                bounded_map(
                    function,
                    records[start : start + self.batch_size],
                    workers,
                    progress,
                )
            ) as results:
                yield from results

    def get(self, stage, key, stamp):
        if stage != 'image':
            return None
        record = self.paths[key]
        if not self._passed(record, 'validation'):
            with self.lock:
                self.stats['image_checked'] += 1
            return None
        # Rebuild transient SHA values for duplicate grouping, never save them.
        width, height = record['image'].get('width'), record['image'].get('height')
        if width is None or height is None:
            from PIL import Image

            with Image.open(record['source']) as image:
                width, height = image.size
        with record['source'].open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if fingerprint(record['source']) != stamp['file']:
            raise ValueError('Source changed while rebuilding grouping hashes')
        with self.lock:
            self.stats['image_decode_skipped'] += 1
            self.stats['grouping_hashes_rebuilt'] += 1
        return {'width': width, 'height': height, 'digest': digest}

    def put(self, stage, key, stamp, payload):
        pass  # Normal per-image results are intentionally never persisted.

    def error(self, stage, record, error):
        with self.lock:
            row = {
                'stage': stage,
                'dataset': record['dataset'],
                'file_name': record['name'],
                'source': str(record['source']),
                'error': str(error),
            }
            self.db.execute(
                'INSERT OR REPLACE INTO errors VALUES (?,?,?)',
                (stage, record_key(record), canonical(row)),
            )
            self.export_errors(stage)

    def clear_error(self, stage, record):
        with self.lock:
            result = self.db.execute(
                'DELETE FROM errors WHERE stage=? AND key=?',
                (stage, record_key(record)),
            )
            if result.rowcount:
                self.export_errors(stage)

    def output_hit(self, record, expected):
        image_rel, label_rel = relative_paths(record)
        image, label = self.safe_path(image_rel), self.safe_path(label_rel)
        try:
            if label.read_text(encoding='utf-8') != label_text(record):
                return None
            mode = 'hardlink' if os.path.samefile(image, record['source']) else 'copy'
            if mode == 'copy':
                if getattr(self, 'reclassifying', False):
                    return None
                with image.open('rb') as stream:
                    if (
                        hashlib.file_digest(stream, 'sha256').hexdigest()
                        != record['digest']
                    ):
                        return None
            self.stats['write_hits'] += 1
            return mode
        except OSError:
            return None

    def begin_output(self, record, expected):
        pass

    def save_output(self, record, expected, mode):
        pass

    def refresh_output_stamp(self, record):
        pass

    def reconcile(self, records):
        expected = {relative for r in records for relative in relative_paths(r)}
        for kind in ('images', 'labels'):
            folder = self.output / kind
            for path in folder.rglob('*'):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.output).as_posix()
                if kind == 'labels' and path.suffix == '.cache':
                    # Ultralytics creates these derived caches during training.
                    self.safe_path(relative).unlink()
                    continue
                # This dedicated output is marked as managed; only remove generated names.
                if not re.fullmatch(
                    r'[0-9a-f]{64}\.[a-z0-9]+(?:\.lila-part)?', path.name
                ):
                    raise ValueError(f'Unexpected file in managed output: {path}')
                if relative not in expected:
                    self.safe_path(relative).unlink()
