"""Persistent, locked checkpoints for LILA conversion and per-file audits."""

from contextlib import contextmanager
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from filelock import FileLock

SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fingerprint(path):
    info = Path(path).stat()
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def same_content_attributes(a, b):
    return all(a[k] == b[k] for k in ("size", "mtime_ns", "device", "inode"))


def record_key(record):
    return record["dataset"] + "/" + record["name"]


def relative_paths(record):
    key = hashlib.sha256(record_key(record).encode()).hexdigest()
    image = (
        Path("images")
        / record["split"]
        / record["dataset"]
        / (key + record["source"].suffix.lower())
    )
    label = Path("labels") / record["split"] / record["dataset"] / (key + ".txt")
    return image.as_posix(), label.as_posix()


def label_text(record):
    return "".join(
        str(b[0]) + " " + " ".join(f"{v:.8f}" for v in b[1:]) + "\n"
        for b in record["labels"]
    )


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ConversionReport(dict):
    """JSON contains dict items only; the live state is an internal runtime attribute."""

    state = None


class RunState:
    def __init__(self, output, force=False):
        self.output = Path(output).resolve()
        self.directory = self.output / ".lila-state"
        self.force = force
        self.lock = threading.RLock()
        self.stats = Counter()
        self.run_id = uuid.uuid4().hex
        self.db = sqlite3.connect(
            str(self.directory / "cache.sqlite3"),
            isolation_level=None,
            check_same_thread=False,
            timeout=60,
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (stage TEXT,key TEXT,stamp TEXT,status TEXT,payload TEXT,PRIMARY KEY(stage,key))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outputs (key TEXT PRIMARY KEY,image TEXT,label TEXT,signature TEXT,stamp TEXT,mode TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS errors (stage TEXT,key TEXT,payload TEXT,PRIMARY KEY(stage,key))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS error_history (run_id TEXT,stage TEXT,key TEXT,payload TEXT)"
        )

    def configure(self, config):
        path = self.directory / "run.json"
        old = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"schema": SCHEMA, "config": {}}
        )
        if old["schema"] != SCHEMA:
            raise ValueError(
                "Unsupported conversion checkpoint version; use a new output directory"
            )
        changed = [
            key
            for key, value in config.items()
            if key in old["config"] and old["config"][key] != value
        ]
        if changed and (
            self.db.execute("SELECT count(*) FROM outputs").fetchone()[0]
            or "source" in changed
        ):
            raise ValueError(
                f"Conversion configuration changed ({changed}); use a new output directory to avoid mixing splits/classes"
            )
        self.config = {**old["config"], **config}
        self.status("RUNNING")

    def status(self, status, error=None):
        atomic_text(
            self.directory / "run.json",
            json.dumps(
                {
                    "schema": SCHEMA,
                    "config": self.config,
                    "status": status,
                    "run_id": self.run_id,
                    "updated_at": time.time(),
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def get(self, stage, key, stamp):
        with self.lock:
            row = (
                None
                if self.force
                else self.db.execute(
                    "SELECT stamp,status,payload FROM cache WHERE stage=? AND key=?",
                    (stage, key),
                ).fetchone()
            )
            if row and row[0] == canonical(stamp) and row[1] == "ok":
                self.stats[stage + "_hits"] += 1
                return json.loads(row[2])
            self.stats[stage + "_checked"] += 1
        return None

    def put(self, stage, key, stamp, payload):
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (stage, key, canonical(stamp), "ok", canonical(payload)),
            )

    def invalidate(self, stage, key):
        with self.lock:
            self.db.execute("DELETE FROM cache WHERE stage=? AND key=?", (stage, key))

    def error(self, stage, record, error):
        key = record_key(record)
        payload = canonical(
            {
                "stage": stage,
                "dataset": record["dataset"],
                "file_name": record["name"],
                "source": str(record["source"]),
                "error": str(error),
                "run_id": self.run_id,
                "time": time.time(),
            }
        )
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO errors VALUES (?,?,?)", (stage, key, payload)
            )
            self.db.execute(
                "INSERT INTO error_history VALUES (?,?,?,?)",
                (self.run_id, stage, key, payload),
            )

    def clear_error(self, stage, record):
        with self.lock:
            self.db.execute(
                "DELETE FROM errors WHERE stage=? AND key=?",
                (stage, record_key(record)),
            )

    def export_errors(self, stage, records=None):
        active = (
            {record_key(r) for r in records}
            if records is not None
            else getattr(self, "active_keys", None)
        )
        with self.lock:
            rows = self.db.execute(
                "SELECT key,payload FROM errors WHERE stage=? ORDER BY key", (stage,)
            ).fetchall()
        lines = [payload for key, payload in rows if active is None or key in active]
        atomic_text(
            self.output / (stage + "_errors.jsonl"),
            "".join(line + "\n" for line in lines),
        )
        return len(lines)

    def safe_path(self, relative):
        path = Path(os.path.abspath(self.output / relative))
        if self.output not in path.parents or Path(relative).parts[0] not in (
            "images",
            "labels",
        ):
            raise ValueError(
                f"Checkpoint output path escapes managed image/label directories: {relative}"
            )
        return path

    def reconcile(self, records):
        expected = {record_key(r): relative_paths(r) for r in records}
        with self.lock:
            rows = self.db.execute("SELECT key,image,label FROM outputs").fetchall()
        for key, image, label in rows:
            if expected.get(key) != (image, label):
                self.safe_path(image).unlink(missing_ok=True)
                self.safe_path(label).unlink(missing_ok=True)
                self.safe_path(image + ".lila-part").unlink(missing_ok=True)
                self.safe_path(label + ".lila-part").unlink(missing_ok=True)
                with self.lock:
                    self.db.execute("DELETE FROM outputs WHERE key=?", (key,))
                    self.db.execute(
                        "DELETE FROM cache WHERE stage=? AND key=?", ("audit", key)
                    )
        # Handle the two bounded temporary names used by interrupted atomic file writes.
        for r in records:
            for relative in relative_paths(r):
                self.safe_path(relative + ".lila-part").unlink(missing_ok=True)

    def output_hit(self, record, expected):
        with self.lock:
            row = self.db.execute(
                "SELECT image,label,signature,stamp,mode FROM outputs WHERE key=?",
                (record_key(record),),
            ).fetchone()
        if row and row[2] == expected:
            try:
                current = {
                    "image": fingerprint(self.safe_path(row[0])),
                    "label": fingerprint(self.safe_path(row[1])),
                }
            except OSError:
                return None
            if canonical(current) == row[3]:
                with self.lock:
                    self.stats["write_hits"] += 1
                return row[4]
        with self.lock:
            self.stats["write_checked"] += 1
        return None

    def begin_output(self, record, expected):
        image, label = relative_paths(record)
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?,?)",
                (record_key(record), image, label, expected, "", "pending"),
            )

    def save_output(self, record, expected, mode):
        image, label = relative_paths(record)
        stamp = {
            "image": fingerprint(self.safe_path(image)),
            "label": fingerprint(self.safe_path(label)),
        }
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?,?)",
                (record_key(record), image, label, expected, canonical(stamp), mode),
            )

    def refresh_source(self, record):
        current = fingerprint(record["source"])
        if not same_content_attributes(record["validation_fingerprint"], current):
            raise ValueError(f'Source changed after validation: {record["source"]}')
        # os.link/replace changes ctime without changing bytes. A full audit still checks the digest.
        self.put(
            "image",
            str(record["source"]),
            {"file": current, "version": record["image_cache_version"]},
            {
                "width": record["actual_dimensions"][0],
                "height": record["actual_dimensions"][1],
                "digest": record["digest"],
            },
        )
        record["validation_fingerprint"] = current

    def refresh_output_stamp(self, record):
        image, label = relative_paths(record)
        stamp = {
            "image": fingerprint(self.safe_path(image)),
            "label": fingerprint(self.safe_path(label)),
        }
        with self.lock:
            self.db.execute(
                "UPDATE outputs SET stamp=? WHERE key=?",
                (canonical(stamp), record_key(record)),
            )

    def close(self):
        self.db.close()


@contextmanager
def managed_run(source, output, config, force=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must not contain each other")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / ("." + output.name + ".lila.lock")), timeout=0):
        marker = output / ".lila-state/run.json"
        if output.exists() and any(output.iterdir()) and not marker.is_file():
            raise ValueError(
                "Non-empty output has no managed checkpoint; use a new output directory"
            )
        (output / ".lila-state").mkdir(parents=True, exist_ok=True)
        state = RunState(output, force)
        try:
            state.configure({"source": str(source), **config})
            # A partial/failed resumed attempt must never leave an old training certificate visible.
            (output / "data.yaml").unlink(missing_ok=True)
            (output / "AUDIT_REPORT.md").unlink(missing_ok=True)
            yield state
            (output / "PROCESSING_REPORT.md").unlink(missing_ok=True)
            state.status("COMPLETE")
        except BaseException as exc:
            if hasattr(state, "config"):
                (output / "data.yaml").unlink(missing_ok=True)
            if hasattr(state, "config"):
                state.status(
                    "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                    str(exc),
                )
            if hasattr(state, 'config') and not isinstance(exc, KeyboardInterrupt):
                atomic_text(
                    output / 'PROCESSING_REPORT.md',
                    '# Conversion could not produce a training dataset\n\n'
                    + str(exc)
                    + '\n\nNo training certificate was issued. Source images are preserved. '
                    'Check validation_errors.jsonl, write_errors.jsonl, audit_errors.jsonl and .lila-state/run.json.\n',
                )
            raise
        finally:
            for stage in ('validation', 'write', 'audit'):
                state.export_errors(stage)
            state.close()
