"""Linux process-group supervision for training workers and their DataLoaders."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GUARD_ENV = 'TRAIN_YOLO_SUPERVISOR_PID'
KILL_SIGNAL = getattr(signal, 'SIGKILL', 9)


def signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def reap_group(pgid):
    """Reap only adopted descendants from this run, never unrelated children."""
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def cleanup_group(pgid, grace=3.0):
    if not signal_group(pgid, signal.SIGTERM):
        reap_group(pgid)
        return
    deadline = time.monotonic()+grace
    while time.monotonic() < deadline:
        reap_group(pgid)
        if not signal_group(pgid, 0):
            return
        time.sleep(0.05)
    signal_group(pgid, KILL_SIGNAL)
    # Children may need a moment to exit and be adopted after the signal.
    deadline = time.monotonic()+1.0
    while time.monotonic() < deadline:
        reap_group(pgid)
        if not signal_group(pgid, 0):
            return
        time.sleep(0.05)
    print(f'[Guard] Process group {pgid} still exists after cleanup; check for uninterruptible I/O.',
          file=sys.stderr, flush=True)


def become_subreaper():
    # Linux PR_SET_CHILD_SUBREAPER: orphaned grandchildren attach to this guard,
    # rather than a container PID 1 that may never reap them.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        print(f'[Guard] Could not enable child subreaper (errno={ctypes.get_errno()}); '
              'process-group termination remains active.', file=sys.stderr, flush=True)


def supervise(command, *, env=None, stop_grace=5.0):
    """The supervisor never imports torch or initializes a CUDA context."""
    become_subreaper()
    child, stopped, stop_time = None, None, None
    old_handlers = {}

    def on_signal(sig, _frame):
        nonlocal stopped, stop_time
        if stopped is None:
            stopped, stop_time = sig, time.monotonic()
        if child is not None:
            signal_group(child.pid, sig)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, on_signal)
        if stopped is not None:
            return 128+stopped
        child = subprocess.Popen(command, env=env, start_new_session=True)
        print(f'[Guard] Worker PID/process group: {child.pid}; '
              'this run\'s remaining workers will be stopped when it exits.', flush=True)
        if stopped is not None:
            signal_group(child.pid, stopped)
        while True:
            if stopped is not None:
                if time.monotonic()-stop_time >= stop_grace:
                    signal_group(child.pid, KILL_SIGNAL)
            try:
                status = child.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if status == -KILL_SIGNAL:
            print('[Guard] Training worker was killed by SIGKILL. Check host/container RAM limits '
                  'and OOM logs or external termination. This is not proof of CUDA out-of-memory.',
                  file=sys.stderr, flush=True)
        return 128+stopped if stopped is not None else (128-status if status < 0 else status)
    finally:
        if child is not None:
            cleanup_group(child.pid)
            child.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def entrypoint(main, script):
    # Do not supervise Windows or wrap the worker recursively. Verify the marker
    # against the actual parent, and remove it before creating DataLoader workers.
    parent = os.environ.pop(GUARD_ENV, None)
    if sys.platform != 'linux' or parent == str(os.getppid()):
        return main()
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        return main()
    env = dict(os.environ)
    env[GUARD_ENV] = str(os.getpid())
    return supervise([sys.executable, str(Path(script).resolve()), *sys.argv[1:]], env=env)
