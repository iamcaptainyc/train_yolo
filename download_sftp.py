"""Resume one file over OpenSSH SFTP, with keepalive, progress and reconnects."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time


def quoted(value):
    if any(c in value for c in '\r\n\0'):
        raise ValueError('Paths must not contain newlines or NUL')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def batch(remote, output):
    # This tool downloads one exact path, never a remote wildcard expansion.
    if not remote.startswith('/') or any(c in remote for c in '*?[]'):
        raise ValueError('Remote path must be absolute, without wildcard characters')
    return f'@reget {quoted(remote)} {quoted("./" + output.name)}\n@bye\n'


def remote_size(line):
    # OpenSSH remote listings use '?' for link count (SFTP does not supply it).
    # Ordinary ls listings may instead contain a numeric link count.
    # A symlink listing gives the link length, not target size: leave it unknown.
    # Windows OpenSSH may render unrepresented permission bits as '*'.
    match = re.match(r'^-[rwxStTs*?-]{9}[+@.]?\s+(?:\d+|\?)\s+\S+\s+\S+\s+(\d+)\s+', line.strip())
    return int(match[1]) if match else None


def amount(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f}{unit}'
        value /= 1024


def duration(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


class Progress:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.last = 0
        self.width = 0
        self.samples = deque()

    def show(self, current, total=None, state='Connecting', force=False):
        now = time.monotonic()
        self.samples.append((now, current))
        while len(self.samples) > 2 and self.samples[1][0] < now - 10:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        rate = max(0, current - self.samples[0][1]) / elapsed if elapsed else 0
        if not force and now - self.last < (0.25 if self.tty else 10):
            return
        self.last = now
        fraction = min(1, current / total) if total else (1 if total == 0 and current == 0 else None)
        percent = f'{fraction * 100:5.1f}%' if fraction is not None else '    --%'
        filled = int(fraction * 16) if fraction is not None else 0
        bar = '=' * filled + ' ' * (16 - filled)
        eta = (max(0, total - current) / rate if rate else (0 if current == total else None)) if total is not None else None
        line = (f'{percent} [{bar}] {amount(current)}/{amount(total) if total is not None else "?"}'
                f' | {amount(rate)}/s | ETA {duration(eta)} | {state}')
        if self.tty:
            columns = shutil.get_terminal_size((120, 24)).columns
            line = line[:max(1, columns - 1)]
            self.stream.write('\r' + line + ' ' * max(0, min(self.width, columns - 1) - len(line)))
            self.width = len(line)
        else:
            self.stream.write(line + '\n')
        self.stream.flush()

    def finish(self):
        if self.tty and self.width:
            self.stream.write('\n')
            self.stream.flush()
            self.width = 0


def command(host, executable='sftp'):
    if not host or host.startswith('-') or any(c.isspace() or c == '\0' for c in host):
        raise ValueError('Use an SSH host alias or user@host')
    return [executable, '-b', '-', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', host]


def size(path):
    return path.stat().st_size if path.exists() else 0


def stop(proc):
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def query_size(cmd, remote, output):
    """Finish a separate SFTP listing before transfer; do not depend on live flushing."""
    options = {'start_new_session': True} if os.name != 'nt' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    progress = Progress()
    progress.show(size(output), state='Querying remote file size', force=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, **options)
    payload = f'@ls -ln {quoted(remote)}\n@bye\n'.encode('utf-8')
    started = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(input=payload, timeout=.25)
                break
            except subprocess.TimeoutExpired:
                payload = None
                progress.show(size(output), state='Querying remote file size')
                if time.monotonic() - started >= 45:
                    raise ValueError('Remote size query timed out after 45s')
        if proc.returncode:
            raise ValueError('Remote size query failed: ' + stderr.decode('utf-8', errors='replace').strip())
        lines = stdout.decode('utf-8', errors='replace').splitlines()
        totals = [n for line in lines if (n := remote_size(line)) is not None]
        if len(totals) != 1:
            raise ValueError('Cannot parse remote size; actual SFTP output: ' + repr('\n'.join(lines)[-2000:]))
        return totals[0]
    finally:
        stop(proc)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream and not stream.closed:
                stream.close()
        progress.finish()


def transfer(cmd, script, output, stall_timeout=600, total=None):
    options = {'start_new_session': True} if os.name != 'nt' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    progress = Progress()
    progress.show(size(output), total, state='Connecting / resuming', force=True)
    proc = subprocess.Popen(cmd, cwd=output.parent, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options)
    details = {'total': total}
    messages = deque(maxlen=20)

    def read_output():
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', errors='replace').strip()
            total = remote_size(line)
            if total is not None:
                details['total'] = total
            elif line and not line.startswith('sftp>'):
                messages.append(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        try:
            proc.stdin.write(script.encode('utf-8'))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        previous = size(output)
        initial = previous
        last_change = time.monotonic()
        while proc.poll() is None:
            time.sleep(0.25)
            now, current = time.monotonic(), size(output)
            if current != previous:
                previous, last_change = current, now
            idle = now - last_change
            state = 'Downloading' if current > initial else 'Connecting / resuming'
            if idle >= 5:
                state = f'Waiting {int(idle)}s'
            progress.show(current, details['total'], state)
            if stall_timeout and now - last_change >= stall_timeout:
                progress.finish()
                print(f'No file growth for {stall_timeout}s; reconnecting.', flush=True)
                stop(proc)
                return 124
        reader.join(timeout=2)
        progress.show(size(output), details['total'],
                      'Transferred' if proc.returncode == 0 else 'Disconnected', force=True)
        return proc.returncode
    finally:
        stop(proc)
        reader.join(timeout=2)
        progress.finish()
        if proc.returncode != 0:
            for message in list(messages):
                print(message, file=sys.stderr, flush=True)
        if not reader.is_alive():
            proc.stdout.close()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()


def download(host, remote, output, retries=10, retry_delay=10, stall_timeout=600, sha256=None):
    executable = shutil.which('sftp')
    if not executable:
        raise ValueError('sftp not found; install/enable the OpenSSH client')
    output = output.expanduser().resolve()
    script, cmd = batch(remote, output), command(host, executable)
    if output.is_dir():
        raise ValueError('--output must be the full destination FILE path')
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    delay = retry_delay
    while True:
        attempt += 1
        print(f'Attempt {attempt}; resume at {size(output):,} bytes: {output}', flush=True)
        try:
            total = query_size(cmd, remote, output)
        except ValueError as exc:
            print(f'Warning: {exc}. Download will continue with unknown total.', flush=True)
            total = None
        if total is not None:
            print(f'Remote size: {total:,} bytes ({amount(total)})', flush=True)
            if size(output) > total:
                raise ValueError('Local file is larger than remote file; check that the remote ZIP is unchanged')
        code = transfer(cmd, script, output, stall_timeout, total=total)
        if code == 0:
            if not output.is_file():
                raise ValueError('SFTP reported success but the destination file is missing')
            if sha256:
                print('Verifying SHA256...', flush=True)
                checksum = hashlib.sha256()
                with output.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        checksum.update(chunk)
                if checksum.hexdigest().lower() != sha256.lower():
                    raise ValueError('SHA256 mismatch. File retained. Check whether the remote ZIP changed; '
                                     'resuming cannot repair a mismatched existing prefix. Use a new output file.')
                print('SHA256 matches.', flush=True)
            print(f'Download complete: {output} ({size(output):,} bytes)', flush=True)
            return
        if retries and attempt > retries:
            raise ValueError(f'SFTP exit={code}; retry budget exhausted. Partial file retained; rerun to resume.')
        print(f'SFTP exit={code}; retry in {delay}s. Partial file retained. '
              'Authentication requires an SSH key/agent; use Ctrl+C to stop.', flush=True)
        progress = Progress()
        try:
            for remaining in range(delay, 0, -1):
                progress.show(size(output), state=f'Reconnecting in {remaining}s', force=remaining == delay)
                time.sleep(1)
        finally:
            progress.finish()
        delay = min(delay * 2, 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', help='Existing SSH alias, e.g. 26080-yc')
    parser.add_argument('remote', help='Absolute path of the remote file')
    parser.add_argument('--output', type=Path, required=True, help='Full local file path; existing partial file is resumed')
    parser.add_argument('--retries', type=int, default=10, help='Retries after first attempt; 0 = unlimited')
    parser.add_argument('--retry-delay', type=int, default=10, help='Initial retry delay; doubles up to 60 seconds')
    parser.add_argument('--stall-timeout', type=int, default=600, help='Reconnect after this many seconds without file growth; 0 disables')
    parser.add_argument('--sha256', help='Optional expected SHA256 of the unchanged remote file')
    args = parser.parse_args()
    if args.retries < 0 or not 1 <= args.retry_delay <= 60 or args.stall_timeout < 0:
        parser.error('retries/stall-timeout must be >=0; retry-delay must be 1..60')
    if args.sha256 and not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256):
        parser.error('--sha256 must contain exactly 64 hexadecimal characters')
    try:
        download(**vars(args))
    except KeyboardInterrupt:
        parser.exit(130, '\nStopped. Partial file retained; rerun the same command to resume.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
