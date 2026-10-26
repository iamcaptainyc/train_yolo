"""Resume one file over OpenSSH SFTP, with keepalive, progress and reconnects."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def quoted(value):
    if any(c in value for c in '\r\n\0'):
        raise ValueError('Paths must not contain newlines or NUL')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def batch(remote, output):
    # This tool downloads one exact path, never a remote wildcard expansion.
    if not remote.startswith('/') or any(c in remote for c in '*?[]'):
        raise ValueError('Remote path must be absolute, without wildcard characters')
    return f'reget {quoted(remote)} {quoted("./" + output.name)}\nbye\n'


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


def transfer(cmd, script, output, stall_timeout=600):
    options = {'start_new_session': True} if os.name != 'nt' else {
        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    proc = subprocess.Popen(cmd, cwd=output.parent, stdin=subprocess.PIPE, **options)
    try:
        try:
            proc.stdin.write(script.encode('utf-8'))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        previous = size(output)
        last_change = last_print = time.monotonic()
        printed_size = previous
        while proc.poll() is None:
            time.sleep(1)
            now, current = time.monotonic(), size(output)
            if current != previous:
                previous, last_change = current, now
            if now - last_print >= 10:
                rate = max(0, current - printed_size) / (now - last_print) / 1024
                print(f'{current / 1024**2:,.1f} MiB saved | {rate:,.1f} KiB/s', flush=True)
                last_print, printed_size = now, current
            if stall_timeout and now - last_change >= stall_timeout:
                print(f'No file growth for {stall_timeout}s; reconnecting.', flush=True)
                stop(proc)
                return 124
        return proc.returncode
    finally:
        stop(proc)
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
        code = transfer(cmd, script, output, stall_timeout)
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
        time.sleep(delay)
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
