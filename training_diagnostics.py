"""Best-effort Linux diagnostics, independent of the training process and CUDA runtime."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

LOG_ENV = 'TRAIN_YOLO_DIAGNOSTICS'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {'unavailable': str(exc)}


def counters(value):
    if not isinstance(value, str):
        return {}
    result = {}
    for line in value.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def cgroup_paths():
    base = Path('/sys/fs/cgroup')
    value = read('/proc/self/cgroup')
    if isinstance(value, str):
        for line in value.splitlines():
            if line.startswith('0::'):
                candidate = base/line[3:].lstrip('/')
                if (candidate/'memory.events').is_file():
                    paths = []
                    while candidate == base or base in candidate.parents:
                        paths.append(candidate)
                        if candidate == base:
                            return paths
                        candidate = candidate.parent
    return [base]  # Namespace-root mount; missing v2 files are explicitly recorded.


def processes(pgid):
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            stat = (path/'stat').read_text()
            fields = stat[stat.rfind(')')+2:].split()
            if int(fields[2]) != pgid:
                continue
            result.append({'pid': int(path.name), 'ppid': int(fields[1]), 'state': fields[0],
                           'cpu_ticks': int(fields[11])+int(fields[12]),
                           'rss_bytes': int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                           'wchan': read(path/'wchan'), 'io': read(path/'io'),
                           'status': read(path/'status')})
        except (OSError, ValueError, IndexError):
            continue
    return result


def event_delta(first, last):
    return {path: {key: value - first.get(path, {}).get(key, value) for key, value in values.items()}
            for path, values in last.items()}


def diagnosis(code, delta, received_signal=None, forced=False):
    if received_signal is not None:
        return f'守护进程收到信号 {received_signal} 并转发；超时强制终止={forced}。'
    killed = sum(values.get('oom_kill', 0) for values in delta.values())
    if code == -9 and killed > 0:
        return '训练进程收到 SIGKILL；运行期间可见 cgroup 的 oom_kill 增加，支持发生 OOM 杀进程，但共享 cgroup 下不能仅凭计数确定被杀对象。'
    if code == -9:
        return '训练进程收到 SIGKILL；未观测到可见 cgroup 的 oom_kill 增量。原因未确定：宿主机/不可见上级 OOM 或外部终止仍可能，需要平台日志。'
    return f'进程退出码 {code}；请结合 console.log 中的异常和资源时间线判断。'


class Diagnostics:
    def __init__(self, command):
        base = Path(os.environ.get('TRAIN_YOLO_LOG_ROOT', 'runs/diagnostics')).resolve()
        self.directory = base/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True)
        self.stream = (self.directory/'resources.jsonl').open('a', encoding='utf-8', buffering=1)
        self.paths = cgroup_paths()
        self.first = {}
        self.last = {}
        self.observed_memory = {}
        self.next_sample = 0
        self.thread = None
        self.error = None
        self.emit({'kind': 'start', 'command': command, 'cwd': str(Path.cwd()),
                   'supervisor_pid': os.getpid(), 'cgroup_membership': read('/proc/self/cgroup'),
                   'sample_interval_seconds': 5})
        self.sample(None, force=True)
        self.first = self.last
        print(f'[Guard] Diagnostics: {self.directory}', flush=True)

    def emit(self, value):
        try:
            self.stream.write(json.dumps({'time_utc': now(), **value}, ensure_ascii=False)+'\n')
            self.stream.flush()
        except OSError as exc:
            if self.error is None:
                print(f'[Guard] Cannot write diagnostics: {exc}', file=sys.stderr, flush=True)
            self.error = str(exc)

    def attach(self, child):
        def tee():
            log = None
            try:
                try:
                    log = (self.directory/'console.log').open('ab', buffering=0)
                except OSError as exc:
                    self.emit({'kind': 'console_open_error', 'error': str(exc)})
                while True:
                    block = child.stdout.read1(16384)
                    if not block:
                        break
                    if log:
                        try:
                            log.write(block)
                        except OSError as exc:
                            self.emit({'kind': 'console_write_error', 'error': str(exc)})
                            log.close()
                            log = None
                    try:
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
                    except (OSError, AttributeError):
                        pass
            except Exception as exc:
                self.emit({'kind': 'console_capture_error', 'error': str(exc)})
            finally:
                if log:
                    log.close()
        self.thread = threading.Thread(target=tee, daemon=True)
        self.thread.start()

    def sample(self, pgid, force=False):
        if not force and time.monotonic() < self.next_sample:
            return
        self.next_sample = time.monotonic()+5
        try:
            groups = {}
            for path in self.paths:
                groups[str(path)] = {name: read(path/name) for name in (
                    'memory.current', 'memory.peak', 'memory.high', 'memory.max',
                    'memory.events', 'memory.events.local', 'memory.stat', 'memory.pressure',
                    'memory.swap.current', 'memory.swap.max', 'io.pressure', 'cpu.stat')}
            self.last = {path: counters(values['memory.events']) for path, values in groups.items()}
            for path, values in groups.items():
                current = values['memory.current']
                if isinstance(current, str) and current.isdecimal():
                    previous = self.observed_memory.get(path, {}).get('peak_observed_bytes', 0)
                    self.observed_memory[path] = {'peak_observed_bytes': max(previous, int(current)),
                                                  'memory_max_bytes_or_max': values['memory.max']}

            gpu = {'unavailable': 'nvidia-smi not installed'}
            if shutil.which('nvidia-smi'):
                try:
                    r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    gpu = {'returncode': r.returncode, 'csv': r.stdout.strip(), 'stderr': r.stderr.strip()}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    gpu = {'unavailable': str(exc)}
            self.emit({'kind': 'sample', 'pgid': pgid, 'cgroups': groups, 'gpu': gpu,
                       'processes': processes(pgid) if pgid else [], 'host_meminfo': read('/proc/meminfo'),
                       'host_memory_pressure': read('/proc/pressure/memory'),
                       'host_io_pressure': read('/proc/pressure/io'), 'shm': self.shm()})
        except Exception as exc:
            self.emit({'kind': 'sample_error', 'error': str(exc)})

    @staticmethod
    def shm():
        try:
            usage = shutil.disk_usage('/dev/shm')
            return {'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free}
        except OSError as exc:
            return {'unavailable': str(exc)}

    def finish(self, pgid, code, received_signal, forced):
        self.sample(pgid, force=True)
        delta = event_delta(self.first, self.last)
        conclusion = diagnosis(code, delta, received_signal, forced)
        self.emit({'kind': 'exit_before_cleanup', 'returncode': code, 'received_signal': received_signal,
                   'guard_sent_sigkill': forced, 'memory_events_delta': delta, 'assessment': conclusion, 'observed_memory': self.observed_memory})
        (self.directory/'DIAGNOSIS.md').write_text(
            '# 训练退出诊断\n\n'+conclusion+'\n\n'
            '## 本次采样到的内存峰值和限额（bytes）\n\n```json\n'+json.dumps(self.observed_memory, ensure_ascii=False, indent=2)+'\n```\n\n'
            '## 本次运行内存事件增量\n\n```json\n'+json.dumps(delta, ensure_ascii=False, indent=2)+'\n```\n\n'
            '资源快照：resources.jsonl（约每 5 秒，时间为 UTC）；完整训练输出：console.log；阶段事件：phases.jsonl（若已启动模型）。\n\n'
            'high/max 增加表示内存压力，不等于 OOM 杀进程。GPU 查询是整张卡统计；主机 meminfo 不等于容器额度。'
            '计数来自可见 cgroup，可能包含其他进程。5 秒采样可能错过瞬时峰值。'
            '若容器/守护进程也被杀，可能只有最后的资源快照，没有退出诊断，不能因此断言退出原因。\n', encoding='utf-8')

    def close(self, pgid):
        self.emit({'kind': 'after_cleanup', 'remaining_processes': processes(pgid) if pgid else []})
        if self.thread:
            self.thread.join(timeout=1)
        # A D-state descendant may still hold the pipe; leave the daemon writer's stream open.
        if not self.thread or not self.thread.is_alive():
            self.stream.close()


def install_phase_logging(model):
    directory = os.environ.get(LOG_ENV)
    if not directory:
        return
    def record(event):
        def callback(trainer):
            try:
                item = {'time_utc': now(), 'pid': os.getpid(), 'event': event,
                        'epoch_index': getattr(trainer, 'epoch', None),
                        'save_dir': str(getattr(trainer, 'save_dir', ''))}
                args = getattr(trainer, 'args', None)
                item['settings'] = {key: getattr(args, key, None) for key in
                                    ('batch', 'workers', 'imgsz', 'epochs', 'cache', 'device')}
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    item['cuda_allocator_bytes'] = {
                        'device': torch.cuda.current_device(), 'allocated': torch.cuda.memory_allocated(),
                        'reserved': torch.cuda.memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated()}
                with (Path(directory)/'phases.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(item, ensure_ascii=False, default=str)+'\n')
                    stream.flush()
            except Exception as exc:
                print(f'[Diagnostics] Phase logging failed: {exc}', file=sys.stderr, flush=True)
        return callback
    for event in ('on_train_start', 'on_train_epoch_start', 'on_train_epoch_end',
                  'on_val_start', 'on_val_end', 'on_model_save', 'on_train_end'):
        model.add_callback(event, record(event))
