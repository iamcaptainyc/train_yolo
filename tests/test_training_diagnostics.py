import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training_diagnostics import Diagnostics, diagnosis, event_delta, install_phase_logging, LOG_ENV


class DiagnosticTests(unittest.TestCase):
    def test_old_oom_counts_are_not_mistaken_for_new_oom(self):
        delta = event_delta({'cg': {'oom_kill': 5, 'high': 100}}, {'cg': {'oom_kill': 5, 'high': 103}})
        self.assertEqual(delta['cg']['high'], 3)
        self.assertIn('原因未确定', diagnosis(-9, delta))
        self.assertIn('支持发生 OOM', diagnosis(-9, {'cg': {'oom_kill': 1}}))
        self.assertIn('守护进程收到信号 2', diagnosis(-9, delta, 2, True))

    def test_samples_and_exit_report_survive_child_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TRAIN_YOLO_LOG_ROOT': tmp}), \
                patch('training_diagnostics.cgroup_paths', return_value=[]), \
                patch('training_diagnostics.shutil.which', return_value=None):
            observer = Diagnostics(['python', 'train.py'])
            observer.finish(12345, -9, None, False)
            observer.close(12345)
            data = [json.loads(line) for line in (observer.directory/'resources.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(data[-2]['kind'], 'exit_before_cleanup')
            self.assertEqual(data[-2]['returncode'], -9)
            self.assertEqual(data[-1]['kind'], 'after_cleanup')
            self.assertIn('原因未确定', (observer.directory/'DIAGNOSIS.md').read_text(encoding='utf-8'))

    def test_console_pipe_keeps_stderr_when_worker_exits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TRAIN_YOLO_LOG_ROOT': tmp}), \
                patch('training_diagnostics.cgroup_paths', return_value=[]), \
                patch('training_diagnostics.shutil.which', return_value=None):
            observer = Diagnostics(['fake-training'])
            child = subprocess.Popen([sys.executable, '-c', 'import sys; print("epoch"); print("failure",file=sys.stderr)'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with patch('training_diagnostics.sys.stdout', SimpleNamespace(buffer=io.BytesIO())):
                observer.attach(child)
                self.assertEqual(child.wait(timeout=10), 0)
                observer.thread.join(timeout=5)
            observer.close(None)
            child.stdout.close()
            content = (observer.directory/'console.log').read_text()
            self.assertIn('epoch', content)
            self.assertIn('failure', content)

    def test_phase_events_record_validation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {LOG_ENV: tmp}):
            callbacks = {}
            model = SimpleNamespace(add_callback=lambda event, fn: callbacks.update({event: fn}))
            install_phase_logging(model)
            callbacks['on_val_start'](SimpleNamespace(epoch=0, save_dir=Path(tmp)))
            row = json.loads((Path(tmp)/'phases.jsonl').read_text())
            self.assertEqual(row['event'], 'on_val_start')
            self.assertEqual(row['epoch_index'], 0)


if __name__ == '__main__':
    unittest.main()
