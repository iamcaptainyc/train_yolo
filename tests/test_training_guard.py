import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from training_guard import GUARD_ENV, cleanup_group, entrypoint, supervise


class GuardTests(unittest.TestCase):
    def test_cleanup_escalates_only_the_recorded_process_group(self):
        with patch('training_guard.signal_group', side_effect=[True, True, False]) as send, \
                patch('training_guard.reap_group'):
            cleanup_group(12348, grace=0)
        self.assertEqual([c.args for c in send.call_args_list],
                         [(12348, signal.SIGTERM), (12348, 9), (12348, 0)])

    def test_already_exited_group_is_not_killed_again(self):
        with patch('training_guard.signal_group', return_value=False) as send, \
                patch('training_guard.reap_group') as reap:
            cleanup_group(12349)
        send.assert_called_once_with(12349, signal.SIGTERM)
        reap.assert_called_once_with(12349)

    def test_killed_worker_triggers_group_cleanup_and_exit_137(self):
        child = Mock(pid=12345)
        child.wait.return_value = -9
        with patch('training_guard.become_subreaper'), patch('training_guard.subprocess.Popen', return_value=child) as launch, \
                patch('training_guard.cleanup_group') as cleanup, patch('training_guard.signal.signal'):
            self.assertEqual(supervise(['python', 'task.py']), 137)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        cleanup.assert_called_once_with(12345)

    def test_worker_exception_also_cleans_only_its_group(self):
        child = Mock(pid=12346)
        child.wait.side_effect = [RuntimeError('wait failed'), 0]
        with patch('training_guard.become_subreaper'), patch('training_guard.subprocess.Popen', return_value=child), \
                patch('training_guard.cleanup_group') as cleanup, patch('training_guard.signal.signal'):
            with self.assertRaisesRegex(RuntimeError, 'wait failed'):
                supervise(['python', 'task.py'])
        cleanup.assert_called_once_with(12346)

    def test_guard_marker_is_removed_in_worker(self):
        main = Mock(return_value=0)
        with patch('training_guard.sys.platform', 'linux'), patch.dict(os.environ, {GUARD_ENV:str(os.getppid())}), \
                patch('training_guard.supervise') as guard:
            self.assertEqual(entrypoint(main, 'train.py'), 0)
            self.assertNotIn(GUARD_ENV, os.environ)
            guard.assert_not_called()
        main.assert_called_once()

    def test_ctrl_c_is_forwarded_and_cleanup_runs(self):
        child = Mock(pid=12347)
        handlers = {}
        def register(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL
        def wait(timeout=None):
            if timeout is not None:
                handlers[signal.SIGINT](signal.SIGINT, None)
                return 0
            return 0
        child.wait.side_effect = wait
        with patch('training_guard.become_subreaper'), patch('training_guard.subprocess.Popen', return_value=child), \
                patch('training_guard.cleanup_group') as cleanup, patch('training_guard.signal_group') as forward, \
                patch('training_guard.signal.signal', side_effect=register):
            self.assertEqual(supervise(['python','task.py']), 130)
        forward.assert_called_with(12347, signal.SIGINT)
        cleanup.assert_called_once_with(12347)

    @unittest.skipUnless(sys.platform == 'linux', 'Requires Linux process groups and subreaper')
    def test_linux_sigkill_reaps_actual_orphan_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/'worker.pid'
            script = Path(tmp)/'task.py'
            script.write_text('import subprocess,sys,os,signal\n'
                              'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])\n'
                              'open(sys.argv[1],"w").write(str(p.pid))\n'
                              'os.kill(os.getpid(),signal.SIGKILL)\n')
            # Run the subreaper in a dedicated process, just like train.py.
            command = [sys.executable, '-c', 'from training_guard import supervise; '
                       'import sys; sys.exit(supervise(sys.argv[1:]))', sys.executable, str(script), str(marker)]
            result = subprocess.run(command, timeout=15, capture_output=True, text=True)
            self.assertEqual(result.returncode, 137, result.stderr)
            pid = int(marker.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
