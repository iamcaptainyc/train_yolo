from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import threading
import unittest

from lila_parallel import bounded_map


class BoundedMapTests(unittest.TestCase):
    def test_queue_is_bounded_and_results_stay_ordered_with_slow_first_item(self):
        release = threading.Event()
        window_ready = threading.Event()
        others_done = threading.Event()
        consumed = []
        active = 0
        peak = 0
        lock = threading.Lock()

        def source():
            for i in range(100):
                consumed.append(i)
                if len(consumed) == 8:
                    window_ready.set()
                yield i

        def work(i):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            if i == 0:
                if not release.wait(10):
                    raise RuntimeError('test coordinator timeout')
            if i == 7:
                others_done.set()
            with lock:
                active -= 1
            return i * i

        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(lambda: list(bounded_map(work, source(), 4)))
            try:
                self.assertTrue(window_ready.wait(10))
                self.assertTrue(others_done.wait(10))
                self.assertEqual(len(consumed), 8)
                self.assertGreaterEqual(peak, 2)
                self.assertLessEqual(peak, 4)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10), [i * i for i in range(100)])

    def test_progress_reports_completed_jobs_before_ordered_first_result(self):
        release = threading.Event()

        class Progress:
            count = 0

            def update(self, n):
                self.count += n
                if self.count >= 3:
                    release.set()

        progress = Progress()

        def work(i):
            if i == 0 and not release.wait(10):
                raise RuntimeError('progress failed to advance')
            return i

        self.assertEqual(list(bounded_map(work, range(6), 3, progress)), list(range(6)))
        self.assertEqual(progress.count, 6)

    def test_failure_and_consumer_close_leave_no_pool_threads(self):
        def work(i):
            if i == 0:
                raise ValueError('invalid image')
            return i

        with self.assertRaisesRegex(ValueError, 'invalid image'):
            list(bounded_map(work, range(1000), 4))
        with closing(bounded_map(lambda i: i, range(1000), 4)) as results:
            self.assertEqual(next(results), 0)
        self.assertFalse(any(t.name.startswith('lila_') for t in threading.enumerate()))

    def test_serial_and_invalid_worker_counts(self):
        self.assertEqual(list(bounded_map(lambda i: i + 1, [1, 2, 3], 1)), [2, 3, 4])
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                list(bounded_map(lambda i: i, range(5), workers))


if __name__ == '__main__':
    unittest.main()
