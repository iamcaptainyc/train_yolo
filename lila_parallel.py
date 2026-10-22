"""Bounded threaded map with ordered results and completion-based progress."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def bounded_map(function, values, workers, progress=None):
    """Keep at most 2*workers submitted/buffered items; yield in input order."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers == 1:
        for value in values:
            result = function(value)
            if progress is not None:
                progress.update(1)
            yield result
        return
    iterator = iter(values)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='lila')
    pending, ready = {}, {}
    submitted, expected = 0, 0

    def submit_one():
        nonlocal submitted
        try:
            value = next(iterator)
        except StopIteration:
            return
        pending[pool.submit(function, value)] = submitted
        submitted += 1

    try:
        for _ in range(2 * workers):
            submit_one()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                ready[index] = future.result()
                if progress is not None:
                    progress.update(1)
            while expected in ready:
                result = ready.pop(expected)
                expected += 1
                yield result
                submit_one()
    finally:
        for future in pending:
            future.cancel()
        # Running file I/O cannot be cancelled; join it instead of leaving background writers.
        pool.shutdown(wait=True, cancel_futures=True)
