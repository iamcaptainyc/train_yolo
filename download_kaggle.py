"""Download any Kaggle dataset by owner/slug or dataset URL using kagglehub."""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
from threading import Event
from urllib.parse import parse_qs, urlparse


def is_retryable(error: Exception) -> bool:
    """Retry transient network failures, including requests/Kaggle HTTP errors."""
    from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, SSLError, Timeout

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HTTPError):
            status = error.response.status_code if error.response is not None else None
            return status in (408, 429) or (status is not None and 500 <= status < 600)
        if isinstance(error, SSLError):
            return False
        if isinstance(error, (Timeout, ConnectionError, ChunkedEncodingError, TimeoutError)):
            return True
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return False


def download_with_retry(download, handle, *, retries=3, retry_delay=5.0,
                        max_retry_delay=60.0, stop_event=None, restart_output=True, **kwargs):
    """retries counts extra attempts after the first call; never catches Ctrl+C.

    Callers must validate output_dir is empty or explicitly authorized for replacement.
    KaggleHub rejects partial output directories, so direct-output retries restart them.
    """
    if retries < 0:
        raise ValueError('--retries must be >= 0')
    if not all(math.isfinite(v) and v >= 0 for v in (retry_delay, max_retry_delay)):
        raise ValueError('Retry delays must be finite and >= 0')
    delay = min(retry_delay, max_retry_delay)
    for attempt in range(retries + 1):
        if stop_event is not None and stop_event.is_set():
            raise CancelledError('Download cancelled')
        print(f'[{handle}] Download attempt {attempt + 1}/{retries + 1}', flush=True)
        try:
            return download(handle, **kwargs)
        except Exception as exc:
            if attempt == retries or not is_retryable(exc):
                raise
            print(f'[{handle}] Temporary download failure ({type(exc).__name__}); '
                  f'retrying in {delay:g}s...', file=sys.stderr, flush=True)
            if stop_event is None:
                time.sleep(delay)
            elif stop_event.wait(delay):
                raise CancelledError('Download cancelled') from exc
            delay = min(delay * 2, max_retry_delay)
            if restart_output and kwargs.get('output_dir'):
                # Initial preflight permits an empty output or explicit --force only.
                # Replace partial files from this run; do not delete paths ourselves.
                kwargs['force_download'] = True


def dataset_handle(value: str, version: int | None = None) -> str:
    value = value.strip()
    if version is not None and version < 1:
        raise ValueError('--version must be a positive integer')
    url_version = None
    if '://' in value:
        url = urlparse(value)
        if url.scheme not in ('http', 'https') or url.hostname not in ('kaggle.com', 'www.kaggle.com'):
            raise ValueError('Expected a kaggle.com dataset URL')
        match = re.fullmatch(r'/datasets/([^/]+/[^/]+)(?:/versions/([1-9]\d*)|/(?:data|code|discussion))?/?', url.path)
        if not match:
            raise ValueError('Expected https://www.kaggle.com/datasets/OWNER/DATASET')
        value = match[1] + (f'/versions/{match[2]}' if match[2] else '')
        query_versions = parse_qs(url.query).get('datasetVersionNumber', [])
        if query_versions:
            if len(query_versions) != 1 or not re.fullmatch(r'[1-9]\d*', query_versions[0]):
                raise ValueError('Invalid datasetVersionNumber in URL')
            url_version = int(query_versions[0])
    match = re.fullmatch(r'([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)(?:/versions/([1-9]\d*))?/?', value)
    if not match:
        raise ValueError('Expected OWNER/DATASET or OWNER/DATASET/versions/N')
    versions = {v for v in (version, url_version, int(match[2]) if match[2] else None) if v is not None}
    if len(versions) > 1:
        raise ValueError('Conflicting dataset versions in handle, URL or --version')
    selected = next(iter(versions), None)
    return match[1] + (f'/versions/{selected}' if selected is not None else '')


def default_output_dir(handle: str) -> Path:
    """Keep owners and explicit versions separate to avoid dataset-name collisions."""
    parts = dataset_handle(handle).split('/')
    output = Path('datasets') / parts[0] / parts[1]
    if len(parts) == 4:
        output /= f'version-{parts[3]}'
    return output


def build_jobs(handles, output=None, cache=False, force=False, resume=False):
    """Validate all destinations before starting any downloads."""
    jobs = []
    for handle in handles:
        target = default_output_dir(handle)
        if output is not None:
            target = output if len(handles) == 1 else output / target.relative_to('datasets')
        target = None if cache else target.expanduser().resolve()
        if target is not None:
            for _, existing in jobs:
                if existing is not None and (target == existing or target in existing.parents or existing in target.parents):
                    raise ValueError(f'Download directories overlap: {existing} and {target}; download these separately')
            if target.exists() and not target.is_dir():
                raise ValueError(f'Output is not a directory: {target}')
            if target.is_dir() and any(target.iterdir()) and not (force or resume):
                raise ValueError(f'Output directory is non-empty: {target}; use --resume to continue, '
                                 'or choose a new directory (--force restarts and may discard partial progress)')
        else:
            # A latest-version handle may resolve to a simultaneously requested explicit version.
            parts = handle.split('/')
            for existing, _ in jobs:
                other = existing.split('/')
                if parts[:2] == other[:2] and (len(parts) == 2 or len(other) == 2):
                    raise ValueError('Latest and explicit versions of the same dataset may share a cache; download separately')
        jobs.append((handle, target))
    return jobs


def run_downloads(download, jobs, *, workers=3, retries=3, retry_delay=5.0,
                  max_retry_delay=60.0, force=False, path=None, resume=False):
    """Run independent datasets concurrently; one failure does not cancel the others."""
    if workers < 1:
        raise ValueError('--workers must be >= 1')
    stop = Event()
    successes, failures = [], []

    def run_one(handle, output):
        kwargs = {'force_download': force}
        if output is not None:
            kwargs['output_dir'] = str(output)
        if path is not None:
            kwargs['path'] = path
        print(f'[{handle}] Download directory: {output if output is not None else "kagglehub cache"}', flush=True)
        return download_with_retry(download, handle, retries=retries, retry_delay=retry_delay,
                                   max_retry_delay=max_retry_delay, stop_event=stop,
                                   restart_output=not resume, **kwargs)

    executor = ThreadPoolExecutor(max_workers=min(workers, max(1, len(jobs))))
    futures = {}
    try:
        for handle, output in jobs:
            futures[executor.submit(run_one, handle, output)] = handle
        for future in as_completed(futures):
            handle = futures[future]
            try:
                location = future.result()
            except Exception as exc:
                failures.append((handle, exc))
                print(f'[{handle}] Failed ({type(exc).__name__}): {exc}', file=sys.stderr, flush=True)
            else:
                successes.append((handle, location))
                print(f'[{handle}] Path to dataset files: {Path(location).resolve()}', flush=True)
    except KeyboardInterrupt:
        stop.set()
        for future in futures:
            future.cancel()
        print('Cancelling queued downloads; waiting for active network calls to return...', file=sys.stderr, flush=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return successes, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='+', help='One or more OWNER/DATASET handles or full Kaggle dataset URLs')
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--output', type=Path, help='Single dataset: exact directory; multiple: base directory for OWNER/DATASET[/version-N]')
    destination.add_argument('--cache', action='store_true', help='Use kagglehub cache instead of an automatically named directory')
    parser.add_argument('--version', type=int, help='Specific version; omitted = latest unless specified in URL/handle')
    parser.add_argument('--file', help='Download only this dataset-relative file or directory')
    parser.add_argument('--force', action='store_true', help='Force download; permits overwriting files in --output')
    parser.add_argument('--resume', action='store_true', help='Continue a full dataset from its existing .archive without clearing the directory')
    parser.add_argument('--login', action='store_true', help='Interactively log in using kagglehub before downloading')
    parser.add_argument('--retries', type=int, default=3, help='Extra attempts after a transient failure (default: 3; 0 disables retries)')
    parser.add_argument('--retry-delay', type=float, default=5.0, help='Initial retry wait in seconds (default: 5; doubles each retry)')
    parser.add_argument('--max-retry-delay', type=float, default=60.0, help='Maximum retry wait in seconds (default: 60)')
    parser.add_argument('--workers', type=int, default=3, help='Maximum simultaneous dataset downloads (default: 3; 1 = sequential)')
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError('--workers must be >= 1')
        if args.resume and (args.force or args.cache or args.file):
            raise ValueError('--resume cannot be combined with --force, --cache or --file')
        if len(args.dataset) > 1 and (args.version is not None or args.file):
            raise ValueError('--version and --file require a single dataset; specify /versions/N per dataset for batches')
        handles = list(dict.fromkeys(dataset_handle(value, args.version).casefold() for value in args.dataset))
        if args.retries < 0:
            raise ValueError('--retries must be >= 0')
        if not all(math.isfinite(v) and v >= 0 for v in (args.retry_delay, args.max_retry_delay)):
            raise ValueError('Retry delays must be finite and >= 0')
        if args.file:
            relative = PurePosixPath(args.file.replace('\\', '/'))
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or str(relative) == '.':
                raise ValueError('--file must be a relative path within the dataset')
        jobs = build_jobs(handles, args.output, args.cache, args.force, args.resume)
        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ValueError('Install dependency first: python -m pip install --upgrade kagglehub') from exc
        if args.login:
            kagglehub.login()
        download = kagglehub.dataset_download
        if args.resume:
            from kaggle_resume import resumable_dataset_download
            download = resumable_dataset_download
        print(f'Datasets: {len(jobs)}; concurrent downloads: {min(args.workers, len(jobs))}', flush=True)
        successes, failures = run_downloads(
            download, jobs, workers=args.workers, retries=args.retries,
            retry_delay=args.retry_delay, max_retry_delay=args.max_retry_delay,
            force=args.force, path=str(relative) if args.file else None, resume=args.resume,
        )
        print(f'Finished: {len(successes)} succeeded, {len(failures)} failed.', flush=True)
        if failures:
            parser.exit(1, 'Failed datasets: ' + ', '.join(handle for handle, _ in failures) + '\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nDownload interrupted.\n')
    except Exception as exc:
        parser.exit(1, f'Download failed ({type(exc).__name__}): {exc}\n'
                    'Check network, dataset access and kagglehub version. For authentication, '
                    'retry with --login; accept dataset terms on Kaggle if required.\n')


if __name__ == '__main__':
    main()
