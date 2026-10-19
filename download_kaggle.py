"""Download any Kaggle dataset by owner/slug or dataset URL using kagglehub."""
from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
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
                        max_retry_delay=60.0, **kwargs):
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
        print(f'Download attempt {attempt + 1}/{retries + 1}', flush=True)
        try:
            return download(handle, **kwargs)
        except Exception as exc:
            if attempt == retries or not is_retryable(exc):
                raise
            print(f'Temporary download failure ({type(exc).__name__}); '
                  f'retrying in {delay:g}s...', file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, max_retry_delay)
            if kwargs.get('output_dir'):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='OWNER/DATASET or full Kaggle dataset URL')
    parser.add_argument('--output', type=Path, help='Download directly to this directory; omitted = kagglehub cache')
    parser.add_argument('--version', type=int, help='Specific version; omitted = latest unless specified in URL/handle')
    parser.add_argument('--file', help='Download only this dataset-relative file or directory')
    parser.add_argument('--force', action='store_true', help='Force download; permits overwriting files in --output')
    parser.add_argument('--login', action='store_true', help='Interactively log in using kagglehub before downloading')
    parser.add_argument('--retries', type=int, default=3, help='Extra attempts after a transient failure (default: 3; 0 disables retries)')
    parser.add_argument('--retry-delay', type=float, default=5.0, help='Initial retry wait in seconds (default: 5; doubles each retry)')
    parser.add_argument('--max-retry-delay', type=float, default=60.0, help='Maximum retry wait in seconds (default: 60)')
    args = parser.parse_args()
    try:
        handle = dataset_handle(args.dataset, args.version)
        if args.retries < 0:
            raise ValueError('--retries must be >= 0')
        if not all(math.isfinite(v) and v >= 0 for v in (args.retry_delay, args.max_retry_delay)):
            raise ValueError('Retry delays must be finite and >= 0')
        if args.file:
            relative = PurePosixPath(args.file.replace('\\', '/'))
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or str(relative) == '.':
                raise ValueError('--file must be a relative path within the dataset')
        if args.output:
            output = args.output.expanduser().resolve()
            if output.exists() and not output.is_dir():
                raise ValueError(f'Output is not a directory: {output}')
            if output.is_dir() and any(output.iterdir()) and not args.force:
                raise ValueError('Output directory is non-empty; choose a new directory or explicitly use --force')
        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ValueError('Install dependency first: python -m pip install --upgrade kagglehub') from exc
        if args.login:
            kagglehub.login()
        kwargs = {'force_download': args.force}
        if args.output:
            kwargs['output_dir'] = str(output)
        if args.file:
            kwargs['path'] = str(relative)
        print(f'Dataset: {handle}', flush=True)
        location = download_with_retry(
            kagglehub.dataset_download, handle, retries=args.retries,
            retry_delay=args.retry_delay, max_retry_delay=args.max_retry_delay, **kwargs,
        )
        print(f'Path to dataset files: {Path(location).resolve()}')
    except KeyboardInterrupt:
        parser.exit(130, '\nDownload interrupted.\n')
    except Exception as exc:
        parser.exit(1, f'Download failed ({type(exc).__name__}): {exc}\n'
                    'Check network, dataset access and kagglehub version. For authentication, '
                    'retry with --login; accept dataset terms on Kaggle if required.\n')


if __name__ == '__main__':
    main()
