"""Download LILA camera-trap datasets with bbox filtering via AWS, Azure or GCP."""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from threading import Event
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

CATALOG_URL = 'https://lila.science/wp-content/uploads/2023/06/lila_camera_trap_datasets.csv'
TOOLS = {'aws': 'aws', 'azure': 'azcopy', 'gcp': 'gsutil'}
PUBLIC_ROOTS = {
    'aws': 's3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/',
    'azure': 'https://lilawildlife.blob.core.windows.net/lila-wildlife/',
    'gcp': 'gs://public-datasets-lila/',
}


def read_catalog(text):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if not {'name', 'short_name'}.issubset(reader.fieldnames or []):
        raise ValueError('Catalog CSV must contain name and short_name columns')
    rows, names = [], set()
    for row in reader:
        row = {k: (v or '').strip() for k, v in row.items() if k is not None}
        if not row['name']:
            continue
        slug = row['short_name']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', slug):
            raise ValueError(f'Invalid short_name in catalog: {slug}')
        if slug.casefold() in names:
            raise ValueError(f'Duplicate short_name in catalog: {slug}')
        names.add(slug.casefold())
        rows.append(row)
    if not rows:
        raise ValueError('Catalog is empty')
    return rows


def load_catalog(local, cached, refresh=False, retries=3):
    if local is not None:
        return read_catalog(local.read_text(encoding='utf-8-sig'))
    if cached.is_file() and not refresh:
        return read_catalog(cached.read_text(encoding='utf-8-sig'))
    for attempt in range(retries + 1):
        try:
            with urlopen(CATALOG_URL, timeout=30) as response:
                text = response.read().decode('utf-8-sig')
            rows = read_catalog(text)
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix('.tmp')
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(cached)
            return rows
        except (OSError, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 10))


def select_dataset(rows, query):
    query = query.strip().casefold()
    exact = [r for r in rows if query in (r['name'].casefold(), r['short_name'].casefold())]
    matches = exact or [r for r in rows if query in r['name'].casefold() or query in r['short_name'].casefold()]
    if len(matches) != 1:
        choices = ', '.join(r['short_name'] for r in matches)
        raise ValueError(f'Dataset {query!r}: ' + (f'ambiguous ({choices})' if matches else 'not found; use --list or --search'))
    return matches[0]


class DatasetPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.text, self.title = [], [], []
        self.heading = False
        self.anchor = None
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'):
            self.ignored += 1
        if tag == 'h1':
            self.heading = True
        if tag == 'a' and attrs.get('href'):
            self.anchor = [attrs['href'], []]

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.ignored = max(0, self.ignored-1)
        if tag == 'h1':
            self.heading = False
        if tag == 'a' and self.anchor is not None:
            self.links.append((self.anchor[0], ' '.join(self.anchor[1])))
            self.anchor = None

    def handle_data(self, data):
        if self.ignored:
            return
        self.text.append(data)
        if self.heading:
            self.title.append(data)
        if self.anchor is not None:
            self.anchor[1].append(data)


def is_page_url(query):
    return '://' in query or query.startswith(('lila.science/', 'www.lila.science/'))


def parse_dataset_page(rows, url, html):
    """Match by actual cloud paths, not by assuming page slugs equal catalog names."""
    page = DatasetPage()
    page.feed(html)
    links = [(urljoin(url, href), label) for href, label in page.links]
    text_urls = re.findall(r'(?:s3|gs|https?)://[^\s<>\"\x27]+', ' '.join(page.text))
    links.extend((value.rstrip('),;'), '') for value in text_urls)
    resources = {}
    for value, label in links:
        for provider in TOOLS:
            try:
                native = cloud_url(value, provider)
            except ValueError:
                continue
            resources.setdefault(value, {'label': label})[provider] = native

    # An ENA24-detection page contains ena24/images, a stronger match than its title.
    keys = {info.get('aws', info.get('gcp', info.get('azure'))) for info in resources.values()}
    matches = []
    for row in rows:
        known = set()
        for column, value in row.items():
            if '_url_' not in column or not value:
                continue
            for provider in TOOLS:
                try:
                    known.add(cloud_url(value, provider))
                    break
                except ValueError:
                    pass
        score = len(keys & known)
        if score:
            matches.append((score, row))
    if matches:
        best = max(score for score, _ in matches)
        candidates = [row for score, row in matches if score == best]
        if len(candidates) != 1:
            raise ValueError('Page matches multiple catalog datasets; refused to choose an unrelated image folder')
        result = candidates[0].copy()
    else:
        slug = re.sub(r'[^A-Za-z0-9_-]+', '-', unquote(urlparse(url).path.rstrip('/').split('/')[-1])).strip('-')
        if not slug:
            raise ValueError('Cannot derive dataset directory name from page URL')
        result = {'name': ' '.join(page.title).strip() or slug, 'short_name': slug}

    candidates = {}
    for value, info in resources.items():
        label = (info['label'] + ' ' + value).casefold()
        path = urlparse(value).path.rstrip('/')
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in ('.json', '.zip', '.csv', '.gz'):
            if any(word in label for word in ('bbox', 'bounding')):
                kind = 'bbox'
            elif (any(word in label for word in ('metadata', 'annotation', 'label'))
                  or suffix == '.json' or path.casefold().endswith('.json.zip')):
                if re.search(r'(?:^|[/_.-])(?:splits?|taxonomy|mapping|results?)(?:[_.-]|$)', path.casefold()):
                    continue  # Split definitions and detector outputs are not annotation metadata.
                kind = 'metadata'
            else:
                continue  # Giant image zipfiles are not annotation files.
        elif not suffix and ('images' in label or value.startswith(('s3://', 'gs://'))):
            kind = 'images'
        else:
            continue  # Individual example images are not dataset roots.
        # Prefer species-class boxes over coarse animal/person/vehicle boxes, and
        # official metadata over optional alternate/non-human-only metadata links.
        score = 10 if any(word in label for word in ('with_classes', 'with_species')) else 0
        if any(word in label for word in ('no_classes', 'no_species', 'unofficial', 'non-human', 'non_human')):
            score -= 10
        for provider in TOOLS:
            if provider in info:
                key = (kind, provider)
                if key not in candidates or score > candidates[key][0]:
                    candidates[key] = (score, info[provider])
    for (kind, provider), (_, value) in candidates.items():
        column = 'image_base_url_'+provider if kind == 'images' else f'{kind}_url_{provider}'
        if kind == 'images' and result.get(column):
            continue  # Preserve the catalog root, rather than selecting an example subfolder.
        result[column] = value
    if not any(result.get('image_base_url_'+p) for p in TOOLS):
        raise ValueError('Page has no supported public image folder; it may offer only archives or restricted access')
    result['page_url'] = url
    return result


def resolve_dataset(rows, query, retries=3):
    if not is_page_url(query):
        return select_dataset(rows, query)
    url = query if '://' in query else 'https://' + query
    parsed = urlparse(url)
    if (parsed.scheme not in ('http', 'https') or parsed.hostname not in ('lila.science', 'www.lila.science')
            or not re.fullmatch(r'/datasets/[^/]+/?', parsed.path) or parsed.username or parsed.password):
        raise ValueError('Expected a LILA dataset page URL: https://lila.science/datasets/...')
    url = parsed._replace(query='', fragment='').geturl()
    for attempt in range(retries+1):
        try:
            with urlopen(url, timeout=30) as response:
                html = response.read(4*1024*1024+1)
            if len(html) > 4*1024*1024:
                raise ValueError('Dataset page is unexpectedly large')
            row = parse_dataset_page(rows, url, html.decode('utf-8-sig'))
            print(f'Page identified: {url} -> {row["name"]} ({row["short_name"]})', flush=True)
            return row
        except OSError:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 10))


def cloud_url(value, provider):
    """Convert the catalog's public HTTPS URLs to the native cloud tool URI."""
    parsed = urlparse(value)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('Expected a public cloud URL without credentials, query or fragment')
    # Some catalog cells use another cloud's URL. Public LILA mirrors preserve
    # relative object paths; normalize these known roots to the selected provider.
    aliases = [*PUBLIC_ROOTS.values(), 'https://storage.googleapis.com/public-datasets-lila/',
               'http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/',
               'https://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/']
    for root in aliases:
        if value.startswith(root):
            return (PUBLIC_ROOTS[provider] + value[len(root):]).rstrip('/')
    if provider == 'aws':
        if parsed.scheme == 's3' and parsed.netloc:
            return value.rstrip('/')
        match = re.fullmatch(r'(.+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com', parsed.netloc)
        if parsed.scheme in ('https', 'http') and match:
            return f's3://{match[1]}{parsed.path}'.rstrip('/')
    elif provider == 'gcp':
        if parsed.scheme == 'gs' and parsed.netloc:
            return value.rstrip('/')
        if parsed.scheme == 'https' and parsed.netloc == 'storage.googleapis.com':
            return 'gs://' + parsed.path.lstrip('/').rstrip('/')
    elif provider == 'azure':
        if parsed.scheme == 'https' and parsed.netloc.endswith('.blob.core.windows.net'):
            return value.rstrip('/')
    raise ValueError(f'URL does not match provider {provider}: {value}')


def relative_prefix(value):
    value = value.replace('\\', '/')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or any(c in value for c in '*?%'):
        raise ValueError('--prefix must be a dataset-relative folder without wildcards or ..')
    return '' if str(path) == '.' else str(path)


def build_plan(rows, provider, output, kind='all', prefix='', bbox_url=None):
    plan = []
    for row in rows:
        root = output / row['short_name']
        for item in (('images', 'metadata', 'bbox') if kind == 'all' else (kind,)):
            column = 'image_base_url_' + provider if item == 'images' else f'{item}_url_{provider}'
            url = bbox_url if item == 'bbox' and bbox_url else row.get(column, '')
            if not url:
                if kind != 'all' or item == 'images':
                    raise ValueError(f'{row["short_name"]}: no {item} URL for {provider}')
                detail = ('no separate bbox file; boxes may be embedded in metadata (checked during selection)'
                          if item == 'bbox' else f'no separate {item} file')
                print(f'[{row["short_name"]}] {detail}', flush=True)
                continue
            source = cloud_url(url, provider)
            if item == 'images':
                if prefix:
                    source += '/' + prefix
                target = root / 'images' / prefix
            else:
                filename = unquote(PurePosixPath(urlparse(source).path).name)
                if not filename or filename in ('.', '..') or '/' in filename or '\\' in filename or ':' in filename:
                    raise ValueError(f'Invalid annotation filename: {filename}')
                target = root / 'annotations' / filename
            target = target.resolve()
            if output.resolve() not in target.parents:
                raise ValueError(f'Destination escapes output root: {target}')
            plan.append({'dataset': row['short_name'], 'kind': item, 'source': source, 'target': str(target)})
    targets = [Path(item['target']) for item in plan]
    if len(set(targets)) != len(targets):
        raise ValueError('Multiple download items have the same destination; choose --kind separately')
    return plan


def command_for(item, provider, workers):
    source, target = item['source'], item['target']
    folder = item['kind'] == 'images'
    if provider == 'aws':
        return ['aws', 's3', 'sync' if folder else 'cp', source, target,
                '--no-sign-request', '--region', 'us-west-2']
    if provider == 'azure':
        return ['azcopy', 'copy', source + '/*' if folder else source, target,
                '--recursive=true' if folder else '--recursive=false',
                '--overwrite=ifSourceNewer', '--check-length=true']
    return ['gsutil', '-o', f'GSUtil:parallel_thread_count={workers}',
            '-o', 'GSUtil:parallel_process_count=1', '-m',
            *(['rsync', '-r', source, target] if folder else ['cp', source, target])]


def execute(command, env, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        raise CancelledError('Download cancelled')
    process = subprocess.Popen(command, env=env)  # Argument list, never a shell command.
    try:
        if stop_event is None:
            return process.wait()
        while True:
            if stop_event.is_set():
                raise CancelledError('Download cancelled')
            try:
                return process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    except (KeyboardInterrupt, CancelledError):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def file_signature(path):
    """Validate a completed annotation file before reusing a local checkpoint."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'size': path.stat().st_size, 'sha256': digest.hexdigest()}


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def run_plan(plan, provider, workers=32, retries=3, dry_run=False, *, state_dir=None,
             resume=True, retry_delay=5.0, max_retry_delay=60.0, stop_event=None):
    if workers < 1 or retries < 0 or not all(math.isfinite(v) and v >= 0 for v in (retry_delay, max_retry_delay)):
        raise ValueError('Invalid worker count, retry count or retry delay')
    if not dry_run and shutil.which(TOOLS[provider]) is None:
        install = {'aws': 'python -m pip install awscli', 'azure': 'install AzCopy v10', 'gcp': 'install Google Cloud SDK (gsutil)'}
        raise ValueError(f'{TOOLS[provider]} is not on PATH; {install[provider]}')
    results = []
    # Isolate AWS transfer tuning: do not edit the user's ~/.aws/config or credentials.
    with tempfile.TemporaryDirectory(prefix='lila-transfer-') as temporary:
        config = Path(temporary) / 'aws-config'
        config.write_text(f'[default]\nregion = us-west-2\ns3 =\n'
                          f'    max_concurrent_requests = {workers}\n'
                          '    addressing_style = path\n    preferred_transfer_client = classic\n', encoding='utf-8')
        env = os.environ.copy()
        if provider == 'aws':
            env['AWS_CONFIG_FILE'] = str(config)
            env.pop('AWS_PROFILE', None)
            env.pop('AWS_DEFAULT_PROFILE', None)
        if provider == 'azure':
            env['AZCOPY_CONCURRENCY_VALUE'] = str(workers)
        for index, item in enumerate(plan, 1):
            if stop_event is not None and stop_event.is_set():
                raise CancelledError('Download cancelled')
            command = command_for(item, provider, workers)
            label = f'[{index}/{len(plan)}][{item["dataset"]}/{item["kind"]}]'
            print(f'{label} {shlex.join(command)}', flush=True)
            if dry_run:
                results.append({**item, 'status': 'planned'})
                continue
            target = Path(item['target'])
            (target if item['kind'] == 'images' else target.parent).mkdir(parents=True, exist_ok=True)
            identity = {**item, 'provider': provider}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            checkpoint = Path(state_dir)/f'{key}.json' if state_dir is not None else None
            previous = {}
            if checkpoint is not None and checkpoint.is_file():
                try:
                    previous = json.loads(checkpoint.read_text(encoding='utf-8'))
                    if not isinstance(previous, dict) or any(previous.get(k) != v for k, v in identity.items()):
                        previous = {}
                except (ValueError, OSError):
                    print(f'{label} Cannot read previous checkpoint; cloud tool will compare local files.', flush=True)
            if (resume and previous.get('status') == 'ok' and item['kind'] != 'images'
                    and previous.get('file_signature') is not None
                    and file_signature(target) == previous['file_signature']):
                print(f'{label} Verified completed annotation; skipping download.', flush=True)
                results.append({**item, 'status': 'skipped', 'exit_code': 0})
                print(f'Task progress: {index}/{len(plan)} processed ({index/len(plan):.0%})', flush=True)
                continue
            if item['kind'] == 'images':
                print(f'{label} Recovering by comparing local images with the source; '
                      'completed matching files are kept. Native byte/file/speed progress follows.', flush=True)
            elif previous.get('status') in ('running', 'failed', 'interrupted'):
                print(f'{label} Recovering an unfinished annotation transfer.', flush=True)
            state = {**identity, 'status': 'running', 'attempt': 0}

            def record(status, **updates):
                state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **updates)
                if checkpoint is not None:
                    save_checkpoint(checkpoint, state)

            started = time.monotonic()
            code = 1
            delay = min(retry_delay, max_retry_delay)
            try:
                for attempt in range(retries + 1):
                    record('running', attempt=attempt+1)
                    print(f'{label} Attempt {attempt+1}/{retries+1}', flush=True)
                    code = execute(command, env) if stop_event is None else execute(command, env, stop_event=stop_event)
                    record('ok' if code == 0 else 'failed', exit_code=code)
                    if code == 0:
                        break
                    if attempt < retries:
                        print(f'{label} Exit {code}; retrying in {delay:g}s '
                              '(completed images will be reused).', flush=True)
                        if stop_event is None:
                            time.sleep(delay)
                        elif stop_event.wait(delay):
                            raise CancelledError('Download cancelled')
                        delay = min(delay*2, max_retry_delay)
            except (KeyboardInterrupt, CancelledError):
                record('interrupted')
                raise
            except OSError as exc:
                record('failed', error=str(exc))
                raise
            elapsed = time.monotonic()-started
            if code == 0:
                signature = file_signature(target) if item['kind'] != 'images' else None
                record('ok', elapsed_seconds=elapsed, file_signature=signature)
            results.append({**item, 'status': 'ok' if code == 0 else 'failed', 'exit_code': code,
                            'elapsed_seconds': elapsed})
            print(f'{label} {"Completed" if code == 0 else "Failed"} in {elapsed:.1f}s. '
                  f'Task progress: {index}/{len(plan)} processed ({index/len(plan):.0%}).', flush=True)
    return results


def dataset_inputs(arguments, urls_file=None):
    inputs = list(arguments)
    if urls_file is not None:
        text = urls_file.read_text(encoding='utf-8-sig')
        if urls_file.suffix.casefold() == '.json':
            values = json.loads(text)
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError('--urls-file JSON must be a list of non-empty URL strings')
        else:
            values = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        inputs.extend(values)
    return list(dict.fromkeys(value.strip() for value in inputs if value.strip()))


def run_datasets(plan, provider, *, mode='serial', dataset_workers=2, image_policy='all',
                 negative_fraction=0.05, seed=42, prefix='', audit_only=False, **options):
    """Schedule one native-tool sequence per dataset, with independent checkpoints."""
    if mode not in ('serial', 'parallel') or dataset_workers < 1:
        raise ValueError('Invalid mode or dataset worker count')
    if image_policy == 'boxed' and any(item['kind'] == 'images' for item in plan):
        from lila_selection import run_boxed_datasets
        return run_boxed_datasets(plan, provider, mode=mode, dataset_workers=dataset_workers,
                                  negative_fraction=negative_fraction, seed=seed, prefix=prefix,
                                  audit_only=audit_only, run_annotations=run_datasets, options=options)
    groups = {}
    for item in plan:
        groups.setdefault(item['dataset'], []).append(item)
    stop = Event()
    results, completed = [], 0

    def run_group(name, items):
        print(f'[Dataset {name}] Starting ({mode})', flush=True)
        try:
            return run_plan(items, provider, stop_event=stop, **options)
        except CancelledError:
            raise
        except (OSError, ValueError, ImportError) as exc:
            print(f'[Dataset {name}] Failed: {exc}', flush=True)
            return [{**item, 'status': 'failed', 'error': str(exc), 'exit_code': -1} for item in items]

    def collect(name, items):
        nonlocal completed
        results.extend(items)
        completed += 1
        failed = any(r['status'] == 'failed' for r in items)
        print(f'[Dataset {name}] {"Failed" if failed else "Completed"}. '
              f'Dataset progress: {completed}/{len(groups)} ({completed/len(groups):.0%})', flush=True)

    concurrency = min(dataset_workers, len(groups)) if mode == 'parallel' else min(1, len(groups))
    print(f'Datasets: {len(groups)}; mode: {mode}; simultaneous datasets: {concurrency}; '
          f'transfer workers per dataset: {options.get("workers", 32)}', flush=True)
    if mode == 'serial':
        try:
            for name, items in groups.items():
                collect(name, run_group(name, items))
        except KeyboardInterrupt:
            stop.set()
            raise
        return results
    executor = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = {}
    try:
        for name, items in groups.items():
            futures[executor.submit(run_group, name, items)] = name
        for future in as_completed(futures):
            collect(futures[future], future.result())
    except KeyboardInterrupt:
        stop.set()
        for future in futures:
            future.cancel()
        print('Stopping active transfers and cancelling queued datasets; progress is preserved.', flush=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    # Keep reports stable even when datasets finish out of order.
    fields = ('dataset', 'kind', 'source', 'target')
    order = {tuple(item[k] for k in fields): i for i, item in enumerate(plan)}
    results.sort(key=lambda r: order[tuple(r[k] for k in fields)])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', help='LILA dataset page URLs, or catalog names/keywords')
    parser.add_argument('--urls-file', type=Path, help='UTF-8 text file (one URL per line) or JSON array of URLs')
    parser.add_argument('--mode', choices=['serial', 'parallel'], default='serial', help='Dataset scheduling mode (default: serial)')
    parser.add_argument('--dataset-workers', type=int, default=2, help='Simultaneous datasets in parallel mode (default: 2)')
    parser.add_argument('--provider', choices=TOOLS, default='aws')
    parser.add_argument('--output', type=Path, default=Path('datasets/lila'), help='Root directory; dataset subdirectories are added')
    parser.add_argument('--kind', choices=['images', 'metadata', 'bbox', 'all'], default='all')
    parser.add_argument('--image-policy', choices=['boxed', 'all'], default='boxed',
                        help='Default: valid boxed images plus confirmed-empty negatives; all restores full-folder downloads')
    parser.add_argument('--negative-fraction', type=float, default=0.05,
                        help='Target negative share of selected images across the batch (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Reproducible negative sampling seed')
    parser.add_argument('--audit-only', action='store_true', help='Download annotations and write selection reports, without downloading images')
    parser.add_argument('--workers', type=int, default=32, help='Image transfer concurrency inside each dataset')
    parser.add_argument('--retries', type=int, default=3, help='Extra retries per transfer')
    parser.add_argument('--retry-delay', type=float, default=5.0, help='Initial retry delay in seconds (doubles after failure)')
    parser.add_argument('--max-retry-delay', type=float, default=60.0, help='Maximum retry delay in seconds')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                        help='Reuse completed annotation and image checkpoints (default: enabled)')
    parser.add_argument('--prefix', default='', help='Download one image subfolder, preserving its relative path')
    parser.add_argument('--bbox-url', help='Override bounding-box URL for one dataset, using the selected provider')
    parser.add_argument('--list', action='store_true', help='List catalog entries')
    parser.add_argument('--search', help='List entries matching a keyword')
    parser.add_argument('--info', action='store_true', help='Print selected dataset catalog records without downloading')
    parser.add_argument('--catalog', type=Path, help='Local CSV in official catalog format, including custom/new entries')
    parser.add_argument('--refresh', action='store_true', help='Refresh cached official catalog')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running cloud tools')
    args = parser.parse_args()
    try:
        if args.workers < 1 or args.dataset_workers < 1 or args.retries < 0:
            raise ValueError('--workers and --dataset-workers must be >= 1; --retries must be >= 0')
        if not all(math.isfinite(v) and v >= 0 for v in (args.retry_delay, args.max_retry_delay)):
            raise ValueError('Retry delays must be finite and >= 0')
        if not math.isfinite(args.negative_fraction) or not 0 <= args.negative_fraction < 1:
            raise ValueError('--negative-fraction must be in [0, 1)')
        if args.audit_only and (args.image_policy != 'boxed' or args.kind not in ('images', 'all')):
            raise ValueError('--audit-only requires --image-policy boxed and --kind images or all')
        output = args.output.expanduser().resolve()
        inputs = dataset_inputs(args.datasets, args.urls_file)
        try:
            rows = load_catalog(args.catalog, output / '.catalog.csv', args.refresh, args.retries)
        except OSError:
            if args.catalog is not None or args.list or args.search or not inputs or not all(is_page_url(q) for q in inputs):
                raise
            print('Catalog unavailable; resolving download sources directly from dataset pages.', flush=True)
            rows = []
        if args.list or args.search:
            query = (args.search or '').casefold()
            for row in rows:
                if query in row['name'].casefold() or query in row['short_name'].casefold():
                    providers = ','.join(p for p in TOOLS if row.get('image_base_url_'+p))
                    print(f'{row["short_name"]:42s} {row["name"]} [{providers}]')
            return
        if not inputs:
            parser.error('Specify dataset URLs, --urls-file, or use --list / --search')
        selected = {row['short_name']: row for row in (resolve_dataset(rows, query, args.retries) for query in inputs)}
        if args.info:
            print(json.dumps(list(selected.values()), indent=2, ensure_ascii=False))
            return
        if args.bbox_url and (len(selected) != 1 or (args.kind not in ('bbox', 'all') and args.image_policy != 'boxed')):
            raise ValueError('--bbox-url requires one dataset and --kind bbox or all')
        plan_kind = 'all' if args.image_policy == 'boxed' and args.kind == 'images' else args.kind
        plan = build_plan(list(selected.values()), args.provider, output, plan_kind,
                          relative_prefix(args.prefix), args.bbox_url)
        if args.image_policy == 'boxed' and args.kind in ('images', 'all') and not args.dry_run:
            try:
                import ijson, requests, tqdm, PIL  # noqa: F401
            except ImportError as exc:
                raise ValueError('Install selection dependencies: python -m pip install -r requirements-lila.txt') from exc
        results = run_datasets(plan, args.provider, mode=args.mode, dataset_workers=args.dataset_workers,
                               image_policy=args.image_policy, negative_fraction=args.negative_fraction,
                               seed=args.seed, prefix=relative_prefix(args.prefix), audit_only=args.audit_only,
                               workers=args.workers, retries=args.retries, dry_run=args.dry_run,
                               state_dir=output/'.transfer-state', resume=args.resume and not args.refresh,
                               retry_delay=args.retry_delay, max_retry_delay=args.max_retry_delay)
        failures = [r for r in results if r['status'] == 'failed']
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            (output/'lila_download_report.json').write_text(json.dumps(
                {'provider': args.provider, 'results': results}, indent=2, ensure_ascii=False), encoding='utf-8')
        skipped = sum(r['status'] == 'skipped' for r in results)
        print(f'Finished: {len(results)-len(failures)-skipped} successful/planned, '
              f'{skipped} verified/skipped, {len(failures)} failed.')
        if failures:
            parser.exit(1, 'Some transfers failed; rerun the command to recover missing images.\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nDownload interrupted; rerun the same command to recover missing images.\n')
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
