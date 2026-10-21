"""Download a Hugging Face dataset repository directly into a local directory."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import time


def download(repo_id, output, *, workers=8, retries=3, revision='main', dry_run=False):
    from huggingface_hub import snapshot_download

    transport_errors = [TimeoutError, ConnectionError]
    for module in ('httpx', 'httpx2'):
        try:
            transport_errors.append(importlib.import_module(module).TransportError)
        except ImportError:
            pass

    if workers < 1 or retries < 0:
        raise ValueError('workers must be >= 1; retries must be >= 0')
    for attempt in range(retries+1):
        try:
            return snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=str(output),
                                     revision=revision, max_workers=workers, dry_run=dry_run)
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            transient = isinstance(exc, tuple(transport_errors)) or (
                status is not None and (status in (408, 429) or status >= 500))
            if not transient or attempt == retries:
                raise
            delay = min(5*2**attempt, 60)
            print(f'Temporary download error ({type(exc).__name__}); retrying in {delay}s. '
                  'Existing files are retained.', flush=True)
            time.sleep(delay)


def write_snake_config(output):
    """Preserve the upstream YAML and write corrected paths for this YOLO export."""
    import yaml
    output = Path(output).resolve()
    config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf-8'))
    for split in ('train', 'valid', 'test'):
        if not (output/split/'images').is_dir() or not (output/split/'labels').is_dir():
            raise ValueError(f'Missing YOLO image/label directory: {output/split}')
    if not config.get('names') or config.get('nc') != len(config['names']):
        raise ValueError('Invalid class configuration in upstream data.yaml')
    config.update(path=output.as_posix(), train='train/images', val='valid/images', test='test/images')
    target = output/'data.local.yaml'
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo_id', nargs='?', default='the-khiem7/snakeaid-yolov12-5000-bbox')
    p.add_argument('--output', type=Path, help='Default: datasets/huggingface/<owner>/<dataset>')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--retries', type=int, default=3, help='Extra retries for temporary network/server errors')
    p.add_argument('--revision', default='main', help='Branch, tag or full commit SHA')
    p.add_argument('--dry-run', action='store_true', help='List pending files and total bytes without downloading them')
    p.add_argument('--snake-yolo-config', action='store_true', help='Write data.local.yaml for the SnakeAid YOLO export')
    args = p.parse_args()
    if args.workers < 1 or args.retries < 0:
        p.error('--workers must be >= 1 and --retries must be >= 0')
    if args.repo_id.count('/') != 1 or any(part in ('', '.', '..') for part in args.repo_id.split('/')):
        p.error('repo_id must be owner/dataset')
    output = (args.output or Path('datasets/huggingface')/args.repo_id).expanduser().resolve()
    try:
        result = download(args.repo_id, output, workers=args.workers, retries=args.retries,
                          revision=args.revision, dry_run=args.dry_run)
        if args.dry_run:
            pending = [r for r in result if r.will_download]
            print(f'{len(pending)} pending files; {sum(r.file_size for r in pending):,} bytes to download.')
        else:
            print(f'Downloaded dataset: {output}')
            if args.snake_yolo_config:
                print(f'Training YAML: {write_snake_config(output)}')
    except KeyboardInterrupt:
        p.exit(130, '\nInterrupted. Keep the output directory and rerun to continue.\n')
    except ImportError as exc:
        p.exit(1, f'Missing dependency: {exc}. Install: pip install -r requirements-huggingface.txt\n')
    except Exception as exc:
        p.exit(1, f'Download/configuration failed ({type(exc).__name__}): {exc}\n'
                  'Existing downloads are retained. Resolve the error and rerun with the same output.\n')


if __name__ == '__main__':
    main()
