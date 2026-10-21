"""Download official YOLOv8 / YOLO11 / YOLO26 detection checkpoints without training."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

MODELS = tuple(f'{family}{size}.pt' for family in ('yolov8', 'yolo11', 'yolo26') for size in 'nsmlx')


def checkpoint_name(value):
    if value not in MODELS:
        hint = ' YOLOv8 filenames start with yolov8, e.g. yolov8n.pt.' if value.startswith('yolo8') else ''
        raise argparse.ArgumentTypeError(f'Unsupported detection checkpoint: {value}.{hint}')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_names', nargs='*', type=checkpoint_name,
                        help='Checkpoint names, e.g. yolov8n.pt yolov8s.pt')
    parser.add_argument('--models', nargs='+', choices=MODELS,
                        type=checkpoint_name, help='Alternative to positional checkpoint names')
    parser.add_argument('--output', type=Path, default=Path('weights'), help='Destination directory (default: weights)')
    parser.add_argument('--retries', type=int, default=3, help='Ultralytics download retries (default: 3)')
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error('--retries must be >= 0')
    args.models = list(dict.fromkeys(args.model_names+(args.models or []))) or ['yolo11s.pt', 'yolo26s.pt']
    return parser, args


def main():
    parser, args = parse_args()
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError:
        parser.exit(1, 'Install dependency first: python -m pip install --upgrade ultralytics\n')
    output = args.output.expanduser().resolve()
    failures = []
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name in dict.fromkeys(args.models):
            target = output / name
            try:
                if target.exists():
                    if not target.is_file() or target.stat().st_size < 100_000:
                        raise ValueError(f'Existing checkpoint is too small or not a file: {target}; '
                                         'check it and remove it manually before retrying')
                    print(f'Already exists: {target}', flush=True)
                    continue
                print(f'Downloading {name} to {output}', flush=True)
                downloaded = Path(attempt_download_asset(
                    str(target), retry=args.retries, unzip=False,
                )).resolve()
                if not downloaded.is_file() or downloaded.stat().st_size < 100_000:
                    raise RuntimeError('Download did not produce a checkpoint; check network and Ultralytics version')
                if downloaded != target:
                    shutil.copy2(downloaded, target)
                print(f'Saved: {target} ({target.stat().st_size / 1024**2:.1f} MiB)', flush=True)
            except Exception as exc:
                failures.append(name)
                print(f'Failed: {name}: {exc}', flush=True)
    except KeyboardInterrupt:
        parser.exit(130, '\nDownload interrupted.\n')
    except OSError as exc:
        parser.exit(1, f'Cannot prepare output directory: {exc}\n')
    if failures:
        parser.exit(1, f'Failed checkpoints: {", ".join(failures)}\n')


if __name__ == '__main__':
    main()
