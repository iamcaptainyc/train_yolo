"""Recursively find decoder warnings and repair shared hardlinks and remove undecodable image/label pairs."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
import signal
from pathlib import Path
import sys
import subprocess
import io
import tempfile
import time

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
_CAPTURE = None


def decode(path):
    """Capture native libjpeg stderr inside an isolated, single-threaded worker."""
    import cv2
    if os.name == 'nt':
        # Native OpenCV on Windows may use a different CRT from Python's dup2.
        # An OS-created child stderr pipe captures that CRT reliably.
        import numpy as np
        code = ("import cv2,numpy as np,sys; cv2.setNumThreads(1); "
                "a=cv2.imread(sys.argv[1],cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION); "
                "sys.exit(2) if a is None else np.save(sys.stdout.buffer,a,allow_pickle=False)")
        child = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(path)], capture_output=True)
        warning = child.stderr.decode('utf-8', errors='replace').strip()
        if child.returncode not in (0, 2):
            raise RuntimeError(f'Decoder subprocess failed: {warning}')
        pixels = np.load(io.BytesIO(child.stdout), allow_pickle=False) if child.returncode == 0 else None
        return pixels, warning
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = tempfile.TemporaryFile()
        cv2.setNumThreads(1)
    _CAPTURE.seek(0)
    _CAPTURE.truncate()
    saved = os.dup(2)
    try:
        os.dup2(_CAPTURE.fileno(), 2)
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    finally:
        # Flush native C stdio before restoring stderr (notably buffered on Windows).
        import ctypes
        ctypes.CDLL('ucrtbase' if os.name == 'nt' else None).fflush(None)
        os.dup2(saved, 2)
        os.close(saved)
    _CAPTURE.seek(0)
    return pixels, _CAPTURE.read().decode('utf-8', errors='replace').strip()


def initialize_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def inspect_image(path, check_only=False, operations=None):
    from filelock import FileLock
    from repair_operations import journal_for, finish
    path = Path(path).absolute()
    operations = Path(operations) if operations else path.parent/'.repair-operations'
    journal = journal_for(operations, path)
    if journal.exists() and not check_only:
        op = json.loads(journal.read_text(encoding='utf-8'))
        if not op.get('done') or (op['action'] == 'delete' and not path.exists()):
            return finish(journal)
    try:
        info = path.stat()
        # Bounded shared locks serialize hardlinked aliases even in different roots.
        stripe = (info.st_dev ^ info.st_ino) % 64
        lock = Path(tempfile.gettempdir())/f'yolo-image-repair-{stripe}.lock'
        with FileLock(str(lock)):
            return _inspect_image(path, check_only, operations)
    except Exception as exc:
        return {'path': str(path), 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}


def _inspect_image(path, check_only, operations):
    """Healthy images are untouched; undecodable image/label pairs are deleted."""
    import cv2
    from PIL import Image
    from repair_operations import mutate
    path = Path(path)
    temporary = None
    result = {'path': str(path), 'status': 'failed'}
    try:
        if path.is_symlink():
            return {**result, 'reason': 'Symbolic links are not modified; scan the real image directory'}
        before = signature(path)
        with path.open('rb') as stream:
            stream.read(1)  # Permission/I/O failure is not evidence of corrupt pixels.
        pixels, warning = decode(path)
        if pixels is None:
            result.update(reason=warning or 'Decoder returned no image', status='detected' if check_only else 'deleted')
            if check_only:
                return result
            if signature(path) != before:
                raise ValueError('Source changed during decode')
            return mutate(path, operations, result)
        # Pillow load catches additional truncation failures missed by permissive OpenCV.
        try:
            with Image.open(path) as im:
                im.load()
        except Exception as exc:
            warning = (warning + '; ' if warning else '') + f'Pillow: {exc}'
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            with path.open('rb') as stream:
                stream.seek(-2, 2)
                if stream.read() != b'\xff\xd9':
                    warning = (warning + '; ' if warning else '') + 'Missing JPEG end marker'
        if not warning:
            return {'status': 'healthy'}
        result['reason'] = warning
        result['width'], result['height'] = pixels.shape[1], pixels.shape[0]
        result['review_content'] = True
        if check_only:
            return {**result, 'status': 'detected'}
        if path.suffix.lower() not in {'.jpg', '.jpeg'}:
            return {**result, 'reason': warning + '; automatic repair is restricted to JPEG'}
        # Validate staged JPEG before writing back to the original inode.
        descriptor, name = tempfile.mkstemp(prefix=f'.{path.stem}.repair-', suffix='.jpg', dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError('JPEG encoder failed')
        with temporary.open('wb') as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        repaired, repair_warning = decode(temporary)
        if repaired is None or repair_warning or repaired.shape != pixels.shape:
            raise ValueError(f'Repair did not pass decoder/dimension verification: {repair_warning}')
        with Image.open(temporary) as im:
            im.load()
        if signature(path) != before:
            raise ValueError('Source changed during scan; stop training/other writers and retry')
        result = mutate(path, operations, {**result, 'status': 'repaired', 'hardlinks': path.stat().st_nlink}, temporary)
        temporary = None
        return result
    except Exception as exc:
        return {**result, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
    finally:
        if temporary is not None:
            from repair_operations import journal_for
            journal = journal_for(operations, path)
            op = json.loads(journal.read_text(encoding='utf-8')) if journal.exists() else {}
            if op.get('stage') != str(temporary) or op.get('done'):
                temporary.unlink(missing_ok=True)


def discover(roots):
    paths = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'Image directory does not exist: {root}')
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory)/name
                if path.suffix.lower() in SUFFIXES and '.repair-' not in name:
                    paths.add(str(path))
    return sorted(paths)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def report(output, state, finished):
    stats = state['counts']
    lines = ['# 图片检查与修复报告', '', '**状态：扫描完成**' if finished else '**状态：已中断，可继续**', '',
             f"图片总数：{state['total']}；已处理：{state['next_index']}。", '',
             f"已删除标签：{state.get('deleted_labels', 0)}。", '',
             '| 结果 | 图片数 |', '| --- | --- |']
    for key, label in [('healthy', '正常，未改写'), ('repaired', '重新编码并验证通过'),
                       ('deleted', '无法解码，已删除图片及对应标签'), ('detected', '仅检查模式发现问题'), ('failed', '无法修复，原文件保留')]:
        lines.append(f'| {label} | {stats.get(key, 0)} |')
    lines += ['', '完整问题路径、原始警告、修复状态见 [issues.jsonl](issues.jsonl)，每行一个 JSON。', '',
              '正常图片不保存逐图记录。断点只保存扫描进度与累计数量，不计算图片内容哈希。', '',
              'JPEG 修复采用重新编码，保持解码后的原始宽高，不旋转、不缩放、不改标签；会移除 EXIF 元数据。',
              '修复仅证明文件能通过解码检查，不能恢复原先丢失的画面。所有修复图均标记 review_content=true，建议抽查。',
              '无法解码的图片自动删除对应路径及 YOLO 标签；其他操作失败会保留错误记录。',
              '修复写回原 inode，所有硬链接同步更新；暂存修复文件及问题操作日志支持中断恢复。',
              '运行前应停止训练；训练器可能保留旧标签缓存，修复后若仍显示旧警告，确认当前路径并重建对应缓存。', '']
    if state.get('report_update_error'):
        lines += ['数据集报告更新失败（重跑同一命令会重试，无需重新解码已扫描图片）：' + state['report_update_error'], '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


from repair_dataset_report import refresh_dataset_reports


def scan(args):
    from tqdm import tqdm
    roots = sorted(set(p.expanduser().resolve() for p in args.roots))
    print('Collecting image paths (no content hashing)...', flush=True)
    from repair_operations import recover, deleted_paths
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    operations = output/'operations'
    if not args.check_only:
        recover(operations)
    checkpoint = output/'checkpoint.json'
    state = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
    paths = discover(roots)
    if state and not args.restart:
        paths = sorted(set(paths) | set(deleted_paths(operations)))
    identity = {'roots': [str(p) for p in roots], 'check_only': args.check_only,
                'path_list_sha256': hashlib.sha256('\n'.join(paths).encode()).hexdigest(), 'version': 2}
    if state and args.restart:
        if operations.exists():
            operations.rename(output/f'operations.previous-{time.time_ns()}')
        state = None
    if state and state['identity'] != identity:
        raise ValueError('Image paths/mode changed. Use --restart to scan again, or a different --output.')
    if state is None:
        state = {'identity': identity, 'total': len(paths), 'next_index': 0, 'counts': {}, 'issues_bytes': 0}
        (output/'issues.jsonl').write_bytes(b'')
        save_json(checkpoint, state)
    def update_reports(finished):
        try:
            refresh_dataset_reports(roots, operations, args.check_only)
            state.pop('report_update_error', None)
        except Exception as exc:
            state['report_update_error'] = f'{type(exc).__name__}: {exc}'
            print(f'Dataset report update failed: {exc}', file=sys.stderr)
        save_json(checkpoint, state)
        report(output, state, finished)

    if state['next_index'] == len(paths):
        update_reports(True)
        print(f'Already complete: {output / "REPORT.md"}; use --restart to check modified images again.')
        return 2 if state['counts'].get('failed') or state.get('report_update_error') else 0
    counts = Counter(state['counts'])
    deleted_label_count = state.get('deleted_labels', 0)
    # Keep a bounded queue; do not allocate a Future or a record for every image.
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
    pending = deque()
    cursor = state['next_index']
    interrupted = False
    with (output/'issues.jsonl').open('r+b') as issues, tqdm(total=len(paths), initial=cursor, unit='img', desc='Check / repair') as progress:
        issues.truncate(state['issues_bytes'])
        issues.seek(0, 2)
        def checkpoint_now():
            issues.flush()
            os.fsync(issues.fileno())
            state['counts'] = dict(counts)
            state['deleted_labels'] = deleted_label_count
            state['issues_bytes'] = issues.tell()
            save_json(checkpoint, state)
        try:
            while cursor < len(paths) or pending:
                while cursor < len(paths) and len(pending) < args.workers * 2:
                    pending.append((cursor, pool.submit(inspect_image, paths[cursor], args.check_only, operations)))
                    cursor += 1
                index, future = pending[0]
                item = future.result()
                pending.popleft()
                counts[item['status']] += 1
                deleted_label_count += item['status'] == 'deleted' and item.get('label_deleted', False)
                if item['status'] != 'healthy':
                    issues.write((json.dumps(item, ensure_ascii=False) + '\n').encode('utf-8'))
                state['next_index'] = index + 1
                progress.update()
                if state['next_index'] % 64 == 0:
                    checkpoint_now()
                    progress.set_postfix(repaired=counts['repaired'], failed=counts['failed'])
        except KeyboardInterrupt:
            interrupted = True
            print('\nStopping after queued images; saving progress...', flush=True)
            # Drain already submitted work so repairs retain their issue records.
            for index, future in pending:
                item = future.result()
                counts[item['status']] += 1
                deleted_label_count += item['status'] == 'deleted' and item.get('label_deleted', False)
                if item['status'] != 'healthy':
                    issues.write((json.dumps(item, ensure_ascii=False) + '\n').encode('utf-8'))
                state['next_index'] = index + 1
                progress.update()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            checkpoint_now()
            update_reports(state['next_index'] == len(paths))
    print(f'Report: {output / "REPORT.md"}', flush=True)
    return 130 if interrupted else (2 if counts['failed'] or state.get('report_update_error') else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path, help='One or more image/dataset directories; recursively scanned')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('runs/image_repair'))
    parser.add_argument('--check-only', action='store_true', help='Locate problems without modifying images')
    parser.add_argument('--restart', action='store_true', help='Start over; replaces this output directory\'s previous report/issue list')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        import cv2
        from filelock import FileLock
        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output/'.repair.lock'), timeout=0):
            return scan(args)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
