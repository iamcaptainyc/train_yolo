"""Select valid boxed images and confirmed-empty negatives from COCO Camera Traps."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import sqlite3
from threading import local
import time
from urllib.parse import quote, urlparse
import zipfile

EMPTY_NAMES = {'empty', 'blank', 'background', 'no animal', 'no animals', 'none'}
EXCLUDED_NAMES = {'human', 'humans', 'person', 'people', 'homo sapiens', 'ignore',
                  'vehicle', 'vehicles', 'car', 'cars', 'blurred', 'problem', 'unlabeled', 'unlabelled'}


def normalized_name(value):
    return ' '.join(str(value).casefold().replace('_', ' ').replace('-', ' ').split())


def json_path(path):
    """Cache one JSON member without trusting ZIP extraction paths or loading it in RAM."""
    path = Path(path)
    if path.suffix.lower() != '.zip':
        return path
    stamp = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    cached = path.parent/'.json-cache'/(path.name+'.json')
    marker = cached.with_suffix('.source.json')
    if cached.is_file() and marker.is_file():
        try:
            if json.loads(marker.read_text()) == stamp:
                return cached
        except ValueError:
            pass
    cached.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as bundle:
        names = [n for n in bundle.namelist() if n.lower().endswith('.json') and not n.startswith('__MACOSX/')]
        if len(names) != 1:
            raise ValueError(f'Expected one COCO JSON in {path}, found {len(names)}')
        temporary = cached.with_suffix('.tmp')
        with bundle.open(names[0]) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, 1024*1024)
        temporary.replace(cached)
    marker.write_text(json.dumps(stamp), encoding='utf-8')
    return cached


def items(path, key):
    import ijson
    with Path(path).open('rb') as stream:
        yield from ijson.items(stream, key+'.item', use_float=True)


def safe_name(value):
    name = str(value).replace('\\', '/')
    parts = PurePosixPath(name)
    if not name or parts.is_absolute() or '..' in parts.parts or ':' in name:
        raise ValueError(f'Unsafe image filename in metadata: {value!r}')
    return str(parts)


def usable_image(image, prefix=''):
    name = safe_name(image['file_name'])
    if name.casefold().startswith('private/') or image.get('corrupt') or image.get('is_corrupt'):
        return False
    return not prefix or name == prefix or name.startswith(prefix.rstrip('/')+'/')


def valid_box(box, image=None):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        return False
    x, y, width, height = box
    if min(x, y) < -1e-6 or width <= 0 or height <= 0:
        return False
    if image:
        iw, ih = image.get('width'), image.get('height')
        if iw and ih and (x+width > iw+1 or y+height > ih+1):
            return False
    return True


def select_images(metadata_file, bbox_file=None, *, negative_fraction=0.05, seed=42, prefix='', candidate_file=None):
    """Never treat missing bbox or missing annotations as evidence of an empty scene."""
    if not math.isfinite(negative_fraction) or not 0 <= negative_fraction < 1:
        raise ValueError('--negative-fraction must be in [0, 1)')
    metadata = json_path(metadata_file)
    boxes_path = json_path(bbox_file) if bbox_file else metadata
    print(f'Scanning COCO categories and annotations: {metadata.name}', flush=True)
    meta_categories = {str(c['id']): c for c in items(metadata, 'categories')}
    box_categories = {str(c['id']): c for c in items(boxes_path, 'categories')} or meta_categories
    if not meta_categories or not box_categories:
        raise ValueError('Missing COCO categories; cannot safely identify empty or excluded images')
    empty_meta = {key for key, cat in meta_categories.items() if normalized_name(cat['name']) in EMPTY_NAMES}
    excluded_meta = {key for key, cat in meta_categories.items() if normalized_name(cat['name']) in EXCLUDED_NAMES}
    excluded_boxes = {key for key, cat in box_categories.items()
                      if normalized_name(cat['name']) in EMPTY_NAMES | EXCLUDED_NAMES}
    empty_box_categories = {key for key, cat in box_categories.items() if normalized_name(cat['name']) in EMPTY_NAMES}
    empty_ids, excluded_ids, invalid_ids, raw_bbox_ids = set(), set(), set(), set()
    boxes, stats = defaultdict(list), Counter()
    # First collect explicit empty labels, then remove contradictory non-empty labels
    # in a second pass so annotation ordering cannot create false negatives.
    for ann in items(metadata, 'annotations'):
        key, category = str(ann.get('image_id')), str(ann.get('category_id'))
        stats['metadata_annotations'] += 1
        if category in empty_meta:
            empty_ids.add(key)
        if category in excluded_meta:
            excluded_ids.add(key)
    for ann in items(metadata, 'annotations'):
        if str(ann.get('category_id')) not in empty_meta:
            empty_ids.discard(str(ann.get('image_id')))
    for ann in items(boxes_path, 'annotations'):
        if 'bbox' not in ann:
            continue
        stats['bbox_annotations_present'] += 1
        key, category = str(ann.get('image_id')), str(ann.get('category_id'))
        if category in empty_box_categories and not valid_box(ann['bbox']):
            continue  # Some exports use bbox=[] or a zero rectangle for a confirmed empty image.
        raw_bbox_ids.add(key)
        empty_ids.discard(key)
        if category in excluded_boxes:
            excluded_ids.add(key)
            continue
        if category not in box_categories or not valid_box(ann['bbox']):
            invalid_ids.add(key)
            stats['invalid_boxes'] += 1
            continue
        boxes[key].append(ann)
    if not boxes:
        raise ValueError('No valid non-human/non-ignore bounding boxes found; refusing full-image fallback')
    positives, negatives, blocked = {}, {}, set()
    seen_box_ids = set()

    def add_positive(image):
        key = str(image.get('id'))
        if key not in boxes:
            return
        seen_box_ids.add(key)
        name = safe_name(image['file_name'])
        if key in excluded_ids or key in invalid_ids or not usable_image(image, prefix):
            blocked.add(name)
            return
        if not all(valid_box(ann['bbox'], image) for ann in boxes[key]):
            blocked.add(name)
            stats['images_with_out_of_bounds_boxes'] += 1
            return
        if name not in positives:
            positives[name] = {'image': image, 'annotations': [], 'role': 'positive'}
        positives[name]['annotations'].extend(boxes[key])

    print(f'Matching {len(boxes):,} boxed image IDs to image paths...', flush=True)
    if boxes_path != metadata:
        for image in items(boxes_path, 'images'):
            add_positive(image)
    for image in items(metadata, 'images'):
        stats['metadata_images'] += 1
        key = str(image.get('id'))
        name = safe_name(image['file_name'])
        stats['metadata_images_with_bbox_record' if key in raw_bbox_ids else 'metadata_images_without_bbox_record'] += 1
        was_seen = key in seen_box_ids
        if key in boxes:
            seen_box_ids.add(key)
        if key in excluded_ids or not usable_image(image, prefix):
            blocked.add(name)
            stats['excluded_or_unusable_images'] += 1
            continue
        if boxes_path == metadata or not was_seen:
            add_positive(image)
        if key in empty_ids and key not in boxes and key not in invalid_ids:
            negatives.setdefault(name, {'image': image, 'annotations': [], 'role': 'negative'})
    for name in blocked:
        positives.pop(name, None)
        negatives.pop(name, None)
    for name in positives:
        negatives.pop(name, None)
    if not positives:
        raise ValueError('No usable boxed images remain after exclusions and prefix filtering')
    requested = math.floor(len(positives)*negative_fraction/(1-negative_fraction)+0.5)
    if candidate_file is not None:
        candidate_file = Path(candidate_file)
        candidate_file.parent.mkdir(parents=True, exist_ok=True)
        with candidate_file.open('w', encoding='utf-8') as stream:
            for name, record in sorted(negatives.items()):
                stream.write(json.dumps({'file_name': name, **record})+'\n')
    sampled = random.Random(seed).sample(sorted(negatives), min(requested, len(negatives)))
    selected = {**positives, **{name: negatives[name] for name in sampled}}
    report = {**stats, 'positive_images': len(positives), 'confirmed_empty_candidates': len(negatives),
              'requested_negatives': requested, 'negative_images': len(sampled),
              'negative_shortfall': max(0, requested-len(sampled)),
              'selected_images': len(selected), 'actual_negative_fraction': len(sampled)/len(selected),
              'requested_negative_fraction': negative_fraction, 'seed': seed,
              'boxed_ids_without_image_record': len(set(boxes)-seen_box_ids),
              'metadata_file': str(Path(metadata_file).resolve()),
              'bbox_file': str(Path(bbox_file or metadata_file).resolve()),
              'categories': list(box_categories.values())}
    return dict(sorted(selected.items())), report


def write_selection(selected, report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for role, filename in [(None, 'selected_images.txt'), ('positive', 'positives.txt'), ('negative', 'negatives.txt')]:
        paths = [name for name, record in selected.items() if role is None or record['role'] == role]
        (directory/filename).write_text(''.join(name+'\n' for name in paths), encoding='utf-8')
    (directory/'selection_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (directory/'selected_images.jsonl').open('w', encoding='utf-8') as stream:
        for name, record in selected.items():
            stream.write(json.dumps({'file_name': name, **record}, ensure_ascii=False)+'\n')
    # Standard COCO subset: negatives have image entries and zero object annotations.
    with (directory/'selected_annotations.json').open('w', encoding='utf-8') as stream:
        stream.write('{"categories":'+json.dumps(report['categories'])+',"images":[')
        for index, record in enumerate(selected.values()):
            image = {**record['image'], 'source_image_id': record['image'].get('id'),
                     'id': index, 'selection_role': record['role']}
            stream.write((',' if index else '')+json.dumps(image))
        stream.write('],"annotations":[')
        index = 0
        for image_id, record in enumerate(selected.values()):
            for ann in record['annotations']:
                stream.write((',' if index else '')+json.dumps({**ann, 'id': index, 'image_id': image_id}))
                index += 1
        stream.write(']}')


def load_selection(directory):
    selected = {}
    with (directory/'selected_images.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            selected[row.pop('file_name')] = row
    return selected


def rebalance_negatives(prepared, fraction, seed):
    """Backfill a dataset with no empties from other datasets' verified empty pools."""
    positives = sum(p['report']['positive_images'] for p in prepared)
    requested = math.floor(positives*fraction/(1-fraction)+0.5)
    available = sum(p['report']['confirmed_empty_candidates'] for p in prepared)
    target = min(requested, available)
    quotas = [p['report']['negative_images'] for p in prepared]
    capacities = [p['report']['confirmed_empty_candidates'] for p in prepared]
    delta = target-sum(quotas)
    while delta:
        active = [i for i, q in enumerate(quotas) if (q < capacities[i] if delta > 0 else q > 0)]
        step = max(1, abs(delta)//len(active))
        for i in active:
            amount = min(step, abs(delta), capacities[i]-quotas[i] if delta > 0 else quotas[i])
            if delta > 0:
                quotas[i] += amount
                delta -= amount
            else:
                quotas[i] -= amount
                delta += amount
            if not delta:
                break
    for p, quota in zip(prepared, quotas):
        directory, report = p['directory'], p['report']
        selected = {name: record for name, record in load_selection(directory).items() if record['role'] == 'positive'}
        indices = set(random.Random(seed).sample(range(report['confirmed_empty_candidates']), quota))
        with (directory/'negative_candidates.jsonl').open(encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                if index in indices:
                    row = json.loads(line)
                    selected[row.pop('file_name')] = row
        report.update(dataset_initial_requested_negatives=report['requested_negatives'],
                      dataset_initial_negative_shortfall=report['negative_shortfall'],
                      requested_negatives=quota, negative_images=quota, negative_shortfall=0,
                      selected_images=len(selected), actual_negative_fraction=quota/len(selected),
                      negative_scope='batch', batch_requested_negatives=requested,
                      batch_negative_shortfall=requested-target,
                      batch_actual_negative_fraction=target/(positives+target))
        write_selection(dict(sorted(selected.items())), report, directory)
    return {'positive_images': positives, 'negative_images': target, 'requested_negatives': requested,
            'negative_shortfall': requested-target, 'selected_images': positives+target,
            'actual_negative_fraction': target/(positives+target) if positives+target else 0,
            'seed': seed, 'datasets': {p['item']['dataset']: p['report'] for p in prepared}}


def run_boxed_datasets(plan, provider, *, mode, dataset_workers, negative_fraction, seed, prefix,
                       audit_only, run_annotations, options):
    """Fetch labels first, choose the whole batch deterministically, then transfer only its manifest."""
    from threading import Event
    annotations = [item for item in plan if item['kind'] != 'images']
    images = [item for item in plan if item['kind'] == 'images']
    results = run_annotations(annotations, provider, mode=mode, dataset_workers=dataset_workers, **options)
    prepared = []
    for item in images:
        name = item['dataset']
        labels = [label for label in annotations if label['dataset'] == name]
        if any(r['status'] == 'failed' for r in results if r['dataset'] == name):
            results.append({**item, 'status': 'failed', 'error': 'Annotation download failed'})
            continue
        if options.get('dry_run'):
            print(f'[{name}] PLAN: valid boxed images + batch-wide {negative_fraction:.1%} confirmed-empty '
                  'negatives; no full-folder sync.', flush=True)
            results.append({**item, 'status': 'planned', 'selection': 'boxed'})
            continue
        try:
            metadata = next((Path(label['target']) for label in labels if label['kind'] == 'metadata'), None)
            bbox = next((Path(label['target']) for label in labels if label['kind'] == 'bbox'), None)
            if metadata is None:
                raise ValueError('Image-level metadata required to identify confirmed empty negatives')
            image_root = Path(item['target'])
            if prefix:
                image_root = image_root.parents[len(PurePosixPath(prefix).parts)-1]
            directory = image_root.parent/'selection'
            selected, report = select_images(metadata, bbox, negative_fraction=negative_fraction, seed=seed,
                                             prefix=prefix, candidate_file=directory/'negative_candidates.jsonl')
            write_selection(selected, report, directory)
            print(f'[{name}] {report["positive_images"]:,} usable boxed images; '
                  f'{report["confirmed_empty_candidates"]:,} confirmed-empty candidates.', flush=True)
            prepared.append({'item': item, 'directory': directory, 'report': report})
            del selected
        except (ValueError, OSError, ImportError) as exc:
            results.append({**item, 'status': 'failed', 'error': str(exc)})
    if not prepared:
        return results
    summary = rebalance_negatives(prepared, negative_fraction, seed)
    root = Path(options['state_dir']).parent if options.get('state_dir') else prepared[0]['directory'].parent.parent
    root.mkdir(parents=True, exist_ok=True)
    (root/'selection_batch_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Batch selection: {summary["positive_images"]:,} boxed + {summary["negative_images"]:,} empty '
          f'({summary["actual_negative_fraction"]:.2%}); shortfall {summary["negative_shortfall"]:,}.', flush=True)
    if audit_only:
        return results + [{**p['item'], 'status': 'audited', 'selection_report': p['report']} for p in prepared]
    stop = Event()

    def run_one(p, position):
        try:
            transfer = download_selected(load_selection(p['directory']), p['item'],
                                         workers=options.get('workers', 32), retries=options.get('retries', 3),
                                         retry_delay=options.get('retry_delay', 5.0),
                                         max_retry_delay=options.get('max_retry_delay', 60.0),
                                         state_dir=p['directory'], stop_event=stop, prefix=prefix,
                                         resume=options.get('resume', True), position=position)
            return {**p['item'], **transfer, 'selection_report': p['report']}
        except (ValueError, OSError) as exc:
            return {**p['item'], 'status': 'failed', 'error': str(exc)}
    executor = ThreadPoolExecutor(max_workers=min(dataset_workers, len(prepared)) if mode == 'parallel' else 1)
    futures = []
    try:
        futures = [executor.submit(run_one, p, i) for i, p in enumerate(prepared)]
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                futures.remove(future)
                print(f'Selected-image datasets remaining: {len(futures)}', flush=True)
            if not futures:
                break
    except KeyboardInterrupt:
        stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def image_url(source, filename):
    parsed = urlparse(source)
    key = parsed.path.rstrip('/')+'/'+filename
    if parsed.scheme == 's3':
        return 'https://s3.us-west-2.amazonaws.com/'+parsed.netloc+quote(key, safe='/')
    if parsed.scheme == 'gs':
        return 'https://storage.googleapis.com/'+parsed.netloc+quote(key, safe='/')
    return parsed._replace(path=quote(key, safe='/')).geturl()


def download_selected(selected, item, *, workers, retries, retry_delay, max_retry_delay,
                      state_dir, stop_event, prefix='', resume=True, position=0):
    """Bounded HTTP workers reuse connections and retain each incomplete .part file."""
    import requests
    from PIL import Image
    from tqdm import tqdm

    state_dir.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(state_dir/'downloaded_images.sqlite')
    database.execute('CREATE TABLE IF NOT EXISTS completed (url TEXT PRIMARY KEY, size INTEGER NOT NULL)')
    completed = dict(database.execute('SELECT url,size FROM completed')) if resume else {}
    sessions = local()

    def verify(path):
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, ValueError):
            return False

    def transfer(name):
        if not hasattr(sessions, 'session'):
            sessions.session = requests.Session()
        relative = str(PurePosixPath(name).relative_to(prefix)) if prefix else name
        target = (Path(item['target'])/relative).resolve()
        if Path(item['target']).resolve() not in target.parents:
            raise ValueError(f'Image destination escapes output: {name}')
        url = image_url(item['source'], relative)
        if target.is_file() and target.stat().st_size == completed.get(url):
            return url, target.stat().st_size, 0, 'skipped'
        target.parent.mkdir(parents=True, exist_ok=True)
        part, state = target.with_name(target.name+'.part'), target.with_name(target.name+'.part.json')
        delay = min(retry_delay, max_retry_delay)
        for attempt in range(retries+1):
            if stop_event.is_set():
                raise CancelledError('Image download cancelled')
            try:
                try:
                    saved = json.loads(state.read_text()) if state.is_file() else {}
                except ValueError:
                    saved = {}
                offset = part.stat().st_size if part.is_file() and saved.get('url') == url else 0
                headers = {'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                    if saved.get('etag'):
                        headers['If-Range'] = saved['etag']
                received = 0
                with sessions.session.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('Encoded image response cannot be safely resumed')
                    total = int(response.headers.get('Content-Length', 0))
                    if response.status_code == 206:
                        import re
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if not match or int(match[1]) != offset or int(match[2]) != int(match[3])-1:
                            raise ValueError('Invalid byte range for partial image')
                        total = int(match[3])
                    elif response.status_code == 200:
                        offset = 0  # No Range support or object changed: restart this image only.
                        if total and target.is_file() and target.stat().st_size == total and verify(target):
                            return url, total, 0, 'skipped'
                    else:
                        raise ValueError(f'Unexpected image HTTP status {response.status_code}')
                    temporary = state.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'url': url, 'etag': response.headers.get('ETag')}), encoding='utf-8')
                    temporary.replace(state)
                    with part.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(256*1024):
                            if stop_event.is_set():
                                raise CancelledError('Image download cancelled')
                            if chunk:
                                stream.write(chunk)
                                received += len(chunk)
                    if total and part.stat().st_size != total:
                        raise requests.exceptions.ConnectionError('Partial image retained for retry')
                if not verify(part):
                    part.unlink(missing_ok=True)
                    state.unlink(missing_ok=True)
                    raise ValueError(f'Download is not a valid image: {name}')
                part.replace(target)
                state.unlink(missing_ok=True)
                return url, target.stat().st_size, received, 'ok'
            except (requests.RequestException, OSError, ValueError) as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if status == 416:
                    # A full .part left by interruption after the last chunk needs no retransmission.
                    length = getattr(exc.response, 'headers', {}).get('Content-Range', '').removeprefix('bytes */')
                    if length.isdigit() and part.is_file() and part.stat().st_size == int(length) and verify(part):
                        part.replace(target)
                        state.unlink(missing_ok=True)
                        return url, target.stat().st_size, 0, 'ok'
                    part.unlink(missing_ok=True)
                    state.unlink(missing_ok=True)
                if attempt == retries or (status is not None and status not in (408, 416, 429) and status < 500):
                    raise
                if stop_event.wait(delay):
                    raise CancelledError('Image download cancelled') from exc
                delay = min(delay*2, max_retry_delay)

    failures, downloaded, skipped = [], 0, 0
    executor = ThreadPoolExecutor(max_workers=workers)
    pending, iterator = {}, iter(selected)

    def submit():
        name = next(iterator, None)
        if name is not None:
            pending[executor.submit(transfer, name)] = name

    try:
        for _ in range(min(workers*2, len(selected))):
            submit()
        with tqdm(total=len(selected), desc=item['dataset'], unit='img', position=position) as progress:
            while pending:
                if stop_event.is_set():
                    raise CancelledError('Image download cancelled')
                done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    name = pending.pop(future)
                    try:
                        url, size, received, status = future.result()
                    except CancelledError:
                        raise
                    except Exception as exc:
                        failures.append({'file_name': name, 'error': str(exc)})
                    else:
                        downloaded += received
                        skipped += status == 'skipped'
                        database.execute('INSERT OR REPLACE INTO completed VALUES (?,?)', (url, size))
                        database.commit()
                    progress.update(1)
                    progress.set_postfix(MiB=f'{downloaded/1048576:.1f}', skipped=skipped, failed=len(failures), refresh=False)
                    submit()
    except (KeyboardInterrupt, CancelledError):
        stop_event.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        database.close()
    (state_dir/'failed_images.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    return {'status': 'failed' if failures else 'ok', 'selected': len(selected),
            'failed_images': len(failures), 'skipped_images': skipped, 'downloaded_bytes': downloaded}
