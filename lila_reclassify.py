"""Rebuild labels from original annotations, inheriting an unchanged completed audit.

No image corpus decoding, content hashing, copying, or resplitting is permitted.
The previous output and original source images must be hardlinks to the same files.
"""

from collections import Counter
import json
import os
from pathlib import Path

from PIL import Image, __version__ as PIL_VERSION
from tqdm import tqdm

from lila_compact import CompactState
from lila_state import check_label_precision, fingerprint, label_text, record_key, relative_paths


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def restore_reclassified(records, state, class_ids, previous_output, strict_boxes=False):
    from prepare_lila import read_records, yolo_boxes

    previous_output = Path(previous_output).resolve()
    if not state or not state.compact or state.force:
        raise ValueError('--remap-from requires compact checkpoints and never runs full validation')
    if previous_output == state.output or previous_output in state.output.parents or state.output in previous_output.parents:
        raise ValueError('--remap-from and --output must be separate, non-nested directories')
    previous = load(previous_output / 'report.json')
    run = load(previous_output / '.lila-state/run.json')
    provenance = load(previous_output / 'audit_sources.json')
    total = sum(s['images'] for s in previous.get('stats', {}).values())
    audit = previous.get('audit', {})
    if not (
        run.get('status') == 'COMPLETE'
        and (previous_output / 'data.yaml').is_file()
        and total > 0
        and sum(previous.get('image_storage_modes', {}).values()) == total
        and audit.get('images_verified') == total
        and audit.get('all_output_images_match_source') is True
        and (previous.get('sha256_overlap_checked') or previous.get('prior_sha256_grouping_preserved'))
        and audit.get('boxes_verified') == sum(sum(s['boxes'].values()) for s in previous['stats'].values())
    ):
        raise ValueError('--remap-from requires a completed, certified audit; full validation will not run')
    if any(n for pair in previous.get('cross_split_overlap', {}).values() for n in pair.values()):
        raise ValueError('Prior split overlap proof failed')
    config = run['config']
    for field in ('source', 'datasets', 'group_map', 'seed', 'val_fraction', 'test_fraction'):
        if state.config.get(field) != config.get(field):
            raise ValueError(f'Cannot change {field} during label remapping; existing splits must be preserved')
    # Verify the source metadata used for the old completed audit is still identical.
    from lila_audit import sha256

    for ds, inv in provenance['datasets'].items():
        selected = Path(config['source']) / ds / 'selection/selected_annotations.json'
        if sha256(selected) != inv['selection_sha256']:
            raise ValueError(f'Selected annotations changed since the old audit: {ds}')
    old_records, _, _, _ = read_records(
        Path(config['source']), config['datasets'], config['class_map'], config.get('group_map'),
        preserve_class_names=config.get('category_policy') == 'camping',
    )
    old_state = CompactState(previous_output)
    try:
        old_state.config = config  # Read-only: do not call configure()/prepare_batches().
        batches = old_state.progress['stages'].get('validation', {})
        count = (len(old_records) + old_state.batch_size - 1) // old_state.batch_size
        if set(batches) != {str(i) for i in range(count)}:
            raise ValueError('Prior validation batches are incomplete; full validation will not run')
        for start in tqdm(range(0, len(old_records), old_state.batch_size), desc='Check prior batch metadata', unit='batch'):
            if batches[str(start // old_state.batch_size)] != old_state._stamp(old_records[start:start + old_state.batch_size], 'validation'):
                raise ValueError('Source image attributes or annotations changed; refusing remap without rerunning validation')
    finally:
        old_state.close()
    old_by_key = {record_key(r): r for r in old_records}
    excluded = {e['dataset'] + '/' + e['file_name']: e['reason'] for e in previous.get('automatic_exclusions', [])}
    if total + len(excluded) != len(old_records) or not set(excluded) <= old_by_key.keys():
        raise ValueError('Prior retained/excluded accounting is inconsistent')
    images, actual = {}, Counter()
    for image in (previous_output / 'images').rglob('*'):
        if not image.is_file():
            continue
        relative = image.relative_to(previous_output / 'images')
        if len(relative.parts) != 3:
            raise ValueError(f'Unexpected prior image path: {image}')
        split, ds, name = relative.parts
        if split not in previous['stats'] or (ds, name) in images:
            raise ValueError(f'Unknown split or duplicate prior image: {image}')
        images[(ds, name)] = (split, image)
        actual[(ds, split)] += 1
    expected = Counter({(ds, split): s['images'] for ds, splits in previous['dataset_split_stats'].items() for split, s in splits.items() if s['images']})
    if len(images) != total or actual != expected:
        raise ValueError('Prior output image inventory changed; refusing remap')
    old_ids = {name: i for i, name in enumerate(previous['classes'])}
    clipped, failures = 0, []
    for record in tqdm(records, desc='Rebuild English labels (no image decode/hash)', unit='img'):
        key = record_key(record)
        if key in excluded or key not in old_by_key:
            # Never resurrect corrupted or previously unverified source pictures.
            issue = excluded.get(key, 'not present in the previously verified source subset')
            record['validation_error'] = issue
            record['labels'] = []
            failures.append(key + ': ' + issue)
            state.error('validation', record, issue)
            state.validation_done(record)
            continue
        old = old_by_key[key]
        old['split'] = 'train'
        image_relative, _ = relative_paths(old)
        match = images.get((old['dataset'], Path(image_relative).name))
        if match is None:
            raise ValueError(f'Missing previously audited image: {key}')
        split, image_path = match
        record['split'] = old['split'] = split
        if not os.path.samefile(record['source'], image_path):
            raise ValueError(f'Fast remapping requires prior hardlinked images: {key}; no copy/hash fallback will run')
        width, height = old['image'].get('width'), old['image'].get('height')
        if width is None or height is None:
            with Image.open(image_path) as image:
                width, height = image.size  # Header only; never load()/verify().
            state.stats['image_headers_read'] += 1
        old['labels'], _ = yolo_boxes(old, width, height, old_ids)
        _, old_label = relative_paths(old)
        if (previous_output / old_label).read_text(encoding='utf-8') != label_text(old):
            raise ValueError(f'Prior label differs from original annotations: {old_label}')
        try:
            record['labels'], count = yolo_boxes(record, width, height, class_ids)
            check_label_precision(record['labels'])
            if count and strict_boxes:
                raise ValueError('Original bbox needs edge clipping')
            clipped += count
        except ValueError as exc:
            record['validation_error'] = str(exc)
            record['labels'] = []
            failures.append(key + ': ' + str(exc))
            state.error('validation', record, exc)
        record['actual_dimensions'] = (width, height)
        record['validation_fingerprint'] = fingerprint(record['source'])
        record['image_cache_version'] = 'decode-exif-verify-sha256-v1:' + PIL_VERSION
        record['digest'] = None
        record['_resume_storage'] = 'hardlink'
        record['_previous_audit_complete'] = True
        if not record.get('validation_error'):
            state.clear_error('validation', record)
        state.validation_done(record)
    state.resumed_written = True
    state.reclassifying = True
    state.previous_written_report = previous
    state.stats['input_image_decodes'] = 0
    state.stats['input_image_hashes'] = 0
    state.stats['prior_image_metadata_verified'] = len(old_records)
    return clipped, failures
