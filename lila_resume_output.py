"""Resume a completed YOLO write without decoding or hashing the input image corpus."""

from collections import Counter
import json
import os
from pathlib import Path

from PIL import Image, __version__ as PIL_VERSION
from tqdm import tqdm

from lila_state import (
    atomic_text,
    check_label_precision,
    fingerprint,
    label_text,
    record_key,
    relative_paths,
)


def restore_written(records, state, class_ids, strict_boxes=False):
    from prepare_lila import yolo_boxes

    if not getattr(state, 'compact', False) or state.force:
        raise ValueError(
            '--resume-output requires compact mode without --recheck-all; full validation will not run'
        )
    path = state.output / 'report.json'
    if not path.is_file():
        raise ValueError(
            '--resume-output requires the saved report from a completed Write YOLO stage; full validation will not run'
        )
    previous = json.loads(path.read_text(encoding='utf-8'))
    total = sum(s['images'] for s in previous.get('stats', {}).values())
    if not total or sum(previous.get('image_storage_modes', {}).values()) != total:
        raise ValueError(
            'Write YOLO did not finish according to report.json; refusing full validation in --resume-output'
        )
    if previous.get('classes') != list(class_ids):
        raise ValueError(
            'Saved class IDs changed; refusing to mix existing labels or run full validation'
        )
    if not (
        previous.get('sha256_overlap_checked')
        or previous.get('prior_sha256_grouping_preserved')
    ):
        raise ValueError(
            'No prior image-content grouping proof; --resume-output cannot invent one or run full validation'
        )
    if any(
        n
        for pair in previous.get('cross_split_overlap', {}).values()
        for n in pair.values()
    ):
        raise ValueError(
            'Previous split overlap audit failed; existing splits cannot be reused'
        )
    batches = state.phases['validation']['batches']
    if not batches or not all(batch['passed'] for batch in batches):
        raise ValueError(
            'Input files/annotations changed or validation batches are incomplete; --resume-output refuses to run Validate and hash'
        )
    old_excluded = {
        row['dataset'] + '/' + row['file_name']: row['reason']
        for row in previous.get('automatic_exclusions', [])
    }
    keys = {record_key(r) for r in records}
    if not set(old_excluded) <= keys or total + len(old_excluded) != len(records):
        raise ValueError(
            'Previous retained/excluded accounting changed; refusing to add unknown images'
        )
    # The old audit order is the input order minus old exclusions, before new precision exclusions.
    old_audit = dict(state.progress['stages'].get('audit', {}))
    images = {}
    for image in (state.output / 'images').rglob('*'):
        if not image.is_file():
            continue
        relative = image.relative_to(state.output / 'images')
        if len(relative.parts) != 3:
            raise ValueError(f'Unexpected file in existing YOLO images: {image}')
        split, dataset, name = relative.parts
        if split not in previous['stats']:
            raise ValueError(f'Unknown existing split: {split}')
        key = (dataset, name)
        if key in images:
            raise ValueError(f'Duplicate image across existing splits: {image}')
        images[key] = (split, image)
    if len(images) != total:
        raise ValueError(
            'Existing image count differs from completed write; --resume-output will not fall back to full validation'
        )
    actual, expected = Counter(), Counter()
    for dataset, splits in previous['dataset_split_stats'].items():
        expected.update(
            {
                (dataset, split): stats['images']
                for split, stats in splits.items()
                if stats['images']
            }
        )
    failures, clipped, index = [], 0, 0
    state.resumed_written = True
    state.previous_written_report = previous
    for record in tqdm(
        records, desc='Resume written YOLO (metadata and labels only)', unit='img'
    ):
        key = record_key(record)
        if key in old_excluded:
            record['validation_error'] = old_excluded[key]
            record['labels'] = []
            failures.append(key + ': ' + record['validation_error'])
            continue
        # Derive exactly the stable output name, without reading image bytes.
        record['split'] = 'train'
        image_rel, _ = relative_paths(record)
        match = images.pop((record['dataset'], Path(image_rel).name), None)
        if match is None:
            raise ValueError(
                f'Missing previously written image: {key}; full validation will not run'
            )
        split, image_path = match
        record['split'] = split
        actual[(record['dataset'], split)] += 1
        record['_previous_audit_complete'] = str(index // state.batch_size) in old_audit
        index += 1
        info = fingerprint(record['source'])
        record['validation_fingerprint'] = info
        record['image_cache_version'] = 'decode-exif-verify-sha256-v1:' + PIL_VERSION
        record['digest'] = (
            None  # No image-content hashes are rebuilt or persisted in this path.
        )
        width, height = record['image'].get('width'), record['image'].get('height')
        if width is None or height is None:
            with Image.open(image_path) as image:
                width, height = image.size  # Header only; no load()/verify().
        record['actual_dimensions'] = (width, height)
        record['_resume_storage'] = (
            'hardlink' if os.path.samefile(image_path, record['source']) else 'copy'
        )
        try:
            record['labels'], count = yolo_boxes(record, width, height, class_ids)
            check_label_precision(record['labels'])
            if count and strict_boxes:
                raise ValueError('Original bbox needs edge clipping')
            clipped += count
            _, label_rel = relative_paths(record)
            label_path = state.safe_path(label_rel)
            expected_text = label_text(record)
            if (
                not label_path.is_file()
                or label_path.read_text(encoding='utf-8') != expected_text
            ):
                atomic_text(label_path, expected_text)
                state.stats['output_labels_repaired'] += 1
            state.clear_error('validation', record)
        except ValueError as exc:
            record['validation_error'] = str(exc)
            failures.append(key + ': ' + str(exc))
            state.error('validation', record, exc)
    if images or actual != expected:
        raise ValueError(
            'Existing dataset/split inventory differs from saved report; no resplitting or full validation will run'
        )
    state.stats['input_image_decodes'] = 0
    state.stats['input_image_hashes'] = 0
    return clipped, failures


def restore_components(records):
    """Describe camera/sequence components without changing any existing split."""
    from prepare_lila import Union

    union, seen = Union(len(records)), {}
    for i, record in enumerate(records):
        keys = [('group', record['group'])]
        if record['sequence'] is not None:
            keys.append(('sequence', record['dataset'], record['sequence']))
        for key in keys:
            if key in seen:
                union.join(i, seen[key])
            else:
                seen[key] = i
    for i, record in enumerate(records):
        record['component'] = union.find(i)
    return len({r['component'] for r in records})


def verify_existing_image(record, image_path, state):
    """Hardlinks prove byte identity via inode; copied files reuse prior audit or compare bytes."""
    if os.path.samefile(image_path, record['source']):
        state.stats['output_hardlink_identities'] += 1
        return
    source_info, output_info = fingerprint(record['source']), fingerprint(image_path)
    if record['_previous_audit_complete'] and all(
        source_info[k] == output_info[k] for k in ('size', 'mtime_ns')
    ):
        state.stats['output_copies_prior_audit_reused'] += 1
        return
    # Only copied images without a usable prior audit need their contents read.
    with record['source'].open('rb') as source, image_path.open('rb') as output:
        while True:
            a, b = source.read(1024 * 1024), output.read(1024 * 1024)
            if a != b:
                raise ValueError(f'Output image bytes differ from source: {image_path}')
            if not a:
                break
    state.stats['output_copies_byte_compared'] += 1
