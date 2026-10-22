"""Recover reviewed fine labels by matching original boxes, never by guessing images."""
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path

from lila_camping import load_policy
from prepare_lila import items, safe_name, yolo_boxes

RULE_FILE = Path(__file__).parent / 'configs/lila_refinements.json'


def image_counts(records):
    # One image contributes once to each class, regardless of its box count.
    return Counter(n for r in records for n in {name for name, _ in r['rows']})


def coord_key(coords):
    return tuple(f'{float(v):.8f}' for v in coords)


def source_key(name):
    # Original author names may contain apostrophes/scientific punctuation.
    # Only reviewed output names must satisfy the strict YOLO English policy.
    return re.sub(r'[\s_-]+', '_', str(name).strip().casefold()).replace('grey', 'gray')


def refine_large(records, datasets, metadata, maximum, source_override=None, check_all=True):
    from merge_yolo_datasets import normalized
    before = image_counts(records)
    oversized = {n for n, count in before.items() if maximum and count > maximum}
    checked = set(before) if check_all else oversized
    policy = load_policy()
    rules = {(r['dataset'], source_key(r['source_name'])): r
             for r in json.loads(RULE_FILE.read_text(encoding='utf-8'))['rules']}
    for rule in rules.values():
        metadata.setdefault(rule['target'], {'parent': rule['parent'], 'chinese': rule['chinese'],
                                            'group': metadata.get(rule['parent'], {}).get('group', 'animal')})
    for rule in rules.values():
        metadata[rule['target']]['group'] = metadata[rule['parent']]['group']
    original = {(ds, source_key(r['source_name'])): normalized(r['target']) if r['target'] else None
                for ds, rows in policy['datasets'].items() for r in rows}
    for r in records:
        r['source_rows'] = r['rows']
        r['target_names'] = [n for n, _ in r['rows']]
        r['row_origins'] = [[(r['dataset'], '', raw, 'source_yolo_only') for raw in
                              r.get('original_row_names', {}).get((n, coords), (n,))]
                            for n, coords in r['rows']]
        r['refined_names'] = list(r['target_names'])
    for d in datasets:
        d['original_mapping'] = dict(d['mapping'])
    result = {'threshold_images': maximum, 'check_all_classes': check_all, 'checked_classes': sorted(checked), 'before_class_images': dict(before),
              'oversized_before': sorted(oversized), 'metadata': [], 'issues': Counter(),
              'rules_sha256': hashlib.sha256(RULE_FILE.read_bytes()).hexdigest()}
    if not checked:
        return result
    lila = next(d for d in datasets if d['alias'] == 'lila')
    run = lila['yaml'].parent / '.lila-state/run.json'
    config = json.loads(run.read_text(encoding='utf-8'))['config'] if run.is_file() else {}
    source = Path(source_override or config.get('source') or 'datasets/lila')
    wanted = defaultdict(list)
    for r in records:
        if r['dataset'] == 'lila' and any(n in checked for n, _ in r['rows']):
            wanted[r['relative']].append(r)
    slugs = sorted({Path(p).parts[0] for p in wanted if len(Path(p).parts) > 1})
    for slug in slugs:
        path = source / slug / 'selection/selected_annotations.json'
        if not path.is_file():
            result['issues']['missing_annotation_file:' + slug] += 1
            continue
        result['metadata'].append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        categories = {str(c['id']): c['name'] for c in items(path, 'categories')}
        candidates = {}
        for image in items(path, 'images'):
            name = safe_name(image['file_name'])
            stem = hashlib.sha256((slug + '/' + name).encode()).hexdigest()
            relative = slug + '/' + stem + Path(name).suffix.lower()
            if relative in wanted:
                candidates[str(image['id'])] = (image, wanted[relative])
        boxes = defaultdict(lambda: defaultdict(set))
        for ann in items(path, 'annotations'):
            image_id = str(ann['image_id'])
            if image_id not in candidates:
                continue
            image, _ = candidates[image_id]
            raw = categories.get(str(ann['category_id']))
            if raw is None:
                result['issues']['unknown_original_category'] += 1
                continue
            key = (slug, source_key(raw))
            parent = original.get(key)
            if parent not in checked:
                continue
            rule = rules.get(key)
            target = rule['target'] if rule else parent
            # A legacy mapping accidentally put yellow-throated marten under bird.
            # Only repair when the original box matches exactly; never guess from ID.
            if raw == 'yellow_throated_marten' and slug == 'swg-camera-traps':
                target = 'yellow_throated_marten'
                rule = {'parent': 'mustelid', 'chinese': '黄喉貂'}
            if rule:
                metadata.setdefault(target, {'parent': rule['parent'], 'chinese': rule['chinese'],
                                              'group': metadata[rule['parent']]['group']})
            try:
                labels, _ = yolo_boxes({'boxes': [(parent, ann['bbox'])]}, image.get('width'), image.get('height'), {parent: 0})
                coords = coord_key(labels[0][1:])
            except (ValueError, TypeError, KeyError, IndexError):
                result['issues']['cannot_reconstruct_original_box_or_dimensions'] += 1
                continue
            boxes[image_id][(parent, coords)].add((str(ann['category_id']), raw, target,
                                                  'refined' if rule else 'original_coarse_or_no_reviewed_rule'))
        for image_id, (_, matches) in candidates.items():
            for record in matches:
                for i, (parent, coords) in enumerate(record['source_rows']):
                    if parent not in checked:
                        continue
                    origins = boxes[image_id].get((parent, coord_key(coords)), set())
                    choices = {x[2] for x in origins}
                    if len(choices) == 1:
                        record['target_names'][i] = next(iter(choices))
                        record['row_origins'][i] = [(slug, cid, raw, reason) for cid, raw, _, reason in sorted(origins)]
                    else:
                        result['issues']['ambiguous_or_unmatched_box_kept_coarse'] += 1
                        record['row_origins'][i] = [(slug, '', parent, 'unmatched_or_ambiguous_original_category_unknown')]
                record['rows'] = [(n, row[1]) for n, row in zip(record['target_names'], record['source_rows'], strict=True)]
                record['refined_names'] = list(record['target_names'])
    return result


def finish_refinement_report(report, records, metadata=None):
    transitions = Counter()
    for r in records:
        transitions.update({(old[0], new) for old, new in zip(r['source_rows'], r['target_names'], strict=True)})
    report['transitions'] = [{'source': old, 'target': new, 'images': count}
                             for (old, new), count in sorted(transitions.items()) if old != new]
    report['after_class_images'] = dict(image_counts(records))
    report['still_above_threshold'] = {n: c for n, c in image_counts(records).items()
                                        if report['threshold_images'] and c > report['threshold_images']}
    # Aggregate only: no persistent per-image cache or new image reads.
    parents = {m.get('parent') for m in (metadata or {}).values()} - {None}
    parents.update(old[0] for r in records for old in r['source_rows'])
    parents.update(n for r in records for n in r['target_names'])
    counts, parent_totals = Counter(), Counter()
    for r in records:
        memberships, present = set(), set()
        for i, target in enumerate(r['target_names']):
            if target not in parents:
                continue
            present.add(target)
            for ds, cid, raw, reason in r['row_origins'][i]:
                if r['refined_names'][i] != target:
                    reason = 'below_min_images_folded_to_parent'
                memberships.add((target, ds, cid, raw, reason))
        counts.update(memberships)
        parent_totals.update(present)
    report['parent_remainders'] = {
        n: {'images': total, 'sources': [
            {'dataset': ds, 'source_id': cid, 'source_name': raw, 'reason': reason, 'images': count}
            for (target, ds, cid, raw, reason), count in sorted(counts.items()) if target == n]}
        for n, total in sorted(parent_totals.items())}
    report['source_count_note'] = 'Count once per image per source category/reason; source rows can overlap in multi-category images. Parent total is the union, not the sum.'
    return report


def soft_cap(records, maximum, minimum, seed):
    """Sample whole images, retaining rare-class evidence and split coverage.

    Selection is round-robin by group within each split. Never remove individual
    boxes, modify source files or move a retained image to another split.
    """
    import random
    counts = image_counts(records)
    if not maximum:
        return records, [], {'maximum_images': 0, 'removed_images': 0}
    oversized = {n for n, c in counts.items() if c > maximum}
    protected = {n for n, c in counts.items() if c <= max(minimum, 1)}
    kept, pending = [], defaultdict(list)
    used, coverage = Counter(), set()
    for r in records:
        names = {n for n, _ in r['rows']}
        if not names & oversized or names & protected:
            kept.append(r)
            used.update(names)
            coverage.update((r['split'], n) for n in names)
        else:
            pending[(r['split'], r.get('component', r['relative']))].append(r)
    rng = random.Random(seed)
    queues = list(pending.values())
    rng.shuffle(queues)
    for q in queues:
        rng.shuffle(q)
    rejected, removed_counts = [], Counter()
    while queues:
        remaining = []
        for q in queues:
            r = q.pop()
            names = {n for n, _ in r['rows']}
            needed = any((r['split'], n) not in coverage or counts[n] - removed_counts[n] <= max(minimum, 1) for n in names)
            if needed or all(used[n] < maximum for n in names & oversized):
                kept.append(r)
                used.update(names)
                coverage.update((r['split'], n) for n in names)
            else:
                r['error'] = 'whole-image sampling to limit class image counts; source untouched'
                rejected.append(r)
                removed_counts.update(names)
            if q:
                remaining.append(q)
        queues = remaining
    # Maintain deterministic original order, independent of writing worker count.
    kept_ids = {id(r) for r in kept}
    kept = [r for r in records if id(r) in kept_ids]
    return kept, rejected, {'maximum_images': maximum, 'removed_images': len(rejected),
                            'remaining_over_limit': {n: c for n, c in used.items() if c > maximum},
                            'exceptions': 'Rare-class images and at least one image per existing class/split are protected; multi-class co-occurrence can exceed cap.'}
