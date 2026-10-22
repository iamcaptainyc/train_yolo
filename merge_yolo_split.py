"""Metadata-only grouping, rare-class folding and deterministic grouped splitting."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from prepare_lila import LOCATION_FIELDS, SEQUENCE_FIELDS, first_value, Union, items, safe_name


def fold_rare(records, datasets, metadata, minimum):
    from merge_yolo_refine import image_counts
    counts = image_counts(records)
    # These application labels must stay distinct even with little data.
    protected = {'wild_boar', 'pig', 'warthog'}
    parents = {n: m.get('parent') for n, m in metadata.items()}
    parents['leopard_cat'] = 'wild_cat'
    metadata['leopard_cat']['parent'] = 'wild_cat'
    support = Counter()
    for r in records:
        covered = set()
        for name, _ in r['rows']:
            lineage, node = set(), name
            while node:
                if node in lineage:
                    raise ValueError('Cycle in class parents')
                lineage.add(node)
                covered.add(node)
                node = parents.get(node)
        support.update(covered)
    mapping = {}
    for name in counts:
        target, visited = name, set()
        while support[target] < minimum and target not in protected and parents.get(target):
            if target in visited:
                raise ValueError('Cycle in class parents')
            visited.add(target)
            target = parents[target]
        mapping[name] = target
    from merge_yolo_datasets import unique_rows
    for r in records:
        old = r['rows']
        if 'target_names' in r:
            r['target_names'] = [mapping.get(n, n) for n in r['target_names']]
        r['rows'] = unique_rows([(mapping[n], c) for n, c in old])
        r['duplicate_boxes_removed'] += len(old) - len(r['rows'])
    for d in datasets:
        d['mapping'] = {i: mapping.get(n, n) for i, n in d['mapping'].items()}
    return {'minimum_total_images': minimum, 'basis': 'images containing the class, counted once per input image across all original splits',
            'parent_support': 'Parent thresholds include descendant image coverage so a residual parent is not folded away merely because common children were restored.',
            'changes': [{'source': n, 'target': t, 'original_images': counts[n]} for n, t in sorted(mapping.items()) if n != t],
            'rare_without_safe_parent': [n for n in sorted(counts) if counts[n] < minimum and mapping[n] == n]}


def attach_groups(records, datasets, source_override=None, group_map_path=None):
    """Recover LILA camera and sequence keys without reading any image bytes."""
    override = json.loads(Path(group_map_path).read_text(encoding='utf-8')) if group_map_path else {}
    if not isinstance(override, dict) or any(not isinstance(v, str) or not v.strip() for v in override.values()):
        raise ValueError('--group-map must map dataset/original_split/relative_image to nonempty group strings')
    lila = next(d for d in datasets if d['alias'] == 'lila')
    run_path = lila['yaml'].parent / '.lila-state/run.json'
    config = json.loads(run_path.read_text(encoding='utf-8'))['config'] if run_path.is_file() else {}
    source = Path(source_override or config.get('source') or 'datasets/lila')
    wanted = defaultdict(list)
    for r in records:
        if r['dataset'] == 'lila':
            wanted[r['relative']].append(r)
    slugs = sorted({Path(p).parts[0] for p in wanted if len(Path(p).parts) > 1})
    evidence = []
    for slug in slugs:
        annotation = source / slug / 'selection/selected_annotations.json'
        if not annotation.is_file():
            continue
        evidence.append({'path': str(annotation.resolve()), 'sha256': hashlib.sha256(annotation.read_bytes()).hexdigest()})
        for image in items(annotation, 'images'):
            name = safe_name(image['file_name'])
            stem = hashlib.sha256((slug + '/' + name).encode()).hexdigest()
            relative = slug + '/' + stem + Path(name).suffix.lower()
            matches = wanted.get(relative)
            if not matches:
                continue
            location = (config.get('group_map') or {}).get(slug + '/' + name) or first_value(image, LOCATION_FIELDS)
            sequence = first_value(image, SEQUENCE_FIELDS)
            for r in matches:
                r['group_keys'] = [('camera', slug, str(location))] if location is not None else []
                if sequence is not None:
                    r['group_keys'].append(('sequence', slug, str(sequence)))
                r['has_camera'] = location is not None
    # If some images lack camera IDs in a dataset with cameras, sequence alone
    # cannot prove camera separation. Keep that dataset together conservatively.
    camera_slugs = {Path(r['relative']).parts[0] for r in records if r.get('has_camera')}
    incomplete = {Path(r['relative']).parts[0] for r in records
                  if r['dataset'] == 'lila' and Path(r['relative']).parts[0] in camera_slugs and not r.get('has_camera')}
    incomplete.update(Path(r['relative']).parts[0] for r in records
                      if r['dataset'] == 'lila' and not r.get('group_keys')
                      and f"lila/{r['split']}/{r['relative']}" not in override)
    fallback, single_images = Counter(), 0
    for r in records:
        key = f"{r['dataset']}/{r['split']}/{r['relative']}"
        if key in override:
            r.setdefault('group_keys', []).append(('explicit', override[key]))
        if r['dataset'] == 'lila':
            slug = Path(r['relative']).parts[0]
            if slug in incomplete:
                # No unsafe random fallback for camera trap data.
                r.setdefault('group_keys', []).append(('unknown_lila_camera', slug))
                r['train_only'] = True
                fallback[slug] += 1
        elif key not in override:
            r['group_keys'] = [('individual_image', key)]
            single_images += 1
    return {'metadata': evidence, 'lila_missing_group_images_train_only': dict(fallback),
            'animals_individual_image_groups': single_images,
            'limitation': 'Animals without explicit groups are individual images; unknown sequences and independently copied/near-duplicate images are not proven disjoint.'}


def resplit(records, val_fraction, test_fraction, seed):
    fractions = {'train': 1 - val_fraction - test_fraction, 'val': val_fraction, 'test': test_fraction}
    union, seen = Union(len(records)), {}
    for i, r in enumerate(records):
        keys = list(r['group_keys'])
        stamp = r['image_stamp']
        if stamp['inode']:
            keys.append(('inode', stamp['device'], stamp['inode']))
        if r.get('duplicate_digest'):
            keys.append(('sha256', r['duplicate_digest']))
        for k in keys:
            if k in seen:
                union.join(i, seen[k])
            else:
                seen[k] = i
    components = defaultdict(list)
    for i in range(len(records)):
        components[union.find(i)].append(i)
    groups = []
    for indices in components.values():
        counts = Counter(n for i in indices for n, _ in records[i]['rows'])
        groups.append({'indices': indices, 'counts': counts, 'size': len(indices),
                       'fixed': any(records[i].get('train_only') for i in indices), 'split': 'train'})
    rng = random.Random(seed)
    rng.shuffle(groups)
    by_class = defaultdict(list)
    for i, g in enumerate(groups):
        for n in g['counts']:
            by_class[n].append(i)
    totals = Counter(n for r in records for n, _ in r['rows'])
    class_counts = {'train': totals.copy(), 'val': Counter(), 'test': Counter()}
    sizes = {'train': len(records), 'val': 0, 'test': 0}

    def can_move(g):
        return not g['fixed'] and g['split'] == 'train' and all(class_counts['train'][n] > c for n, c in g['counts'].items())

    def move(g, split):
        g['split'] = split
        sizes['train'] -= g['size']
        sizes[split] += g['size']
        class_counts['train'].subtract(g['counts'])
        class_counts[split].update(g['counts'])

    # Scarce classes first. Never remove a class's last training group.
    for n in sorted(by_class, key=lambda n: (len(by_class[n]), totals[n], n)):
        for split in ('val', 'test'):
            if class_counts[split][n]:
                continue
            candidates = [groups[i] for i in by_class[n] if can_move(groups[i])]
            if candidates:
                move(min(candidates, key=lambda g: (g['size'], g['counts'][n])), split)

    def gain(g, split):
        target = len(records) * fractions[split]
        size_gain = (abs(sizes[split] - target) - abs(sizes[split] + g['size'] - target)) / len(records)
        class_gain = sum((abs(class_counts[split][n] - totals[n] * fractions[split]) -
                          abs(class_counts[split][n] + c - totals[n] * fractions[split])) / totals[n]
                         for n, c in g['counts'].items()) / max(1, len(totals))
        return size_gain + .25 * class_gain

    for g in sorted(groups, key=lambda g: -g['size']):
        if can_move(g):
            improvement, split = max((gain(g, s), s) for s in ('val', 'test'))
            if improvement > 0:
                move(g, split)
    for component, g in enumerate(groups):
        for i in g['indices']:
            r = records[i]
            r['original_split'] = r['split']
            r['split'] = g['split']
            r['component'] = component
            for field in ('image_rel', 'label_rel'):
                p = r[field]
                r[field] = Path(p.parts[0], r['split'], *p.parts[2:])
    # Check every grouping key independently after allocation.
    assigned = {}
    for r in records:
        for k in r['group_keys'] + [('component', r['component'])]:
            if k in assigned and assigned[k] != r['split']:
                raise ValueError('Cross-split grouping overlap')
            assigned[k] = r['split']
    return {'target_fractions': fractions, 'seed': seed, 'groups': len(groups), 'group_overlap': 0,
            'class_independent_groups': {n: len(v) for n, v in sorted(by_class.items())},
            'missing_classes': {s: sorted(n for n in totals if not class_counts[s][n]) for s in fractions},
            'method': 'whole connected camera/sequence/identity groups; rare-class coverage first, then ratio and class balance; heuristic, ratios not exact'}
