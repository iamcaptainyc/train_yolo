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
    members = defaultdict(set)
    for index, r in enumerate(records):
        for name, _ in r['rows']:
            members[name].add(index)
    def depth(name):
        seen = set()
        while name:
            if name in seen:
                raise ValueError('Cycle in class parents')
            seen.add(name)
            name = parents.get(name)
        return len(seen)
    folded = {}
    # Keep sufficiently populated children separate, then count the actual
    # residual parent plus small children by image-set union, not box sums.
    for name in sorted(metadata, key=lambda n: (-depth(n), n)):
        parent = parents.get(name)
        if len(members[name]) < minimum and parent and name not in protected:
            members[parent].update(members[name])
            folded[name] = parent
    mapping = {}
    for name in counts:
        target = name
        while target in folded:
            target = folded[target]
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
            'parent_support': 'Bottom-up image-set unions: independently retained children do not inflate residual parent counts; small children and original coarse images contribute once per image.',
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
            r.setdefault('group_keys', []).append(('individual_image', key))
            single_images += 1
    return {'metadata': evidence, 'lila_missing_group_images_train_only': dict(fallback),
            'animals_individual_image_groups': single_images,
            'limitation': 'Animals/AP-10K without explicit groups are individual images; unknown sequences and independently copied/near-duplicate images are not proven disjoint.'}


def resplit(records, val_fraction, test_fraction, seed, groups_only=False):
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
        counts = Counter(n for i in indices for n in {n for n, _ in records[i]['rows']})
        groups.append({'indices': indices, 'counts': counts, 'size': len(indices),
                       'fixed': any(records[i].get('train_only') for i in indices), 'split': 'train'})
    if groups_only:
        for component, group in enumerate(groups):
            for i in group['indices']:
                records[i]['component'] = component
                records[i]['group_keys'].append(('preselection_component', component))
                if group['fixed']:
                    records[i]['train_only'] = True
        return {'groups': len(groups), 'method': 'freeze full-pool connected groups before sampling'}
    rng = random.Random(seed)
    rng.shuffle(groups)
    by_class = defaultdict(list)
    for i, g in enumerate(groups):
        for n in g['counts']:
            by_class[n].append(i)
    totals = Counter(n for r in records for n in {n for n, _ in r['rows']})
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
            if not fractions[split] or class_counts[split][n]:
                continue
            candidates = [groups[i] for i in by_class[n] if can_move(groups[i])]
            if candidates:
                move(min(candidates, key=lambda g: (g['size'], g['counts'][n])), split)

    def gain(g, destination):
        origin = g['split']
        # Squared per-class IMAGE fraction error, equally weighted across species.
        # Overall size is secondary; multi-box images count once for each class.
        improvement = 0.0
        for split, sign in ((origin, -1), (destination, 1)):
            for n, count in g['counts'].items():
                before = class_counts[split][n] / totals[n] - fractions[split]
                after = before + sign * count / totals[n]
                improvement += before * before - after * after
            before = sizes[split] / len(records) - fractions[split]
            after = before + sign * g['size'] / len(records)
            improvement += .01 * (before * before - after * after)
        return improvement

    # Revisit whole groups so an early coverage decision can be corrected.
    for _ in range(20):
        changed = False
        for g in sorted(groups, key=lambda g: -g['size']):
            origin = g['split']
            if g['fixed'] or any(class_counts[origin][n] <= c for n, c in g['counts'].items()):
                continue
            options = [(gain(g, s), s) for s in fractions if s != origin and fractions[s] > 0]
            if not options:
                continue
            improvement, destination = max(options)
            if improvement > 1e-12:
                sizes[origin] -= g['size']
                sizes[destination] += g['size']
                class_counts[origin].subtract(g['counts'])
                class_counts[destination].update(g['counts'])
                g['split'] = destination
                changed = True
        if not changed:
            break
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
            'class_image_counts': {n: {s: class_counts[s][n] for s in fractions} for n in sorted(totals)},
            'class_image_fractions': {n: {s: class_counts[s][n] / totals[n] for s in fractions} for n in sorted(totals)},
            'method': 'whole connected groups; minimize equally weighted per-class image-fraction error; heuristic, ratios not exact'}
