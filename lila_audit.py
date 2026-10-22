"""Pinned LILA taxonomy, raw COCO inventories, and selection-to-original crosschecks."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from tqdm import tqdm

from lila_selection import EMPTY_NAMES, EXCLUDED_NAMES, items, json_path, safe_name
from prepare_lila import normalize

DATASET_NAMES = {
    'swg-camera-traps': 'SWG Camera Traps',
    'caltech-camera-traps': 'Caltech Camera Traps',
    'ena24': 'ENA24',
    'oregon-critters': 'Oregon Critters',
    'nkhotakota-camera-traps': 'Nkhotakota Camera Traps',
}
TAXONOMY = Path(__file__).parent / 'configs/lila_taxonomy.csv'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def categories(path):
    result = {}
    for row in items(path, 'categories'):
        key = str(row['id'])
        if key in result:
            raise ValueError(f'Duplicate category ID {key}: {path}')
        result[key] = row
    if not result:
        raise ValueError(f'No categories: {path}')
    return result


def box_key(ann):
    # Numeric 1 and 1.0 must compare equally; reject malformed values later, never infer units.
    box = ann.get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f'Missing/malformed positive bbox: {ann}')
    return (str(ann['category_id']), tuple(box))


def resolve_original(source, dataset, kind, selection_report, overrides):
    value = overrides.get(dataset, {}).get(kind, selection_report.get(kind + '_file'))
    if not value:
        raise ValueError(
            f'{dataset}: missing {kind}_file in selection_report.json; provide --original-files'
        )
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    # Selection reports may contain the downloader machine's absolute paths.
    basename = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    matches = list((source / dataset / 'annotations').rglob(basename))
    matches = [p for p in matches if p.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f'{dataset}: cannot uniquely locate original {kind} {value}; provide --original-files'
        )
    return matches[0].resolve()


def verified_json_path(path):
    original = Path(path)
    cached = json_path(original)
    if original.suffix.lower() == '.zip':
        with zipfile.ZipFile(original) as bundle:
            names = [
                name
                for name in bundle.namelist()
                if name.lower().endswith('.json') and not name.startswith('__MACOSX/')
            ]
            if len(names) != 1:
                raise ValueError(f'Expected exactly one original JSON: {original}')
            with bundle.open(names[0]) as stream:
                expected = hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha256(cached) != expected:
            raise ValueError(
                f'Original ZIP JSON cache differs from ZIP contents: {cached}'
            )
    return cached


def inventory(path, selected_by_original, *, collect_boxes=False):
    """Stream all annotations; disk-backed unique (category,image) counts bound RAM usage."""
    path = verified_json_path(path)
    cats = categories(path)
    ann_counts, bbox_counts, selected_anns = Counter(), Counter(), defaultdict(list)
    selected_images, image_rows = {}, 0
    with tempfile.TemporaryDirectory(prefix='lila_inventory_') as temp:
        db = sqlite3.connect(str(Path(temp) / 'counts.sqlite3'))
        try:
            db.execute(
                'CREATE TABLE seen (category TEXT, image TEXT, boxed INTEGER, PRIMARY KEY(category,image)) WITHOUT ROWID'
            )
            batch = []
            for ann in tqdm(
                items(path, 'annotations'), desc='Original annotations', unit='ann'
            ):
                category, key = str(ann['category_id']), str(ann['image_id'])
                if category not in cats:
                    raise ValueError(f'Unknown original category {category}: {path}')
                ann_counts[category] += 1
                boxed = bool(
                    isinstance(ann.get('bbox'), list) and len(ann['bbox']) == 4
                )
                bbox_counts[category] += boxed
                batch.append((category, key, int(boxed)))
                if len(batch) >= 10000:
                    db.executemany(
                        'INSERT INTO seen VALUES (?,?,?) ON CONFLICT(category,image) DO UPDATE SET boxed=max(boxed,excluded.boxed)',
                        batch,
                    )
                    batch.clear()
                if key in selected_by_original and (not collect_boxes or 'bbox' in ann):
                    selected_anns[key].append(ann)
            if batch:
                db.executemany(
                    'INSERT INTO seen VALUES (?,?,?) ON CONFLICT(category,image) DO UPDATE SET boxed=max(boxed,excluded.boxed)',
                    batch,
                )
            image_counts = {
                c: n
                for c, n in db.execute(
                    'SELECT category,count(*) FROM seen GROUP BY category'
                )
            }
            boxed_images = {
                c: n
                for c, n in db.execute(
                    'SELECT category,sum(boxed) FROM seen GROUP BY category'
                )
            }
        finally:
            db.close()
    for image in tqdm(
        items(path, 'images'), desc='Original image metadata', unit='img'
    ):
        image_rows += 1
        key = str(image['id'])
        if key in selected_by_original:
            if key in selected_images:
                raise ValueError(f'Duplicate selected original image ID {key}: {path}')
            selected_images[key] = image
    return (
        {
            'file': str(path),
            'sha256': sha256(path),
            'images': image_rows,
            'annotations': sum(ann_counts.values()),
            'bbox_records': sum(bbox_counts.values()),
            'categories': [
                {
                    **cat,
                    'annotations': ann_counts[key],
                    'images': image_counts.get(key, 0),
                    'bbox_records': bbox_counts[key],
                    'bbox_images': boxed_images.get(key, 0),
                }
                for key, cat in cats.items()
            ],
        },
        selected_images,
        selected_anns,
    )


def inspect_sources(source, datasets, overrides=None):
    """Check every selected image/box against the original, before applying taxonomy exclusions."""
    result = {}
    for dataset in datasets:
        directory = source / dataset / 'selection'
        selected = directory / 'selected_annotations.json'
        selection_report = json.loads(
            (directory / 'selection_report.json').read_text(encoding='utf-8-sig')
        )
        cats = categories(selected)
        images, source_ids, filenames = {}, {}, set()
        for image in items(selected, 'images'):
            key, original = str(image['id']), str(image.get('source_image_id', ''))
            name = safe_name(image['file_name'])
            if (
                not original
                or key in images
                or original in source_ids
                or name in filenames
            ):
                raise ValueError(
                    f'{dataset}: duplicate ID/path or missing source_image_id: {image}'
                )
            if image.get('selection_role') not in ('positive', 'negative'):
                raise ValueError(f'{dataset}: invalid selection role')
            images[key], source_ids[original] = image, key
            filenames.add(name)
        annotations = defaultdict(list)
        ann_ids = set()
        for ann in items(selected, 'annotations'):
            key, category = str(ann['image_id']), str(ann['category_id'])
            if key not in images or category not in cats:
                raise ValueError(f'{dataset}: dangling selected annotation {ann}')
            if str(ann['id']) in ann_ids:
                raise ValueError(
                    f'{dataset}: duplicate selected annotation ID {ann["id"]}'
                )
            ann_ids.add(str(ann['id']))
            annotations[key].append(ann)
        metadata_file = resolve_original(
            source, dataset, 'metadata', selection_report, overrides or {}
        )
        bbox_file = resolve_original(
            source, dataset, 'bbox', selection_report, overrides or {}
        )
        print(
            f'[{dataset}] Inventory full original metadata and detection labels...',
            flush=True,
        )
        metadata, meta_images, meta_anns = inventory(metadata_file, source_ids)
        if metadata_file == bbox_file:
            bbox, box_images, box_anns = metadata, meta_images, meta_anns
        else:
            bbox, box_images, box_anns = inventory(
                bbox_file, source_ids, collect_boxes=True
            )
        raw_cats = {str(c['id']): c for c in bbox['categories']}
        meta_cats = {str(c['id']): c for c in metadata['categories']}
        for key, cat in cats.items():
            if key not in raw_cats or normalize(cat['name']) != normalize(
                raw_cats[key]['name']
            ):
                raise ValueError(
                    f'{dataset}: selected category namespace differs from original bbox categories: {key}'
                )
        checked_boxes, checked_negatives = 0, 0
        raw_ids = {}
        for original, key in source_ids.items():
            image = images[key]
            raw_image = box_images.get(original, meta_images.get(original))
            if raw_image is None or safe_name(raw_image['file_name']) != safe_name(
                image['file_name']
            ):
                raise ValueError(
                    f'{dataset}: original image ID/path mismatch for {key}'
                )
            for dimension in ('width', 'height'):
                if (
                    raw_image.get(dimension) is not None
                    and image.get(dimension) != raw_image[dimension]
                ):
                    raise ValueError(f'{dataset}: changed original {dimension}: {key}')
            original_boxes = [a for a in box_anns.get(original, []) if 'bbox' in a]
            selected_boxes = annotations.get(key, [])
            if image['selection_role'] == 'negative':
                original_meta = meta_anns.get(original, [])
                if original not in meta_images or safe_name(
                    meta_images[original]['file_name']
                ) != safe_name(image['file_name']):
                    raise ValueError(
                        f'{dataset}: negative image does not match original metadata path: {key}'
                    )
                if (
                    selected_boxes
                    or not original_meta
                    or any(
                        normalize(meta_cats[str(a['category_id'])]['name'])
                        not in EMPTY_NAMES
                        for a in original_meta
                    )
                ):
                    raise ValueError(
                        f'{dataset}: negative image lacks unambiguous original empty label: {key}'
                    )
                if any(
                    normalize(raw_cats[str(a['category_id'])]['name'])
                    not in EMPTY_NAMES
                    for a in original_boxes
                ):
                    raise ValueError(
                        f'{dataset}: negative image has original object bbox: {key}'
                    )
                checked_negatives += 1
            else:
                if not selected_boxes or Counter(
                    map(box_key, original_boxes)
                ) != Counter(map(box_key, selected_boxes)):
                    raise ValueError(
                        f'{dataset}: selected boxes do not exactly match original image/category/coordinates: {key}'
                    )
                checked_boxes += len(selected_boxes)
                lookup = defaultdict(list)
                for ann in original_boxes:
                    lookup[box_key(ann)].append(ann.get('id'))
                raw_ids[key] = [lookup[box_key(a)].pop() for a in selected_boxes]
        counts, image_sets = Counter(), defaultdict(set)
        for key, rows in annotations.items():
            for ann in rows:
                category = str(ann['category_id'])
                counts[category] += 1
                image_sets[category].add(key)
        result[dataset] = {
            'metadata': metadata,
            'bbox': bbox,
            'selection_sha256': sha256(selected),
            'selected_images': len(images),
            'selected_positive_images': len(images) - checked_negatives,
            'selected_negative_images': checked_negatives,
            'selected_boxes': checked_boxes,
            'selected_categories': [
                {**c, 'boxes': counts[k], 'images': len(image_sets[k])}
                for k, c in cats.items()
            ],
            'raw_annotation_ids': raw_ids,
            'source_files': [
                {'path': str(p), 'sha256': sha256(p)}
                for p in dict.fromkeys((metadata_file, bbox_file))
            ],
        }
    return result


def taxonomy_mapping(inventories, taxonomy_file=TAXONOMY, exclusions=()):
    rows = {}
    if Path(taxonomy_file).resolve() == TAXONOMY.resolve():
        pinned = json.loads(
            TAXONOMY.with_suffix('.source.json').read_text(encoding='utf-8')
        )
        if sha256(taxonomy_file) != pinned['snapshot_sha256']:
            raise ValueError(
                'Bundled taxonomy snapshot changed; restore it or use a reviewed custom --taxonomy-file'
            )
    with Path(taxonomy_file).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['dataset_name'], normalize(row['query']))
            if key in rows and any(
                rows[key][f] != row[f]
                for f in ('taxonomy_level', 'scientific_name', 'taxonomy_string')
            ):
                raise ValueError(f'Conflicting taxonomy query: {key}')
            rows[key] = row
    mapping, details, nodes, labels = {}, [], {}, {}
    known_keys = {
        f'{d}:{c["id"]}'
        for d, inv in inventories.items()
        for c in inv['selected_categories']
    }
    if set(exclusions) - known_keys:
        raise ValueError(
            f'Unknown --exclude-category keys: {sorted(set(exclusions)-known_keys)}'
        )
    for dataset, inv in inventories.items():
        for category in inv['selected_categories']:
            key = dataset + ':' + str(category['id'])
            name = normalize(category['name'])
            detail = {
                'key': key,
                'dataset': dataset,
                'source_id': category['id'],
                'source_name': category['name'],
                'selected_boxes': category['boxes'],
                'target': None,
                'lineage': [],
            }
            if name in EMPTY_NAMES | EXCLUDED_NAMES:
                detail['reason'] = 'empty/non-animal/excluded source label'
            elif not category['boxes']:
                detail['reason'] = 'zero selected boxes; does not influence hierarchy'
            elif key in exclusions:
                detail['reason'] = 'explicit user exclusion; drop entire image'
            else:
                row = rows.get((DATASET_NAMES.get(dataset, dataset), name))
                if row is None or not row['taxonomy_string']:
                    raise ValueError(
                        f'No official taxonomy mapping for active category {key} ({name}); review taxonomy CSV'
                    )
                parsed = ast.literal_eval(row['taxonomy_string'])
                lineage = []
                for node in parsed:
                    taxon, rank, scientific = (
                        str(node[0]),
                        str(node[1]),
                        str(node[2]).casefold(),
                    )
                    lineage.append(taxon)
                    label = rank + ':' + scientific
                    if taxon in labels and labels[taxon] != label:
                        raise ValueError(f'Inconsistent official taxonomy node {taxon}')
                    labels[taxon] = label
                    if taxon in nodes and nodes[taxon] != [
                        str(x[0]) for x in parsed[len(lineage) - 1 :]
                    ]:
                        raise ValueError(
                            f'Inconsistent official taxonomy ancestry {taxon}'
                        )
                    nodes[taxon] = [str(x[0]) for x in parsed[len(lineage) - 1 :]]
                if (
                    not lineage
                    or labels[lineage[0]]
                    != row['taxonomy_level'] + ':' + row['scientific_name'].casefold()
                ):
                    raise ValueError(f'Official taxonomy identity mismatch: {key}')
                detail['official_taxon'] = labels[lineage[0]]
                detail['official_query'] = row['query']
                detail['lineage'] = lineage
                rank = row['taxonomy_level']
                if (
                    rank
                    in (
                        'stateofmatter',
                        'kingdom',
                        'phylum',
                        'subphylum',
                        'superclass',
                        'subclass',
                        'infraclass',
                        'superorder',
                    )
                    or labels[lineage[0]] == 'class:mammalia'
                ):
                    detail['reason'] = (
                        'overly broad/ambiguous mapping; drop entire image'
                    )
                else:
                    detail['node'] = lineage[0]
                    detail['reason'] = 'official taxon identity'
            details.append(detail)
    active = {d['node'] for d in details if 'node' in d}
    for detail in details:
        if 'node' in detail:
            # Choose the highest annotated ancestor; the output nodes form an antichain.
            target = next(n for n in reversed(detail['lineage']) if n in active)
            detail['target_taxon_id'] = target
            detail['target'] = labels[target]
            if target != detail['node']:
                detail['reason'] = 'coarsened to annotated ancestor ' + labels[target]
        mapping[detail['key']] = detail['target']
    targets = {d['target_taxon_id'] for d in details if d.get('target')}
    if any((set(nodes[t][1:]) & targets) for t in targets):
        raise ValueError('Output taxonomy contains an ancestor/descendant conflict')
    return mapping, details
