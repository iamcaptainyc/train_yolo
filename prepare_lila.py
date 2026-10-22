"""Merge selected LILA COCO annotations into YOLO, splitting whole cameras/sequences."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from functools import partial
from contextlib import closing
import json
from itertools import combinations
import math
import os
from pathlib import Path
import random
import re
import shutil

from PIL import Image, __version__ as PIL_VERSION
from lila_state import (
    atomic_text,
    ConversionReport,
    fingerprint,
    signature,
    record_key,
    relative_paths,
    label_text,
)
import yaml
from tqdm import tqdm

from lila_selection import EMPTY_NAMES, EXCLUDED_NAMES, items, safe_name
from lila_parallel import bounded_map

DEFAULT_DATASETS = [
    "swg-camera-traps",
    "caltech-camera-traps",
    "ena24",
    "oregon-critters",
    "nkhotakota-camera-traps",
]
# Keep species and broad groups distinct; only explicit aliases are merged.
ALIASES = {
    "domestic dog": "dog",
    "domestic cat": "cat",
    "domestic cattle": "cow",
    "mountain lion": "cougar",
    "bat species": "bat",
}
# Dataset-specific common names resolve only the species used in these datasets.
DATASET_ALIASES = {
    ("caltech-camera-traps", "opossum"): "virginia opossum",
    ("caltech-camera-traps", "raccoon"): "northern raccoon",
    ("oregon-critters", "black bear"): "american black bear",
}
LOCATION_FIELDS = ("location", "camera_id", "camera", "location_id", "site_id")
SEQUENCE_FIELDS = ("seq_id", "sequence_id", "sequence")


def normalize(name):
    return re.sub(r'\s+', ' ', re.sub(r'[_\-]+', ' ', str(name).casefold())).strip().replace('grey', 'gray')


def first_value(image, fields):
    for field in fields:
        value = image.get(field)
        if value is not None and str(value).strip().casefold() not in ('', 'none', 'null', 'unknown', 'nan'):
            return str(value)
    return None


class Union:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1]*count

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def read_records(source, datasets, class_map=None, group_map=None):
    records, mapping, skipped, fallback = [], [], Counter(), []
    class_map, group_map = class_map or {}, group_map or {}
    for dataset in datasets:
        annotation = source/dataset/'selection/selected_annotations.json'
        if not annotation.is_file():
            raise ValueError(f'Missing selected annotations: {annotation}; run download_lila.py first')
        print(f'[{dataset}] Reading selected categories, boxes and grouping metadata...', flush=True)
        categories = {}
        for category in items(annotation, 'categories'):
            original = normalize(category['name'])
            default = DATASET_ALIASES.get((dataset, original), ALIASES.get(original, original))
            override = class_map.get(dataset+':'+str(category['id']), class_map.get(original, default))
            if override is not None and not isinstance(override, str):
                raise ValueError(f'Category target must be a string or null: {dataset}:{category["id"]}')
            target = None if override is None or original in EMPTY_NAMES | EXCLUDED_NAMES else normalize(override)
            if target == '':
                raise ValueError(f'Empty target category for {dataset}:{category["id"]}')
            categories[str(category['id'])] = target
            mapping.append({'dataset': dataset, 'source_id': category['id'], 'source_name': category['name'], 'target': target})
        boxes, source_annotations, forbidden = defaultdict(list), defaultdict(list), set()
        for ann in items(annotation, 'annotations'):
            key, category = str(ann['image_id']), str(ann['category_id'])
            if category not in categories:
                raise ValueError(f'{dataset}: unknown category ID {category}')
            if categories[category] is None:
                forbidden.add(key)  # Drop the whole image rather than turn an object into background.
            else:
                boxes[key].append((categories[category], ann['bbox']))
                source_annotations[key].append(ann)
        current = []
        for image in items(annotation, 'images'):
            key = str(image['id'])
            if key in forbidden:
                skipped[dataset+':excluded_category_images'] += 1
                continue
            name = safe_name(image['file_name'])
            role = image.get('selection_role')
            annotations = boxes.get(key, [])
            if role not in ('positive', 'negative') or (role == 'positive') != bool(annotations):
                raise ValueError(f'{dataset}/{name}: inconsistent positive/negative selection role')
            override = group_map.get(dataset+'/'+name)
            if override is not None and (not isinstance(override, str) or not override.strip()):
                raise ValueError(f'Invalid group-map entry for {dataset}/{name}')
            location = override or first_value(image, LOCATION_FIELDS)
            sequence = first_value(image, SEQUENCE_FIELDS)
            current.append({'dataset': dataset, 'name': name, 'image': image, 'boxes': annotations,
                            'role': role, 'location': location, 'sequence': sequence,
                            'source': source/dataset/'images'/name, 'source_annotations': source_annotations[key]})
        # If a location is missing, sequence alone cannot prove camera disjointness.
        # With only sequence information for a dataset, use sequence disjointness.
        has_locations = any(r['location'] is not None for r in current)
        missing = any(r['location'] is None if has_locations else r['sequence'] is None for r in current)
        if missing:
            fallback.append(dataset)
        for record in current:
            record['fallback'] = missing
            record['group'] = ('dataset:'+dataset if missing else
                               dataset+':'+('camera:'+record['location'] if has_locations else 'sequence:'+record['sequence']))
        records.extend(current)
    if not records:
        raise ValueError('No usable selected images')
    return records, mapping, dict(skipped), fallback


def yolo_boxes(record, width, height, class_ids):
    if not record["boxes"]:
        return [], 0
    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) and v > 0
        for v in (width, height)
    ):
        raise ValueError("Invalid image dimensions")
    labels, clipped = set(), 0
    for category, box in record["boxes"]:
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or not all(
                isinstance(v, (float, int))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in box
            )
        ):
            raise ValueError("Invalid bbox coordinates")
        x, y, w, h = box
        if (
            w <= 0
            or h <= 0
            or min(x, y) < -1
            or x + w > width + 1
            or y + h > height + 1
        ):
            raise ValueError(f"Invalid/out-of-image bbox {box} for {width}x{height}")
        x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Degenerate clipped bbox")
        clipped += (x1, y1, x2, y2) != (x, y, x + w, y + h)
        labels.add(
            (
                class_ids[category],
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    return sorted(labels), clipped


def split_fractions(val_fraction, test_fraction=0):
    if (
        not all(math.isfinite(v) for v in (val_fraction, test_fraction))
        or not 0 < val_fraction < 1
        or test_fraction < 0
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            'val/test fractions must be finite, val > 0, test >= 0 and val+test < 1'
        )
    fractions = {'train': 1 - (val_fraction + test_fraction), 'val': val_fraction}
    if test_fraction:
        fractions['test'] = test_fraction
    return fractions


def split_overlap(records):
    """Check every pair of splits, including dataset-scoped cameras and sequences."""
    splits = [
        s for s in ('train', 'val', 'test') if any(r['split'] == s for r in records)
    ]
    keys = {
        s: {
            field: set()
            for field in ('group', 'camera', 'sequence', 'component', 'digest')
        }
        for s in splits
    }
    for record in records:
        current = keys[record['split']]
        for field in ('group', 'component', 'digest'):
            if record.get(field) is not None:
                current[field].add(record[field])
        for field, source in (('camera', 'location'), ('sequence', 'sequence')):
            if record.get(source) is not None:
                current[field].add((record['dataset'], record[source]))
    return {
        a
        + '_vs_'
        + b: {field: len(keys[a][field] & keys[b][field]) for field in keys[a]}
        for a, b in combinations(splits, 2)
    }


def split_records(records, val_fraction, seed, test_fraction=0):
    """Allocate whole connected groups jointly; never exhaust a training class."""
    fractions = split_fractions(val_fraction, test_fraction)
    holdouts = [s for s in fractions if s != 'train']
    union, seen = Union(len(records)), {}
    for index, r in enumerate(records):
        keys = [('group', r['group'])]
        if r['sequence'] is not None:
            keys.append(('sequence', r['dataset'], r['sequence']))
        if r.get('digest'):
            keys.append(('sha256', r['digest']))
        for key in keys:
            if key in seen:
                union.join(index, seen[key])
            else:
                seen[key] = index
    components = defaultdict(list)
    for index in range(len(records)):
        components[union.find(index)].append(index)
    eligible = []
    for indices in components.values():
        if not any(records[i]['fallback'] for i in indices):
            eligible.append(
                {
                    'indices': indices,
                    'datasets': Counter(records[i]['dataset'] for i in indices),
                    'classes': Counter(
                        label[0]
                        for i in indices
                        for label in records[i].get('labels', [])
                    ),
                    'positive': any(records[i].get('labels', [True]) for i in indices),
                }
            )
    if not eligible:
        raise ValueError(
            'No reliable independent camera/sequence groups available for holdouts'
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    totals = Counter(r['dataset'] for r in records)
    all_classes = Counter(label[0] for r in records for label in r.get('labels', []))
    withheld_classes, withheld_count = Counter(), 0
    allocations, counts = {s: set() for s in holdouts}, {s: Counter() for s in holdouts}

    def can_withhold(component):
        return withheld_count + len(component['indices']) < len(records) and all(
            withheld_classes[k] + n < all_classes[k]
            for k, n in component['classes'].items()
        )

    # Seed a positive group for every holdout, jointly, so filling val cannot starve test.
    positive = sorted(
        (i for i, c in enumerate(eligible) if c['positive']),
        key=lambda i: len(eligible[i]['indices']),
    )
    reserved = []

    def reserve(start=0):
        nonlocal withheld_count
        if len(reserved) == len(holdouts):
            return True
        for position in range(start, len(positive)):
            i = positive[position]
            component = eligible[i]
            if not can_withhold(component):
                continue
            reserved.append(i)
            withheld_classes.update(component['classes'])
            withheld_count += len(component['indices'])
            if reserve(position + 1):
                return True
            withheld_count -= len(component['indices'])
            withheld_classes.subtract(component['classes'])
            reserved.pop()
        return False

    if not reserve():
        raise ValueError(
            'Only one connected group or insufficient independent positive groups preserving training classes; '
            'cannot create disjoint ' + '/'.join(fractions)
        )
    for split, index in zip(holdouts, reserved):
        allocations[split].update(eligible[index]['indices'])
        counts[split].update(eligible[index]['datasets'])

    def cost(split, component=None):
        contribution = component['datasets'] if component else {}
        amount = len(component['indices']) if component else 0
        global_error = abs(
            len(allocations[split]) + amount - len(records) * fractions[split]
        ) / len(records)
        dataset_error = sum(
            abs(counts[split][d] + contribution.get(d, 0) - n * fractions[split]) / n
            for d, n in totals.items()
        ) / len(totals)
        return global_error + 0.25 * dataset_error

    for index, component in enumerate(eligible):
        if index in reserved or not can_withhold(component):
            continue
        improvements = [
            (cost(split) - cost(split, component), split)
            for split in rng.sample(holdouts, len(holdouts))
        ]
        gain, split = max(improvements, key=lambda entry: entry[0])
        if gain > 1e-12:
            allocations[split].update(component['indices'])
            counts[split].update(component['datasets'])
            withheld_classes.update(component['classes'])
            withheld_count += len(component['indices'])
    assigned = {i: split for split, indices in allocations.items() for i in indices}
    for index, record in enumerate(records):
        record['split'] = assigned.get(index, 'train')
        record['component'] = union.find(index)
    return len(components)


def validate_record(record, class_ids, plan_only=False, strict_boxes=False, state=None):
    """One task owns one record; images are closed before returning to the coordinator."""
    unknown_dimensions = 0
    width, height = record["image"].get("width"), record["image"].get("height")
    try:
        if not plan_only:
            before = fingerprint(record["source"])
            version = "decode-exif-verify-sha256-v1:" + PIL_VERSION
            stamp = {"file": before, "version": version}
            cached = state.get("image", str(record["source"]), stamp) if state else None
            if cached is None:
                with Image.open(record["source"]) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError(
                            "EXIF orientation must be normalized before conversion"
                        )
                    actual = image.size
                    image.load()
                with Image.open(record["source"]) as image:
                    image.verify()
                with record["source"].open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if fingerprint(record["source"]) != before:
                    raise ValueError("Source changed during validation")
                cached = {"width": actual[0], "height": actual[1], "digest": digest}
                if state:
                    state.put("image", str(record["source"]), stamp, cached)
            if (width is not None and width != cached["width"]) or (
                height is not None and height != cached["height"]
            ):
                raise ValueError(
                    f"Image dimensions {(cached['width'], cached['height'])} differ from metadata {(width, height)}"
                )
            width, height = cached["width"], cached["height"]
            record.update(
                digest=cached["digest"],
                validation_fingerprint=before,
                image_cache_version=version,
                actual_dimensions=(width, height),
            )
        if plan_only and record["boxes"] and (not width or not height):
            # A plan needs class counts, not usable labels; actual conversion reads the image.
            unknown_dimensions = 1
            record["labels"], count = [
                (class_ids[name], 0, 0, 0, 0) for name, _ in record["boxes"]
            ], 0
        else:
            geometry_stamp = signature(
                [width, height, record["boxes"], class_ids, "geometry-v2"]
            )
            geometry = (
                state.get("geometry", record_key(record), geometry_stamp)
                if state
                else None
            )
            if geometry is None:
                labels, count = yolo_boxes(record, width, height, class_ids)
                geometry = {"labels": labels, "count": count}
                if state:
                    state.put("geometry", record_key(record), geometry_stamp, geometry)
            record["labels"] = [tuple(b) for b in geometry["labels"]]
            count = geometry["count"]
            if count and strict_boxes:
                raise ValueError(
                    "Original bbox needs edge clipping; review it or explicitly use --allow-one-pixel-clipping"
                )
        if state:
            state.clear_error("validation", record)
        return count, unknown_dimensions, None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        record["validation_error"] = str(exc)
        if state:
            state.error("validation", record, exc)
        return 0, 0, f'{record["dataset"]}/{record["name"]}: {exc}'


def write_record(record, output, copy_mode, state=None):
    if state:
        expected = signature(
            [
                record["digest"],
                record["labels"],
                relative_paths(record),
                copy_mode,
                record["validation_fingerprint"]["device"],
                record["validation_fingerprint"]["inode"],
            ]
        )
        hit = state.output_hit(record, expected)
        if hit:
            return hit
        image_rel, label_rel = relative_paths(record)
        image, label = state.safe_path(image_rel), state.safe_path(label_rel)
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        state.begin_output(record, expected)
        temporary = image.with_name(image.name + ".lila-part")
        temporary_label = label.with_name(label.name + ".lila-part")
        try:
            mode = copy_mode
            if mode in ("auto", "hardlink"):
                try:
                    os.link(record["source"], temporary)
                    mode = "hardlink"
                except OSError:
                    if copy_mode == "hardlink":
                        raise
                    mode = "copy"
            if mode == "copy":
                shutil.copy2(record["source"], temporary)
            os.replace(temporary, image)
            temporary_label.write_text(label_text(record), encoding="utf-8")
            os.replace(temporary_label, label)
            state.refresh_source(record)
            state.save_output(record, expected, mode)
            state.clear_error("write", record)
            return mode
        except (OSError, ValueError) as exc:
            state.error("write", record, exc)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            temporary_label.unlink(missing_ok=True)

    # Flatten with a stable digest so JPEG/PNG stems and cross-dataset names cannot collide.
    key = hashlib.sha256(
        (record["dataset"] + "/" + record["name"]).encode()
    ).hexdigest()
    relative = Path(record["dataset"]) / (key + record["source"].suffix.lower())
    image = output / "images" / record["split"] / relative
    label = output / "labels" / record["split"] / relative.with_suffix(".txt")
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    mode = copy_mode
    if mode in ("auto", "hardlink"):
        try:
            os.link(record["source"], image)
            mode = "hardlink"
        except OSError:
            if copy_mode == "hardlink":
                raise
            mode = "copy"
    if mode == "copy":
        shutil.copy2(record["source"], image)
    label.write_text(
        "".join(
            str(b[0]) + " " + " ".join(f"{v:.8f}" for v in b[1:]) + "\n"
            for b in record["labels"]
        ),
        encoding="utf-8",
    )
    return mode


def prepare(
    source,
    output,
    *,
    datasets=None,
    val_fraction=0.2,
    seed=42,
    copy_mode="auto",
    plan_only=False,
    class_map=None,
    group_map=None,
    audit_hook=None,
    strict_boxes=False,
    test_fraction=0,
    workers=8,
    state=None,
    auto_clean=False,
):
    fractions = split_fractions(val_fraction, test_fraction)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("--workers must be a positive integer")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must not contain each other")
    if state is None and output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output is non-empty: {output}; choose a new output directory"
        )
    records, mapping, skipped, fallback = read_records(
        source, datasets or DEFAULT_DATASETS, class_map, group_map
    )
    if state:
        state.active_keys = {record_key(r) for r in records}
    names = sorted({name for r in records for name, _ in r["boxes"]})
    if not names:
        raise ValueError("No positive classes remain")
    class_ids = {name: index for index, name in enumerate(names)}
    clipped, missing, unknown_dimensions = 0, [], 0
    function = partial(
        validate_record,
        class_ids=class_ids,
        plan_only=plan_only,
        strict_boxes=strict_boxes,
        state=state,
    )
    try:
        with tqdm(
            total=len(records),
            desc="Validate and hash" if not plan_only else "Plan labels",
            unit="img",
        ) as progress, closing(
            bounded_map(function, records, 1 if plan_only else workers, progress)
        ) as results:
            for count, unknown, issue in results:
                clipped += count
                unknown_dimensions += unknown
                if issue is not None:
                    missing.append(issue)
    finally:
        if state:
            state.export_errors("validation", records)
    rejected = [r for r in records if r.get("validation_error")]
    if missing and not auto_clean:
        raise ValueError(
            "Input files or annotations invalid; finish downloads before conversion. First issues:\n"
            + "\n".join(missing[:20])
            + "\nFull list: validation_errors.jsonl"
        )
    if auto_clean:
        records = [r for r in records if not r.get("validation_error")]
        # An exact duplicate with conflicting annotations has no defensible winner.
        variants = defaultdict(set)
        for r in records:
            variants[r["digest"]].add(signature(r["labels"]))
        conflicts = {key for key, values in variants.items() if len(values) > 1}
        for r in records:
            if r["digest"] in conflicts:
                r["validation_error"] = (
                    "Identical image with conflicting annotations; excluded all versions"
                )
                rejected.append(r)
                if state:
                    state.error("validation", r, r["validation_error"])
        records = [r for r in records if not r.get("validation_error")]
        if state:
            state.export_errors("validation", records + rejected)
    if not records or not any(r["labels"] for r in records):
        raise ValueError("No valid positive images remain after filtering")
    components = split_records(records, val_fraction, seed, test_fraction)
    overlaps = split_overlap(records)
    if any(n for pair in overlaps.values() for n in pair.values()):
        raise ValueError(f"Cross-split grouping overlap: {overlaps}")
    # Identical image bytes with differing labels need manual review, even within one split.
    duplicates = {}
    for r in records:
        digest = r.get("digest")
        if digest is not None:
            if digest in duplicates and duplicates[digest] != r["labels"]:
                raise ValueError(
                    f'Identical image with conflicting annotations: {r["dataset"]}/{r["name"]}'
                )
            duplicates[digest] = r["labels"]
    stats = {
        split: {
            "images": 0,
            "negative_images": 0,
            "boxes": Counter(),
            "datasets": Counter(),
        }
        for split in fractions
    }
    dataset_stats = {
        d: {s: {"images": 0, "negative_images": 0, "boxes": 0} for s in stats}
        for d in sorted(datasets or DEFAULT_DATASETS)
    }
    for r in records:
        s = stats[r["split"]]
        s["images"] += 1
        s["negative_images"] += r["role"] == "negative"
        s["datasets"][r["dataset"]] += 1
        s["boxes"].update(names[label[0]] for label in r["labels"])
        ds = dataset_stats[r["dataset"]][r["split"]]
        ds["images"] += 1
        ds["negative_images"] += r["role"] == "negative"
        ds["boxes"] += len(r["labels"])
    if not all(stats[s]["boxes"] for s in stats):
        raise ValueError(
            "Every split must contain positive boxes; adjust grouping or split fractions"
        )
    report = ConversionReport(
        {
            "source": str(source),
            "seed": seed,
            "processing_workers": 1 if plan_only else workers,
            "requested_val_fraction": val_fraction,
            "actual_val_fraction": stats["val"]["images"] / len(records),
            "groups": components,
            "requested_test_fraction": test_fraction,
            "requested_split_fractions": fractions,
            "actual_split_fractions": {
                s: stats[s]["images"] / len(records) for s in stats
            },
            "cross_split_overlap": overlaps,
            "dataset_split_stats": dataset_stats,
            "split_groups": {
                s: len({r["component"] for r in records if r["split"] == s})
                for s in stats
            },
            "train_only_datasets_missing_group_info": fallback,
            "excluded_images": skipped,
            "classes": names,
            "stats": stats,
            "clipped_boxes_within_one_pixel": clipped,
            "plan_only": plan_only,
            "image_files_verified": not plan_only,
            "plan_positive_images_needing_dimensions": unknown_dimensions,
            "sha256_overlap_checked": not plan_only,
            "missing_classes": {
                s: sorted(set(names) - stats[s]["boxes"].keys()) for s in stats
            },
            "automatic_exclusions": [
                {
                    "dataset": r["dataset"],
                    "file_name": r["name"],
                    "reason": r["validation_error"],
                }
                for r in rejected
            ],
            "automatic_cleanup": auto_clean,
        }
    )
    report.state = state
    if state:
        state.reconcile(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "split_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "dataset",
                            "name",
                            "role",
                            "location",
                            "sequence",
                            "group",
                            "component",
                            "split",
                            "digest",
                        )
                    }
                )
                + "\n"
            )
    if not plan_only:
        used_modes = Counter()
        with tqdm(
            total=len(records), desc="Write YOLO", unit="img"
        ) as progress, closing(
            bounded_map(
                partial(write_record, output=output, copy_mode=copy_mode, state=state),
                records,
                workers,
                progress,
            )
        ) as results:
            for mode in results:
                used_modes[mode] += 1
        report["image_storage_modes"] = dict(used_modes)
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if state:
            state.export_errors("write", records)
            report["cache"] = dict(state.stats)
        if audit_hook is not None:
            audit_hook(records, mapping, report, output)
            (output / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        atomic_text(
            output / "data.yaml",
            yaml.safe_dump(
                {
                    "path": output.as_posix(),
                    "train": "images/train",
                    **{s: "images/" + s for s in stats if s != "train"},
                    "nc": len(names),
                    "names": dict(enumerate(names)),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
        )
    counts = ", ".join(f'{s}={stats[s]["images"]:,}' for s in stats)
    print(
        f"{len(records):,} images; {len(names)} classes; {components} disjoint groups; {counts}; plan_only={plan_only}"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('datasets/lila'))
    p.add_argument('--output', type=Path, default=Path('datasets/lila_yolo'))
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--test-fraction', type=float, default=0, help='Optional independent test split; use 0.05 with --val-fraction 0.05 for 90/5/5')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=8, help='Bounded concurrent image validation and writing; use 1 for serial')
    p.add_argument('--copy-mode', choices=['auto', 'copy', 'hardlink'], default='auto')
    p.add_argument('--plan-only', action='store_true', help='Check metadata and plan splits without image I/O or training YAML')
    p.add_argument('--class-map', type=Path, help='JSON object: normalized name or dataset:category_id -> target name (null excludes whole image)')
    p.add_argument('--group-map', type=Path, help='JSON object: dataset/relative_filename -> camera ID')
    args = p.parse_args()
    try:
        overrides = {}
        for key in ('class_map', 'group_map'):
            value = getattr(args, key)
            overrides[key] = json.loads(value.read_text(encoding='utf-8-sig')) if value else None
            if overrides[key] is not None and not isinstance(overrides[key], dict):
                raise ValueError(f'--{key.replace("_", "-")} must contain a JSON object')
        prepare(args.source, args.output, datasets=args.datasets, val_fraction=args.val_fraction,
                seed=args.seed, copy_mode=args.copy_mode, plan_only=args.plan_only, test_fraction=args.test_fraction, workers=args.workers, **overrides)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == "__main__":
    main()
