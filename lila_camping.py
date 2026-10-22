"""Reviewed, dataset-scoped English common-name labels for the camping task."""

import json
from pathlib import Path
import re

from prepare_lila import normalize
from lila_selection import EMPTY_NAMES

POLICY = Path(__file__).parent / 'configs/lila_camping.json'


def load_policy(path=POLICY):
    policy = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if policy.get('schema') != 1:
        raise ValueError('Unsupported camping label policy')
    for name, info in policy['classes'].items():
        if not re.fullmatch(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*', name):
            raise ValueError(f'Camping class must be an English snake_case name: {name}')
        if info.get('parent') and info['parent'] not in policy['classes']:
            raise ValueError(f'Unknown display parent for {name}')
        seen, current = set(), name
        while current:
            if current in seen:
                raise ValueError(f'Cyclic display hierarchy: {name}')
            seen.add(current)
            current = policy['classes'][current].get('parent')
    return policy


def camping_mapping(inventories, policy_file=POLICY, exclusions=()):
    policy = load_policy(policy_file)
    mapping, details = {}, []
    known = {f'{ds}:{c["id"]}' for ds, inv in inventories.items() for c in inv['selected_categories']}
    if set(exclusions) - known:
        raise ValueError(f'Unknown --exclude-category keys: {sorted(set(exclusions) - known)}')
    for ds, inv in inventories.items():
        decisions = {}
        for row in policy['datasets'].get(ds, []):
            name = normalize(row['source_name'])
            if name in decisions and decisions[name]['target'] != row['target']:
                raise ValueError(f'Conflicting camping decisions: {ds}/{name}')
            if row['target'] is not None and row['target'] not in policy['classes']:
                raise ValueError(f'Unknown camping target: {ds}/{name}')
            decisions[name] = row
        for c in inv['selected_categories']:
            key, name = f'{ds}:{c["id"]}', normalize(c['name'])
            row = decisions.get(name)
            if row is None and name in EMPTY_NAMES | {'ignore', 'blurred', 'problem'}:
                row = {'target': None, 'reason': 'non-training status label', 'source_chinese': '空图或图像状态标签'}
            # IDs are scoped to each source file; names must be explicitly reviewed.
            if row is None:
                raise ValueError(f'No reviewed camping mapping for {key} ({c["name"]})')
            target, reason = row['target'], row['reason']
            if key in exclusions:
                target, reason = None, 'explicit user exclusion; drop whole image'
            elif not c['boxes']:
                target, reason = None, 'zero selected boxes; no active output class'
            if name in policy['excluded_generic_names']:
                target, reason = None, 'user excluded unidentified animal/mammal labels; drop whole image'
            mapping[key] = target
            details.append({
                'key': key, 'dataset': ds, 'source_id': c['id'],
                'source_name': c['name'], 'source_chinese': row['source_chinese'],
                'selected_boxes': c['boxes'], 'target': target, 'reason': reason,
                'lineage': [], 'policy': policy['name'],
            })
    return mapping, details


def describe_policy(report, path=POLICY):
    policy = load_policy(path)
    return {
        'name': policy['name'], 'scope': 'LILA only',
        'ancestor_descendant_labels_allowed': True,
        'unknown_animal_fallback_classes_enabled': False,
        'wild_pig_target': 'wild_boar',
        'classes': {name: policy['classes'][name] for name in report['classes']},
        'notes': policy['notes'],
    }
