"""Recount current YOLO outputs after deletion, without decoding/hashing images."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from repair_operations import atomic_json, label_for

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def refresh_dataset_reports(roots, operations, check_only=False):
    if check_only:
        return
    affected = defaultdict(list)
    for journal in operations.glob('*.json'):
        op = json.loads(journal.read_text(encoding='utf-8'))
        path = Path(op['path'])
        for parent in path.parents:
            if parent.name == 'images':
                affected[parent.parent].append(op)
                break
    for root, ops in affected.items():
        # Ultralytics label caches may contain removed samples or old warnings.
        labels = root/'labels'
        for cache in list(labels.rglob('*.cache')) + [root/'labels.cache']:
            if cache.is_file() and not cache.is_symlink():
                cache.unlink()
        if not any(op['action'] == 'delete' for op in ops):
            continue
        recount(root, ops)


def recount(root, ops):
    import yaml
    config_path = root/'data.yaml'
    if not config_path.exists():
        # Raw non-YOLO roots have no per-class report to update.
        return
    names = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    prior_path = root/'report.json'
    prior_report = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {}
    layout = prior_report.get('image_layout', {})
    stats = {s: {'images': 0, 'negative_images': 0, 'boxes': Counter(), 'class_images': Counter()} for s in ('train','val','test')}
    sources = defaultdict(lambda: {s: {'images': 0, 'negative_images': 0, 'boxes': 0} for s in stats})
    for source in prior_report.get('dataset_split_stats', {}):
        sources[source]  # Preserve explicit zero-count sources in the report.
    class_sources = defaultdict(Counter)
    for split, row in stats.items():
        base = root/'images'/split
        for image in base.rglob('*'):
            if not image.is_file() or image.suffix.lower() not in SUFFIXES:
                continue
            label = label_for(image)
            if not label or not label.exists():
                raise ValueError(f'Cannot recount: missing YOLO label for {image}')
            classes = [names[int(line.split()[0])] for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
            relative = image.relative_to(base)
            source = relative.parts[0] if len(relative.parts) > 1 else 'dataset'
            if len(relative.parts) == 1 and layout.get('mode') == 'flat':
                prefix, separator, tail = image.name.partition('__')
                if separator and prefix in layout.get('sources', []):
                    source = prefix
            row['images'] += 1
            row['negative_images'] += not classes
            row['boxes'].update(classes)
            row['class_images'].update(set(classes))
            item = sources[source][split]
            item['images'] += 1
            item['negative_images'] += not classes
            item['boxes'] += len(classes)
            for name in set(classes):
                class_sources[name][source] += 1
    path = root/'report.json'
    baseline = root/'report.before_image_repair.json'
    if not baseline.exists():
        if path.exists():
            baseline.write_bytes(path.read_bytes())
        else:
            atomic_json(baseline, {'classes': list(names.values())})
    report = json.loads(baseline.read_text(encoding='utf-8')) if baseline.exists() else {'classes': list(names.values())}
    names = {idx: name for idx, name in names.items() if idx >= report.get('class_id_start', 0)}
    # Original sampling/provenance/audit remain available in the immutable baseline.
    # Retire derived fields that cannot be rebuilt without original group manifests.
    historical = {}
    for key in ('audit','grouping','original_to_new_split_images','refinement','rare_class_policy'):
        if key in report:
            historical[key] = report.pop(key)
    report['before_image_repair_history'] = historical
    total = sum(r['images'] for r in stats.values())
    totals = Counter()
    for row in stats.values():
        totals.update(row['class_images'])
    deleted = [op for op in ops if op['action'] == 'delete' and op.get('done')]
    prior_stats = report.get('stats', {})
    prior_sources = report.get('dataset_split_stats', {})
    removed_since_baseline = max(0, sum(v['images'] for v in prior_stats.values()) - total) if prior_stats else len(deleted)
    report['excluded_images'] = report.get('excluded_images', 0) + removed_since_baseline
    reasons = report.setdefault('excluded_by_reason', {})
    reasons['undecodable_image_cleanup'] = removed_since_baseline
    for source, rows in prior_sources.items():
        removed = max(0, sum(v['images'] for v in rows.values()) - sum(v['images'] for v in sources.get(source, {}).values()))
        report.setdefault('excluded_by_dataset', {})[source] = report.get('excluded_by_dataset', {}).get(source, 0) + removed
    report.update(stats=stats, dataset_split_stats=dict(sources),
                  missing_classes={s: sorted(set(names.values())-set(row['class_images'])) for s,row in stats.items()})
    report['image_repair'] = {'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'current_images': total, 'deleted_images_this_operation_log': len(deleted),
        'deleted_labels_this_operation_log': sum(op['result'].get('label_deleted', False) for op in deleted),
        'method': 'Recount current images and YOLO labels; class IDs and splits unchanged; original conversion audit is historical',
        'original_report': baseline.name if baseline.exists() else None}
    if 'hierarchy_sampling' in report and 'names' in report and (root/'selection.jsonl').exists():
        from animal_sampling import compile_policy, write_preview
        mapping = compile_policy(report['catalog'],report['policy'])
        manifest = root/'selection.jsonl'
        backup = root/'selection.before_image_repair.jsonl'
        if not backup.exists():
            backup.write_bytes(manifest.read_bytes())
        retained,hierarchy,provenance = [],Counter(),Counter()
        components, class_components = set(),defaultdict(set)
        for line in backup.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if not (root/record['image']).is_file():
                continue
            retained.append(line)
            components.add(record['component'])
            paths = set()
            for source,raw,concept,group in record['original_classes']:
                paths.add(mapping[concept][1])
                provenance[group,concept,source,raw] += 1
                class_components[group].add(record['component'])
            hierarchy.update({p[:n] for p in paths for n in range(1,len(p)+1)})
        if len(retained)!=total:
            raise ValueError('Current images do not match animal selection manifest')
        report.update(selected_images=total,split_image_counts={s:r['images'] for s,r in stats.items()},
                      negative_images=sum(r['negative_images'] for r in stats.values()))
        counts = {n:{s:r['class_images'].get(n,0) for s,r in stats.items()} for n in report['names']}
        report['split']['class_image_counts'] = counts
        report['split']['groups'] = len(components)
        report['split']['class_independent_groups'] = {n:len(class_components[n]) for n in report['names']}
        report['audited_boxes'] = sum(sum(r['boxes'].values()) for r in stats.values())
        report['split']['class_image_fractions'] = {n:{s:c/totals[n] if totals[n] else 0 for s,c in v.items()} for n,v in counts.items()}
        report['split']['missing_classes'] = {s:[n for n,v in counts.items() if not v[s]] for s in stats}
        report['unselected_groups'] = [g for g in report['requested_groups'] if not totals[g]]
        for row in report['hierarchy_sampling']:
            p = tuple(row['path']) + (('@self',) if row['source_bucket'] else ())
            row['selected_images'] = hierarchy[p]
        for row in report['provenance']:
            row['selected_images'] = provenance[row['group'],row['concept'],row['source'],row['original_name']]
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(''.join(line+'\n' for line in retained),encoding='utf-8')
        temporary.replace(manifest)
        atomic_json(path,report)
        write_preview(report,root/'REPORT.md')
        return
    sampling = report.setdefault('sampling', {})
    sampling.update(total_selected=total, negative_images=sum(r['negative_images'] for r in stats.values()),
                    source_images={source: sum(r['images'] for r in rows.values()) for source,rows in sources.items()},
                    selected_class_images=dict(totals))
    if 'compact' in report:
        report['compact']['class_source_images'] = dict(class_sources)
    if 'resplit' in report:
        old = report['resplit']
        report['before_image_repair_history']['resplit'] = old
        report['resplit'] = {'target_fractions': old.get('target_fractions'),
            'missing_classes': report['missing_classes'],
            'class_image_counts': {n: {s: r['class_images'].get(n,0) for s,r in stats.items()} for n in names.values()},
            'class_image_fractions': {n: {s: r['class_images'].get(n,0)/totals[n] if totals[n] else 0 for s,r in stats.items()} for n in names.values()}}
    atomic_json(path, report)
    lines = ['# 图片清理后的数据集统计', '', f'当前图片：**{total:,} 张**；当前有样本类别：**{len(totals)}**。类别 ID 保持不变。', '',
             f"本操作日志已删除 {len(deleted)} 张无法解码的图片及 {report['image_repair']['deleted_labels_this_operation_log']} 个标签。", '',
             '| 集合 | 图片 | 负样本 | 框 |', '| --- | ---: | ---: | ---: |']
    for split, row in stats.items():
        lines.append(f"| {split} | {row['images']} | {row['negative_images']} | {sum(row['boxes'].values())} |")
    lines += ['', '| ID | 类别 | 总图片 | train | val | test | 总框 | 实际比例 train/val/test |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for idx, name in names.items():
        counts = [r['class_images'].get(name,0) for r in stats.values()]
        ratios = ' / '.join(f'{c/sum(counts):.1%}' if sum(counts) else '—' for c in counts)
        lines.append(f"| {idx} | {name} | {sum(counts)} | " + ' | '.join(map(str,counts)) + f" | {sum(r['boxes'].get(name,0) for r in stats.values())} | {ratios} |")
    lines += ['', '| 来源 | train | val | test | 总图片 |', '| --- | ---: | ---: | ---: | ---: |']
    for source, rows in sources.items():
        counts = [r['images'] for r in rows.values()]
        lines.append(f'| {source} | ' + ' | '.join(map(str,counts)) + f' | {sum(counts)} |')
    lines += ['', '当前数量由磁盘图片和标签重数，不重新划分、不重编号。原始转换映射、抽样及分组审计保存在清理前报告中，属于历史记录。', '']
    targets = [p for p in (root/'SUBSET_REPORT.md',root/'MERGE_REPORT.md',root/'AUDIT_REPORT.md',root/'REPORT.md') if p.exists()]
    for target in targets or [root/'REPORT.md']:
        backup = target.with_name(target.stem+'.before_image_repair.md')
        if target.exists() and not backup.exists():
            backup.write_bytes(target.read_bytes())
        text = '\n'.join(lines)
        if backup.exists():
            text += f'\n[清理前的详细报告（历史数量）]({backup.name})\n'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(text,encoding='utf-8')
        tmp.replace(target)
