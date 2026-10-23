"""Reviewed 70-class remapping and bounded, Animals-first whole-image sampling."""
from collections import Counter, defaultdict, deque
import heapq
import json
import math
from pathlib import Path
import random

POLICY_FILE = Path(__file__).parent/'configs/camping70.json'
SOURCE_PRIORITY = ('ap10k', 'animals', 'lila')


def source_rank(source):
    return SOURCE_PRIORITY.index(source) if source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)


SAMPLING_REASON = 'camping70 balanced whole-image sampling; source file retained'


def policy():
    value = json.loads(POLICY_FILE.read_text(encoding='utf-8'))
    if len(value['classes']) != 70 or not set(value['aliases'].values()) <= value['classes'].keys():
        raise ValueError('Invalid reviewed camping70 policy')
    return value


def target_name(name, metadata, rules):
    """Walk only reviewed ancestors; ambiguous rodent/mustelid labels stay excluded."""
    seen = set()
    while name and name not in seen:
        seen.add(name)
        if name in rules['excluded']:
            return None
        if name in rules['aliases']:
            return rules['aliases'][name]
        if name in rules['classes']:
            return name
        name = metadata.get(name, {}).get('parent')
    return None


def remap(records, metadata):
    from merge_yolo_datasets import unique_rows
    rules = policy()
    kept, rejected = [], []
    changes = Counter()
    for r in records:
        targets = [target_name(n, metadata, rules) for n, _ in r['rows']]
        if any(n is None for n in targets):
            unsupported = sorted({n for (n, _), target in zip(r['rows'], targets) if target is None})
            r['error'] = 'camping70 unsupported/ambiguous label; exclude WHOLE image: ' + ', '.join(unsupported)
            rejected.append(r)
            continue
        changes.update({(r['dataset'], old, new) for (old, _), new in zip(r['rows'], targets)})
        # Keep one target per original source row for the independent output audit.
        r['target_names'] = targets
        r['rows'] = unique_rows([(n, row[1]) for n, row in zip(targets, r['source_rows'], strict=True)])
        kept.append(r)
    metadata.update(rules['classes'])
    return kept, rejected, {'candidate_classes': list(rules['classes']),
                            'mapping': [{'dataset': ds, 'source': old, 'target': new, 'images': count}
                                        for (ds, old, new), count in sorted(changes.items())]}


def deduplicate(records):
    """Known identical files get one sampling slot; prefer the Animals record."""
    seen, kept, rejected = set(), [], []
    order = sorted(range(len(records)), key=lambda i: (source_rank(records[i]['dataset']), i))
    chosen = set()
    for i in order:
        r = records[i]
        stamp = r['image_stamp']
        key = ('sha256', r['duplicate_digest']) if r.get('duplicate_digest') else (
            ('inode', stamp['device'], stamp['inode']) if stamp['inode'] else ('path', str(r['image'])))
        if key not in seen:
            seen.add(key)
            chosen.add(i)
    for i, r in enumerate(records):
        if i in chosen:
            kept.append(r)
        else:
            r['error'] = 'camping70 duplicate file identity; kept one record, preferring AP-10K then Animals'
            rejected.append(r)
    return kept, rejected


class Queue:
    def __init__(self, indices, records, rng):
        buckets = defaultdict(lambda: defaultdict(list))
        for index in indices:
            r = records[index]
            buckets[r['dataset']][r.get('component', r['relative'])].append(index)
        self.sources = []
        for source in sorted(buckets, key=lambda s: (source_rank(s), s)):
            groups = list(buckets[source].values())
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
            self.sources.append(deque(deque(group) for group in groups))

    def pop(self, chosen):
        # Exhaust AP-10K, then Animals, then LILA candidates for this class.
        for groups in self.sources:
            while groups:
                group = groups.popleft()
                value = None
                while group and value is None:
                    candidate = group.popleft()
                    if candidate not in chosen:
                        value = candidate
                if group:
                    groups.append(group)
                if value is not None:
                    return value
        return None


def sample(records, maximum, negative_fraction, class_limit, seed, val_fraction, test_fraction):
    """Never exceed total image budget, including negatives and multi-class images."""
    if not 1 <= maximum <= 30000 or not 0 <= negative_fraction <= .2 or class_limit < 1:
        raise ValueError('max-images must be 1..30000, negative-fraction 0..0.2 and class target positive')
    rng = random.Random(seed)
    memberships = [{n for n, _ in r['rows']} for r in records]
    available = Counter(n for names in memberships for n in names)
    fractions = {'train': 1-val_fraction-test_fraction, 'val': val_fraction, 'test': test_fraction}
    negatives = sum(not names for names in memberships)
    negative_budget = min(negatives, int(maximum*negative_fraction))
    positive_budget = maximum-negative_budget
    def budgets(total, coverage=False):
        val, test = int(total*val_fraction), int(total*test_fraction)
        if coverage and total >= 3:
            val, test = max(1, val), max(1, test)
        requested = {'train': total-val-test, 'val': val, 'test': test}
        available_by_split = Counter(r['split'] for r, names in zip(records, memberships)
                                     if bool(names) == coverage)
        result = {s: min(count, available_by_split[s]) for s, count in requested.items()}
        remaining = total-sum(result.values())
        for split in ('train', 'val', 'test'):
            extra = min(remaining, available_by_split[split]-result[split])
            result[split] += extra
            remaining -= extra
        return result
    pos_budgets, neg_budgets = budgets(positive_budget, coverage=True), budgets(negative_budget)
    chosen, counts, source_counts = set(), Counter(), Counter()
    for split in ('train', 'val', 'test'):
        indices = [i for i, r in enumerate(records) if r['split'] == split]
        by_class = defaultdict(list)
        for i in indices:
            for name in sorted(memberships[i]):
                by_class[name].append(i)
        queues = {n: Queue(v, records, rng) for n, v in by_class.items()}
        goals = {n: min(len(v), max(1, math.ceil(class_limit*fractions[split]))) for n, v in by_class.items()}
        used, picked = Counter(), 0
        # Prefer the least-filled class in absolute image counts. Small classes
        # exhaust their evidence while rich classes continue toward the soft target.
        heap = [(0, i, n, 0) for i, n in enumerate(sorted(queues))]
        heapq.heapify(heap)
        while heap and picked < pos_budgets[split]:
            _, tie, name, previous = heapq.heappop(heap)
            if used[name] >= goals[name]:
                continue
            if previous != used[name]:
                heapq.heappush(heap, (used[name], tie, name, used[name]))
                continue
            index = queues[name].pop(chosen)
            if index is None:
                continue
            chosen.add(index)
            picked += 1
            used.update(memberships[index])
            counts.update(memberships[index])
            source_counts[records[index]['dataset']] += 1
            heapq.heappush(heap, (used[name], tie, name, used[name]))
        negative_queue = Queue([i for i in indices if not memberships[i]], records, rng)
        for _ in range(neg_budgets[split]):
            index = negative_queue.pop(chosen)
            if index is None:
                break
            chosen.add(index)
            source_counts[records[index]['dataset']] += 1
    kept, rejected = [], []
    for i, r in enumerate(records):
        if i in chosen:
            kept.append(r)
        else:
            r['error'] = SAMPLING_REASON
            rejected.append(r)
    if len(kept) > maximum:
        raise ValueError('Internal error: total image budget exceeded')
    stats = {'maximum_images': maximum, 'total_selected': len(kept), 'removed_images': len(rejected),
             'negative_fraction_target': negative_fraction, 'negative_images': sum(not r['rows'] for r in kept),
             'soft_class_target_images': class_limit, 'source_priority': list(SOURCE_PRIORITY),
             'source_images': dict(source_counts), 'available_class_images': dict(available),
             'selected_class_images': dict(counts),
             'remaining_over_limit': {n: c for n, c in counts.items() if c > class_limit},
             'positive_split_budgets': pos_budgets, 'negative_split_budgets': neg_budgets,
             'note': 'Known groups were allocated before sampling; source preference within each class/split; '
                     'multi-class co-occurrence may exceed soft class targets. Scarce groups may leave budget unused.'}
    return kept, rejected, stats


def filter_lila_area(records, minimum):
    if not 0 <= minimum < 1:
        raise ValueError('LILA minimum bbox area ratio must be in [0, 1)')
    kept, rejected = [], []
    for r in records:
        if r['dataset'] == 'lila' and r['rows'] and not any(float(coords[2])*float(coords[3]) > minimum for _,coords in r['rows']):
            r['error'] = f'LILA has no bbox area ratio strictly > {minimum}; exclude whole image'
            rejected.append(r)
        else:
            kept.append(r)
    return kept, rejected, {'minimum_bbox_area_ratio': minimum, 'comparison': 'strictly greater',
                            'images_excluded': len(rejected), 'multi_box_rule': 'ANY animal box must pass; retain ALL boxes including smaller ones',
                            'negative_rule': 'original empty labels are exempt; they have no bbox'}


def sample_global(records, maximum, negative_fraction, class_limit, seed):
    # Input source splits carry no allocation constraint here; groups already frozen.
    splits = [r['split'] for r in records]
    try:
        for r in records:
            r['split'] = 'train'
        kept, rejected, stats = sample(records, maximum, negative_fraction, class_limit, seed, 0, 0)
    finally:
        for r, original in zip(records, splits):
            r['split'] = original
    stats['global_positive_budget'] = sum(stats.pop('positive_split_budgets').values())
    stats['global_negative_budget'] = sum(stats.pop('negative_split_budgets').values())
    stats['order'] = 'merge -> full-pool grouping -> global sample -> grouped resplit'
    stats['note'] = 'Source preference is global per class; original splits are not retained. Full-pool component links survive sampling.'
    return kept, rejected, stats


def markdown(report):
    rules = policy()
    stats = report['stats']
    sample_info = report['sampling']
    total = sum(s['images'] for s in stats.values())
    lines = ['# 露营动物小数据集提取报告', '',
             f"状态：输出检查通过。共 **{total:,} 张**，上限 **{sample_info['maximum_images']:,} 张**；实际有样本类别 **{len(report['classes'])} / {len(rules['classes'])}**。", '',
             '先合并来源与类别，再按类别全局优先选 AP-10K，其次 Animals Detection，最后 LILA；抽样完成后才划分集合。'
             '这是来源优先，不是图像清晰度评分。没有重新解码或哈希图片。', '',
             f"LILA 框面积门槛：**严格大于 {report.get('compact', {}).get('lila_area_filter', {}).get('minimum_bbox_area_ratio', 0):.6f}**；任意一个正框达标即保留整图和全部框；全部不达标才排除。原本空标签的负样本不受面积门槛限制。", '',
             '## 来源和划分', '', '| 来源 | 划分 | 图片 | 负样本 | 框 |', '| --- | --- | ---: | ---: | ---: |']
    for source, splits in report['dataset_split_stats'].items():
        for split, row in splits.items():
            lines.append(f"| {source} | {split} | {row['images']} | {row['negative_images']} | {row['boxes']} |")
    for split, row in stats.items():
        lines.append(f"| 合计 | {split} | {row['images']} | {row['negative_images']} | {sum(row['boxes'].values())} |")
    lines += ['', '## 全部候选类别及实际图片数量', '',
              '同图多类分别计数，列相加不等于总图数。无样本类别不分配训练 ID，不制造空类别。', '',
              '| ID | 中文 | 英文 | AP-10K | Animals | LILA | 总图片 | train | val | test |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    ids = {n: i for i, n in enumerate(report['classes'])}
    for n, info in rules['classes'].items():
        counts = [stats[s]['class_images'].get(n, 0) for s in ('train', 'val', 'test')]
        source = report['compact']['class_source_images'].get(n, {})
        lines.append(f"| {ids.get(n, '无样本')} | {info['chinese']} | {n} | {source.get('ap10k', 0)} | {source.get('animals', 0)} | {source.get('lila', 0)} | {sum(counts)} | " + ' | '.join(map(str, counts)) + ' |')
    lines += ['', '## 每类划分比例', '', '按包含该类的图片计数；同图多个同类框只计一次。大相机组、仅限训练的来源及多类共现可能限制比例。', '',
              '| 类别 | train / val / test 实际比例 | 相对目标最大偏差（百分点） |', '| --- | --- | ---: |']
    fractions = report['resplit']['target_fractions']
    for n in report['classes']:
        counts = [stats[s]['class_images'].get(n, 0) for s in ('train', 'val', 'test')]
        total_class = sum(counts)
        actual = [c / total_class for c in counts]
        deviation = max(abs(value - fractions[split]) for value, split in zip(actual, ('train', 'val', 'test'))) * 100
        lines.append(f"| {n} | " + ' / '.join(f'{v:.1%}' for v in actual) + f" | {deviation:.1f} |")
    lines += ['', '## 合并来源', '', '| 输入来源 | 输入类别 | 输出类别 | 抽样前符合范围的图片数 |', '| --- | --- | --- | ---: |']
    for row in report['compact']['mapping']:
        lines.append(f"| {row['dataset']} | {row['source']} | {row['target']} | {row['images']} |")
    lines += ['', '## 检查与限制', '',
              '- 总图片数包含正样本和负样本，严格不超过上限；不复制稀缺图来凑配额。',
              '- 默认逐类目标 80%/10%/10%（自定义参数以 report.json 为准），先绑定完整池中的相机/序列/已知同图分组，全局抽样后再按保留分组划分；允许不足额，不能拆组造成泄漏。',
              '- 原始分组缺失的 LILA 子数据集保守限于训练集。Animals/AP-10K 无序列元数据时按单图分组，不能证明未知近似图完全隔离。',
              '- 已有 YOLO 坐标 token 与源标签逐项核对；AP-10K 从 COCO xywh 与原图尺寸计算归一化坐标，审计时重新计算核对。只改类别或格式，保留全部有效框。图片检查硬链接身份，不重新认证源物种或清晰度。',
              '- 一张图只要含有范围外/无法安全映射的正标签就整图排除，不能删掉该框当背景。只有原本空标签的图片能作负样本。',
              '- 应用分组：豹/猎豹/美洲豹/雪豹统一 leopard；狮子/美洲狮统一 lion；浣熊/小熊猫统一 raccoon；鸽子并入 bird。这些标签不代表同一生物物种。蜗牛、鲨鱼、海豹、海狮排除。',
              '- 棕熊、黑熊和其他熊合入 bear；北极熊、大熊猫单独保留。bear 不代表无威胁。wolf 现在可含 AP-10K 原始 wolf 和原 LILA 胡狼/郊狼业务分组，不能把全部 wolf 样本视为纯灰狼；狗独立。',
              '- 图像主体不复制；新标签和报告占少量空间。源图片不删除。',
              '- Ctrl+C 后同命令重跑：重扫标签与元数据，不解码图片；复用已写好且一致的硬链接/标签。', '',
              '各集合缺少的类别：`'+json.dumps(report['missing_classes'], ensure_ascii=False)+'`。', '',
              '问题/范围外图片明细见 excluded_images.jsonl；正常但未抽中的图片仅汇总计数，不保存逐图记录。完整统计见 report.json。', '']
    return '\n'.join(lines)
