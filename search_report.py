"""Human-readable experiment summaries; metrics never influence model selection here."""
from pathlib import Path


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def metric(record, split, key):
    result = record.get(split, {})
    value = result.get('metrics', {}).get(key)
    return f'{value * 100:.2f}%' if result.get('status') == 'ok' and value is not None else '—'


def usable(record):
    return record['state'] == 'COMPLETE' and all(record.get(k, {}).get('status') == 'ok' for k in ('train', 'val'))


def delta(record, baseline):
    if not baseline or not usable(baseline) or not usable(record):
        return '—（无有效基线）'
    value = (record['val']['metrics']['map50_95'] - baseline['val']['metrics']['map50_95']) * 100
    return f'{value:+.2f} 个百分点'


def write_report(root, records, winners, models, complete=False):
    root = Path(root)
    baselines = {r['model']: r for r in records if r['trial'] == 0}
    best = {}
    for model in models:
        candidates = [r for r in records if r['model'] == model and usable(r)]
        if candidates:
            best[model] = max(candidates, key=lambda r: r['val']['metrics']['map50_95'])
    lines = ['# YOLO 训练对比报告', '',
             '**状态：本次搜索完成。**' if complete else '**状态：阶段性结果；本次搜索尚未确认全部完成。**', '',
             f'已记录 {len(records)} 个候选，其中 {sum(usable(r) for r in records)} 个完成训练与验证。', '',
             '选优依据：**验证集 mAP50-95，越高越好**。测试集仅用于最终评估，不参与排名。', '',
             '所有精度指标以百分数显示；相对基线变化用百分点表示。空白项（—）表示没有有效结果，不是 0。', '',
             '## 最佳 n / s 模型', '',
             '| Size | 推荐模型 | 候选 | 验证 mAP50-95 | 较同模型基线 | 测试状态 | 权重 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for size in ('n', 's'):
        r = winners.get(size)
        if not r:
            lines.append(f'| {size} | 暂无有效候选 | — | — | — | 未评估 | — |')
            continue
        test_status = {'ok': '已完成', 'failed': '失败，需重试'}.get(r['test'].get('status'), '尚未评估')
        lines.append(f"| **{size}** | **{r['model']}** | t{r['trial']:03d} | **{metric(r, 'val', 'map50_95')}** | {delta(r, baselines.get(r['model']))} | {test_status} | [best_{size}.pt](best/best_{size}.pt) |")
    lines += ['', '## 最佳模型的完整指标', '',
              '| Size | 数据划分 | mAP50-95 | mAP50 | Precision（精确率） | Recall（召回率） |',
              '| --- | --- | --- | --- | --- | --- |']
    for size, r in winners.items():
        for split, label in [('val', '验证集'), ('test', '测试集')]:
            values = ' | '.join(metric(r, split, key) for key in ('map50_95', 'map50', 'precision', 'recall'))
            lines.append(f'| {size} | {label} | {values} |')
    lines += ['', '## 按 Size 比较架构', '']
    for size in ('n', 's'):
        lines += [f'### {size} 模型', '', '| 模型 | 最佳候选 | 基线验证 mAP50-95 | 最佳验证 mAP50-95 | 提升 | 结论 |',
                  '| --- | --- | --- | --- | --- | --- |']
        for model in models:
            if Path(model).stem[-1] != size:
                continue
            r, baseline = best.get(model), baselines.get(model)
            if not r:
                lines.append(f'| {model} | — | — | — | — | 暂无有效结果 |')
                continue
            verdict = '本尺寸最佳' if winners.get(size) == r else '备选'
            if r['trial'] == 0:
                verdict += '；基线仍最佳（含并列）'
            lines.append(f"| {model} | t{r['trial']:03d} | {metric(baseline or {}, 'val', 'map50_95')} | {metric(r, 'val', 'map50_95')} | {delta(r, baseline)} | {verdict} |")
    lines += ['', '## 按模型查看每次尝试', '']
    for model in models:
        lines += [f'### {model}', '',
                  '| 候选 | 状态 | 验证 mAP50-95 | 较基线 | 测试 mAP50-95 | batch | 优化器配置 | lr0 | weight_decay | mosaic |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in sorted((r for r in records if r['model'] == model), key=lambda r: r['trial']):
            status = {'COMPLETE': '训练与验证完成', 'RUNNING': '未完成', 'FAIL': '失败'}.get(r['state'], r['state'])
            if r.get('status', {}).get('state') == 'INTERRUPTED':
                status = '已中断，可继续'
            if r['test'].get('status') == 'failed':
                status += '；测试失败'
            tag = f"t{r['trial']:03d}" + ('（基线）' if r['trial'] == 0 else '')
            if best.get(model) == r:
                tag += ' ★'
            params = r['params']
            values = ' | '.join(cell(params.get(k, '默认/自动')) for k in ('batch', 'optimizer', 'lr0', 'weight_decay', 'mosaic'))
            lines.append(f"| {tag} | {status} | {metric(r, 'val', 'map50_95')} | {delta(r, baselines.get(model))} | {metric(r, 'test', 'map50_95')} | {values} |")
        lines.append('')
    lines += ['## 获胜候选与日志', '']
    for size, r in winners.items():
        directory = Path(r['trial_dir']).relative_to(root).as_posix()
        lines += [f"- **{size}：{r['model']} / t{r['trial']:03d}**。原始候选目录：`{directory}/`；"
                  f'[参数]({directory}/config.json)、[训练日志]({directory}/train.log)、[验证结果]({directory}/val_result.json)。']
    errors = [(r, r.get('status', {}).get('error') or r.get('test', {}).get('error')) for r in records]
    errors = [(r, e) for r, e in errors if e and (not usable(r) or r['test'].get('status') == 'failed')]
    if errors:
        lines += ['', '## 待处理问题', '']
        for r, error in errors:
            lines.append(f"- {r['model']} / t{r['trial']:03d}：{cell(error)}")
    lines += ['', '## 如何读这些指标', '',
              '- **mAP50-95**：综合多个框重叠阈值的检测指标，本报告的选优指标。',
              '- **mAP50**：只采用 IoU=0.5 的检测指标，通常比 mAP50-95 高。',
              '- **Precision**：预测出的目标中有多少是对的；**Recall**：真实目标中有多少被找到。',
              '- ★ 表示同一模型当前最优；“提升”仅与该模型 t000 的验证结果比较，不是测试集提升。',
              '- auto 是传入的配置，不代表固定 AdamW；实际优化器与学习率请查看 train.log。',
              '- 未逐候选测试时，测试列留空属正常；不能将缺失测试指标视为性能差。',
              '- n/s 仅按验证精度分别选优，不意味着已经验证部署速度或所有物种的召回率。',
              '- 阶段性报告会持续更新；完成报告也只代表本次搜索预算内的最优结果。', '',
              '机器可读明细：[iterations.csv](iterations.csv)、[summary.json](summary.json)。', '']
    temporary = root/'REPORT.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(root/'REPORT.md')
