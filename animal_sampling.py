"""Configurable hierarchy projection and recursive fair whole-image sampling."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import re
from tqdm import tqdm

from animal_index import connect, fingerprint, stamp
from merge_yolo_split import resplit


def key(name):
    return re.sub(r'[\s-]+','_',name.strip().casefold())


def compile_policy(catalog, policy):
    nodes = catalog['nodes']
    paths = {}
    for node in nodes:
        chain,seen,current = [],set(),node
        while current:
            if current in seen or current not in nodes:
                raise ValueError(f'Cyclic or missing catalog parent: {node}/{current}')
            seen.add(current); chain.append(current)
            current = nodes[current].get('parent')
        paths[node] = chain
    groups = policy['groups']
    if not groups or any(not re.fullmatch(r'[a-z][a-z0-9_]*',g) for g in groups):
        raise ValueError('Output groups must have English snake_case names')
    expanded = {}
    for group,rule in groups.items():
        members = list(rule.get('members',[]))
        if rule.get('member_tag'):
            tagged = [n for n,v in nodes.items() if rule['member_tag'] in v.get('tags',[])]
            if not tagged:
                raise ValueError(f'Unknown/empty member tag: {rule["member_tag"]}')
            # A tagged parent already includes all its descendants; do not award
            # additional root-member quotas simply because children carry tags too.
            tagged = [n for n in tagged if not any(a!=n and a in paths[n] for a in tagged)]
            members = list(dict.fromkeys(members+sorted(tagged)))
        if not members or len(members)!=len(set(members)) or set(members+rule.get('exclude',[]))-nodes.keys():
            raise ValueError(f'Unknown/duplicate/empty members: {group}')
        if any(a!=b and a in paths[b] for a in members for b in members):
            raise ValueError(f'Overlapping ancestor/child members within {group}')
        expanded[group] = {**rule,'members':members}
    mapping = {}
    for node,chain in paths.items():
        matches = []
        for group,rule in expanded.items():
            if set(chain)&set(rule.get('exclude',[])):
                continue
            for member in rule['members']:
                if member in chain:
                    # A self bucket keeps coarse source annotations distinct from children.
                    matches.append((group,tuple([group]+list(reversed(chain[:chain.index(member)+1]))+['@self'])))
        if len(matches)>1:
            raise ValueError(f'Concept assigned to multiple output groups: {node}')
        if matches:
            mapping[node] = matches[0]
    return mapping


def pool(index, catalog, policy):
    mapping = compile_policy(catalog,policy)
    db = connect(index)
    try:
        sources = {r['id']:dict(r) for r in db.execute('SELECT * FROM sources ORDER BY name')}
        cats = {(r['source'],r['cid']):r['name'] for r in db.execute('SELECT * FROM categories')}
        annotations = defaultdict(list)
        for r in db.execute('SELECT * FROM boxes ORDER BY image,rowid'):
            annotations[r['image']].append((r['cid'],tuple(json.loads(r['coords']))))
        records, excluded = [],Counter()
        priority = policy.get('source_priority',['ap10k','animals','lila'])
        for image in db.execute('SELECT images.* FROM images JOIN sources ON sources.id=images.source ORDER BY sources.name,images.original_id'):
            source = sources[image['source']]
            spec = json.loads(source['spec'])
            scope = spec.get('catalog_scope',source['name'])
            projected,paths,origins = [],set(),set()
            bad = False
            for cid,box in annotations[image['id']]:
                raw = cats[image['source'],cid]
                scoped = catalog.get('source_names',{}).get(scope,{})
                concept = scoped.get(key(raw),key(raw) if key(raw) in catalog['nodes'] else None)
                if concept not in mapping:
                    excluded[f'{source["name"]}/{raw}'] += 1
                    bad = True
                    continue
                group,path = mapping[concept]
                paths.add(path); origins.add((source['name'],raw,concept,group))
                projected.append((group,box))
            if bad:  # Keep the established whole-image exclusion rule.
                continue
            if projected and spec.get('family')=='lila' and not any(
                    float(b[2])*float(b[3])>policy.get('lila_min_area',1/16) for _,b in projected):
                excluded['lila_area_filter'] += 1
                continue
            s = json.loads(image['stamp'])
            source_type = spec.get('family',source['name'])
            rank = priority.index(source_type) if source_type in priority else len(priority)
            token = hashlib.sha256(f'{source["name"]}/{image["original_id"]}'.encode()).hexdigest()
            records.append({'index_id':image['id'],'dataset':source['name'],'priority':rank,
                'image':Path(image['path']),'stamp':s,'image_stamp':{'device':s[0],'inode':s[1]},
                'split':'train','relative':image['original_id'],'rows':list(dict.fromkeys(projected)),
                'paths':paths,'origins':origins,'group_keys':[(source['name'],*g) for g in json.loads(image['groups_json'])],
                'train_only':bool(image['train_only']), 'image_rel':Path('images/train')/(token+Path(image['path']).suffix.lower()),
                'label_rel':Path('labels/train')/(token+'.txt')})
        if not records:
            raise ValueError('No eligible images; inspect source-name mappings and policy')
        resplit(records,.1,.1,policy.get('seed',42),groups_only=True)
        # Resolve known duplicate files before allocating sampling slots. Conflicting
        # annotations are excluded rather than silently choosing partial labels.
        identities = defaultdict(list)
        for r in records:
            s = r['stamp']
            identities[(s[0],s[1]) if s[1] else str(r['image'])].append(r)
        unique = []
        for group in identities.values():
            if len({tuple(sorted(r['rows'])) for r in group})>1:
                excluded['duplicate_annotation_conflict_images'] += len(group)
                continue
            winner = min(group,key=lambda r:(r['priority'],r['dataset'],r['relative']))
            unique.append(winner)
            excluded['duplicate_images'] += len(group)-1
        errors = [dict(r) for r in db.execute('SELECT sources.name AS source,errors.path,errors.error FROM errors JOIN sources ON sources.id=errors.source')]
        return unique,dict(excluded),errors,sources
    finally:
        db.close()


def choose(records, maximum, negative_fraction, seed):
    if maximum<1 or not 0<=negative_fraction<1:
        raise ValueError('Invalid image budget/negative fraction')
    rng = random.Random(seed)
    children,leaves,available = defaultdict(set),defaultdict(list),Counter()
    for i,r in enumerate(records):
        prefixes = set()
        for path in r['paths']:
            leaves[path].append(i)
            for n in range(1,len(path)+1):
                prefixes.add(path[:n])
                children[path[:n-1]].add(path[:n])
        available.update(prefixes)
    queues = {}
    for path,indices in leaves.items():
        buckets = defaultdict(lambda:defaultdict(list))
        for i in indices:
            buckets[records[i]['priority']][records[i]['component']].append(i)
        groups = []
        for rank in sorted(buckets):
            part = list(buckets[rank].values()); rng.shuffle(part)
            for values in part: rng.shuffle(values)
            groups.append(deque(deque(values) for values in part))
        queues[path] = groups
    selected,counts = set(),Counter()
    active = {p:set(v) for p,v in children.items()}

    def pop(path):
        if path in queues:
            for groups in queues[path]:
                while groups:
                    values = groups.popleft()
                    while values and values[0] in selected: values.popleft()
                    if values:
                        value = values.popleft()
                        if values: groups.append(values)
                        return value
            return None
        while active.get(path):
            candidates = sorted(active[path])
            low = min(counts[p] for p in candidates)
            branch = rng.choice([p for p in candidates if counts[p]==low])
            value = pop(branch)
            if value is not None: return value
            active[path].remove(branch)
        return None

    negatives = [i for i,r in enumerate(records) if not r['rows']]
    rng.shuffle(negatives)
    negative_budget = min(len(negatives),int(maximum*negative_fraction))
    while len(selected)<maximum-negative_budget:
        i = pop(())
        if i is None: break
        selected.add(i)
        prefixes = {p[:n] for p in records[i]['paths'] for n in range(1,len(p)+1)}
        counts.update(prefixes)
    selected.update(negatives[:negative_budget])
    picked = [r for i,r in enumerate(records) if i in selected]
    return picked,available,counts


def plan(index,catalog,policy,maximum):
    records,excluded,errors,sources = pool(index,catalog,policy)
    chosen,available,counts = choose(records,maximum,policy.get('negative_fraction',.05),policy.get('seed',42))
    details = [{'path':list(p[:-1] if p[-1]=='@self' else p),
                'source_bucket':p[-1]=='@self','available_images':n,'selected_images':counts[p]}
               for p,n in sorted(available.items())]
    for group in policy['groups']:
        if (group,) not in available:
            details.append({'path':[group],'source_bucket':False,'available_images':0,'selected_images':0})
    provenance = defaultdict(lambda:{'available_images':0,'selected_images':0})
    chosen_ids = {r['index_id'] for r in chosen}
    for r in records:
        for source,raw,concept,group in r['origins']:
            row = provenance[group,concept,source,raw]
            row['available_images'] += 1
            row['selected_images'] += r['index_id'] in chosen_ids
    db = connect(index)
    try:
        inventory = [dict(r) for r in db.execute('''WITH n AS (
          SELECT i.source,b.cid,count(DISTINCT i.id) AS images,count(*) AS boxes
          FROM boxes b JOIN images i ON i.id=b.image GROUP BY i.source,b.cid)
          SELECT s.name AS source,c.cid,c.name AS original_name,coalesce(n.images,0) AS images,
            coalesce(n.boxes,0) AS boxes FROM categories c JOIN sources s ON s.id=c.source
            LEFT JOIN n ON n.source=c.source AND n.cid=c.cid ORDER BY s.name,c.cid''')]
    finally:
        db.close()
    return chosen,{'maximum_images':maximum,'selected_images':len(chosen),'hierarchy_sampling':details,
        'requested_groups':list(policy['groups']),
        'unselected_groups':[g for g in policy['groups'] if not counts[(g,)]],
        'raw_category_inventory':inventory,
        'provenance':[dict(group=k[0],concept=k[1],source=k[2],original_name=k[3],**v) for k,v in sorted(provenance.items())],
        'excluded':excluded,'index_errors':errors,'source_priority':policy.get('source_priority',[]),
        'scope':'Counts are image memberships; sums may exceed unique images. No image decoding/content hash.'},sources


def export(index,catalog,policy,maximum,output,start_id=0):
    output = Path(output).resolve()
    if start_id<0 or output.exists():
        raise ValueError('Use a new output directory and nonnegative class-id-start')
    chosen,report,sources = plan(index,catalog,policy,maximum)
    if not chosen:
        raise ValueError('No selected images')
    # Verify source annotation freshness once; never re-decode images.
    for source in sources.values():
        spec = json.loads(source['spec'])
        if fingerprint(spec)!=source['fingerprint']:
            raise ValueError(f'Source changed; update index first: {source["name"]}')
    for r in tqdm(chosen,desc='Check selected file stamps',unit='img'):
        if stamp(r['image'])!=r['stamp']:
            raise ValueError(f'Image changed; refresh index: {r["image"]}')
        if output==r['image'].parent or output in r['image'].parents or r['image'].is_relative_to(output):
            raise ValueError('Output must not overlap indexed images')
    labels = sorted({n for r in chosen for n,_ in r['rows']})
    if not labels:
        raise ValueError('No positive output classes; cannot export a zero-class training dataset')
    ids = {n:i+start_id for i,n in enumerate(labels)}
    splitting = resplit(chosen,policy.get('val_fraction',.1),policy.get('test_fraction',.1),policy.get('seed',42))
    output.mkdir(parents=True)
    (output/'INCOMPLETE.txt').write_text('Do not train until REPORT.md and data.yaml exist.\n')
    audit_rows = 0
    evidence_db = connect(index)
    try:
        for r in chosen:
            for box in evidence_db.execute('SELECT * FROM boxes WHERE image=?',(r['index_id'],)):
                if not box['evidence']:
                    raise ValueError('Index lacks original box evidence; rebuild with index --refresh')
                evidence = json.loads(box['evidence'])
                if evidence['format']=='coco':
                    x,y,w,h = map(float,evidence['bbox']); W,H = evidence['width'],evidence['height']
                    expected = [(x+w/2)/W,(y+h/2)/H,w/W,h/H]
                else:
                    expected = list(map(float,evidence['row'].split()[1:]))
                if any(abs(a-float(b))>1e-10 for a,b in zip(expected,json.loads(box['coords']),strict=True)):
                    raise ValueError('Original bbox evidence does not match indexed coordinates')
    finally:
        evidence_db.close()
    for r in tqdm(chosen,desc='Write and audit hardlinks/labels',unit='img'):
        image,label = output/r['image_rel'],output/r['label_rel']
        image.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
        os.link(r['image'],image)
        label.write_text(''.join(f'{ids[n]} '+ ' '.join(b)+'\n' for n,b in r['rows']),encoding='utf-8')
        actual = [(int(line.split()[0]),tuple(line.split()[1:])) for line in label.read_text().splitlines()]
        if actual!=[(ids[n],b) for n,b in r['rows']] or not os.path.samefile(image,r['image']):
            raise ValueError('Independent written label/link audit failed')
        audit_rows += len(actual)
    with (output/'selection.jsonl').open('w',encoding='utf-8') as stream:
        for r in chosen:
            stream.write(json.dumps({'image':r['image_rel'].as_posix(),'label':r['label_rel'].as_posix(),
                'source':r['dataset'],'original_id':r['relative'],'source_image':str(r['image']),
                'index_image_id':r['index_id'],'component':r['component'],
                'original_classes':sorted(r['origins']),'output_boxes':[(ids[n],b) for n,b in r['rows']]},ensure_ascii=False)+'\n')
    report.update(names=ids,split=splitting,audited_boxes=audit_rows,
        original_box_evidence_audited=True,
        index=str(Path(index).resolve()),source_snapshots=[json.loads(s['spec'])|{'fingerprint':s['fingerprint']} for s in sources.values()],
        catalog=catalog,policy=policy)
    names = {i:f'reserved_{i}' for i in range(start_id)} | {v:k for k,v in ids.items()}
    config = {'path':str(output),'train':'images/train','val':'images/val','test':'images/test','names':names,'nc':len(names)}
    import yaml
    (output/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write_preview(report,output/'REPORT.md')
    (output/'INCOMPLETE.txt').unlink()
    return report


def write_preview(report,path):
    lines = ['# 通用动物数据集采样报告','',f'实际选择 {report["selected_images"]} 张，上限 {report["maximum_images"]} 张。','',
        f'请求类别数：{len(report["requested_groups"])}；没有抽到样本的类别：{report["unselected_groups"]}。','',
        '按最终类别→成员→更细类别递归均衡；多类别同图只保存一次，图片计数可重叠。','',
        '| 层级路径 | 原始标签采样桶 | 可用图片 | 选择图片 |','| --- | --- | ---: | ---: |']
    for r in report['hierarchy_sampling']:
        lines.append(f'| {" / ".join(r["path"])} | {r["source_bucket"]} | {r["available_images"]} | {r["selected_images"]} |')
    if 'names' in report:
        lines += ['','## 最终训练类别与划分','', '| ID | 标签 | train 图片 | val 图片 | test 图片 |','| ---: | --- | ---: | ---: | ---: |']
        for name,cid in report['names'].items():
            counts = report['split']['class_image_counts'].get(name,{})
            lines.append(f'| {cid} | {name} | {counts.get("train",0)} | {counts.get("val",0)} | {counts.get("test",0)} |')
        lines += ['',f'审计框数：{report["audited_boxes"]}；原始框证据核对：{report["original_box_evidence_audited"]}；分组跨集合重叠：{report["split"]["group_overlap"]}。']
    lines += ['','## 索引中的全部原始类别','', '以下为有效索引记录的原始类别数量，尚未应用输出方案；问题图片见 index_errors。','',
        '| 来源 | 原始 ID | 原始名称 | 图片数 | 框数 |','| --- | --- | --- | ---: | ---: |']
    for r in report['raw_category_inventory']:
        values = [str(r[k]).replace('|','\\|') for k in ('source','cid','original_name','images','boxes')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['','## 原始细类与来源','', '| 输出类 | 通用细类 | 来源 | 原始类别 | 可用 | 选取 |','| --- | --- | --- | --- | ---: | ---: |']
    for r in report.get('provenance',[]):
        values = [str(r[k]).replace('|','\\|') for k in ('group','concept','source','original_name','available_images','selected_images')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['','## 排除与限制','',json.dumps(report['excluded'],ensure_ascii=False),'',
        '索引问题记录数：'+str(len(report['index_errors'])),
        '粗标注不猜测细类；来源优先不是清晰度评分；不保证未知相邻帧或独立复制图片无泄漏。',
        '分组划分为启发式近似比例，缺失集合/类别详见 report.json。']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
