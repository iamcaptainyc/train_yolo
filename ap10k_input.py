"""Read one AP-10K COCO split scheme directly; normalize boxes without image decoding."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from lila_state import fingerprint, same_content_attributes

# Explicit AP-10K original labels. Ambiguous or without a reviewed common-name mapping are excluded.
AP_NAMES = {
 'antelope':'antelope','argali sheep':'sheep','bison':None,'buffalo':'buffalo','cow':'cattle','sheep':'sheep',
 'arctic fox':'fox','dog':'dog','fox':'fox','wolf':'wolf','beaver':None,
 'alouatta':'monkey','monkey':'monkey','noisy night monkey':'monkey','spider monkey':'monkey','uakari':'monkey',
 'deer':'deer','moose':'deer','hamster':'hamster','elephant':'elephant','horse':'horse','zebra':'zebra',
 'bobcat':'lynx','cat':'cat','cheetah':'cheetah','jaguar':'jaguar','king cheetah':'cheetah',
 'leopard':'leopard','lion':'lion','panther':None,'snow leopard':'leopard','tiger':'tiger','giraffe':'giraffe',
 'hippo':'hippopotamus','chimpanzee':None,'gorilla':None,'orangutan':None,'rabbit':'rabbit',
 'skunk':'skunk','mouse':'mouse','rat':'mouse','otter':'otter','weasel':'weasel','raccoon':'raccoon',
 'rhino':'rhinoceros','marmot':'marmot','squirrel':'squirrel','pig':'pig','mole':None,
 'black bear':'black_bear','brown bear':'brown_bear','panda':'panda','polar bear':'polar_bear','bat':'bat',
}


def load_ap10k(root, scheme=1):
    root = Path(root).expanduser().resolve()
    candidates = [root, root/'ap-10k', root/'ap10k']
    matches = [p for p in candidates if (p/'annotations').is_dir() and (p/'data').is_dir()]
    if len(matches) != 1:
        raise ValueError(f'Expected one AP-10K root containing annotations/ and data/: {root}')
    root = matches[0]
    files = {s: root/'annotations'/f'ap10k-{s}-split{scheme}.json' for s in ('train','val','test')}
    original, mapping, seen = {}, {}, set()
    loaded, stamps = {}, {}
    for split, path in files.items():
        before = fingerprint(path)
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not same_content_attributes(before, fingerprint(path)):
            raise ValueError(f'Annotation changed during read: {path}')
        for cat in value['categories']:
            cid, raw = cat['id'], cat['name']
            if raw not in AP_NAMES:
                raise ValueError(f'Unreviewed AP-10K category: {raw!r}')
            if cid in original and original[cid] != raw:
                raise ValueError('AP-10K category IDs disagree across the selected split scheme')
            original[cid], mapping[cid] = raw, AP_NAMES[raw]
        by_image = defaultdict(list)
        for ann in value['annotations']:
            by_image[ann['image_id']].append(ann)
        known = {im['id'] for im in value['images']}
        if set(by_image)-known:
            raise ValueError(f'Annotations refer to unknown AP-10K image IDs: {path}')
        for image in value['images']:
            key = image['id']
            if key in seen:
                raise ValueError('Repeated AP-10K image ID across selected train/val/test; do not mix split schemes')
            seen.add(key)
        loaded[split] = [(im, by_image[im['id']]) for im in value['images']]
        stamps[split] = before
    manifest = {s: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for s,p in files.items()}
    return {'alias':'ap10k', 'native_ap10k':True, 'root':root, 'yaml':files['train'],
            'yaml_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
            'manifest':manifest,'scheme':scheme,'directories':{s:root/'data' for s in files},
            'original_names':original,'mapping':mapping,'annotation_files':files,'stamps':stamps,'loaded':loaded}


def rows_from_coco(image, annotations, original, mapping):
    """Independent bbox transformation used by scan and output audit."""
    width, height = image['width'], image['height']
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (width,height)):
        raise ValueError('Invalid AP-10K image dimensions')
    if not annotations:
        raise ValueError('AP-10K image has no animal annotations; never infer a verified negative')
    rows = []
    for ann in annotations:
        cid = ann['category_id']
        if cid not in original:
            raise ValueError('Unknown AP-10K category ID')
        target = mapping[cid]
        if target is None:
            raise ValueError(f'AP-10K unsupported/ambiguous category; exclude WHOLE image: {original[cid]}')
        if ann.get('iscrowd') or ann.get('ignore'):
            raise ValueError('AP-10K crowd/ignored annotation; exclude whole image, do not drop its box')
        box = ann['bbox']
        if len(box) != 4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
            raise ValueError('Invalid AP-10K bbox')
        x,y,w,h = box
        if min(x,y)<0 or min(w,h)<=0 or x+w>width+1e-6 or y+h>height+1e-6:
            raise ValueError('AP-10K bbox outside image bounds; no silent clipping')
        coords = ((x+w/2)/width,(y+h/2)/height,w/width,h/height)
        rows.append((target,tuple(f'{v:.10f}' for v in coords)))
    return rows


def scan_ap10k(task):
    from merge_yolo_datasets import unique_rows
    dataset, split, image, anns = task
    root = dataset['root']/'data'
    relative = Path(image['file_name'])
    path = (root/relative).resolve()
    if relative.is_absolute() or root not in path.parents:
        raise ValueError('Unsafe AP-10K image path')
    stem = hashlib.sha256(f'ap10k/{split}/{relative.as_posix()}'.encode()).hexdigest()
    record = {'dataset':'ap10k','split':split,'relative':relative.as_posix(),'image':path,
              'label':dataset['annotation_files'][split], 'label_stamp':dataset['stamps'][split],
              'image_rel':Path('images')/split/'ap10k'/(stem+path.suffix.lower()),
              'label_rel':Path('labels')/split/'ap10k'/(stem+'.txt'),
              'ap_image':image, 'ap_annotations':anns,
              'ap_original_names':dataset['original_names'], 'ap_mapping':dataset['mapping']}
    try:
        record['image_stamp'] = fingerprint(path)
        if record['image_stamp']['size']<=0:
            raise ValueError('Empty AP-10K image')
        raw = rows_from_coco(image,anns,dataset['original_names'],dataset['mapping'])
        record['rows'] = unique_rows(raw)
        record['duplicate_boxes_removed'] = len(raw)-len(record['rows'])
        origins = defaultdict(set)
        for ann in anns:
            row = rows_from_coco(image,[ann],dataset['original_names'],dataset['mapping'])[0]
            origins[row].add(dataset['original_names'][ann['category_id']])
        record['original_row_names'] = {row:sorted(origins[row]) for row in record['rows']}
        record['group_keys'] = []
        for key in ('sequence_id','seq_id','video_id','camera_id','location'):
            if image.get(key) is not None:
                record['group_keys'].append(('ap10k_metadata',key,str(image[key])))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        record['error'] = str(exc)
    return record
