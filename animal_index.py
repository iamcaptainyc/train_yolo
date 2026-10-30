"""Policy-independent raw annotation inventory; no image decoding or byte hashing."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from tqdm import tqdm

from merge_yolo_ids import images_for, label_for, load_dataset, validate_label


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,name TEXT UNIQUE,spec TEXT,fingerprint TEXT);
      CREATE TABLE IF NOT EXISTS categories(source INTEGER REFERENCES sources(id) ON DELETE CASCADE,
        cid TEXT,name TEXT,PRIMARY KEY(source,cid));
      CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY,source INTEGER REFERENCES sources(id) ON DELETE CASCADE,
        original_id TEXT,path TEXT,groups_json TEXT,train_only INTEGER,stamp TEXT);
      CREATE TABLE IF NOT EXISTS boxes(image INTEGER REFERENCES images(id) ON DELETE CASCADE,cid TEXT,coords TEXT,evidence TEXT);
      CREATE INDEX IF NOT EXISTS boxes_image ON boxes(image);
      CREATE TABLE IF NOT EXISTS errors(source INTEGER REFERENCES sources(id) ON DELETE CASCADE,path TEXT,error TEXT);
    ''')
    if 'evidence' not in {r['name'] for r in db.execute('PRAGMA table_info(boxes)')}:
        db.execute('ALTER TABLE boxes ADD COLUMN evidence TEXT')
    return db


def stamp(path):
    s = Path(path).stat()
    if s.st_size <= 0:
        raise ValueError(f'Empty image: {path}')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def fingerprint(spec):
    parts = [spec]
    if spec['kind']=='yolo':
        source = load_dataset(spec['yaml'])
        parts.append(hashlib.sha256(source['yaml'].read_bytes()).hexdigest())
        for split in ('train','val','test'):
            for image in images_for(source,split):
                label = label_for(image)
                s = label.stat()
                parts.append((str(image),str(label),s.st_size,s.st_mtime_ns))
    else:
        for path in spec['annotations']:
            parts.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()


def coords(values):
    if len(values)!=4:
        raise ValueError('Expected four box coordinates')
    x,y,w,h = map(float,values)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or max(w,h)>1 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
        raise ValueError('Invalid normalized bbox')
    return [str(v) for v in values]


def raw_records(spec, categories):
    """Yield original class IDs and boxes, without applying any camping policy."""
    if spec['kind']=='yolo':
        source = load_dataset(spec['yaml'])
        categories.update({str(k):v for k,v in source['names'].items()})
        seen = set()
        for split in ('train','val','test'):
            for image in images_for(source,split):
                key = str(image.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    raw,_ = validate_label(label_for(image),source)
                    boxes = [(line.split()[0],coords(line.split()[1:]),{'format':'yolo','row':line,'label':str(label_for(image))})
                             for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
                    yield key,image,[],False,boxes,None
                except (ValueError,OSError,UnicodeError) as exc:
                    yield key,image,[],False,[],str(exc)
        return
    root = Path(spec['images']).resolve()
    seen = set()
    for annotation in spec['annotations']:
        data = json.loads(Path(annotation).read_text(encoding='utf-8-sig'))
        for c in data['categories']:
            key = str(c['id'])
            if key in categories and categories[key]!=c['name']:
                raise ValueError('Category IDs disagree between annotation files')
            categories[key] = c['name']
        by_image = defaultdict(list)
        known = {str(im['id']) for im in data['images']}
        for a in data['annotations']:
            if str(a['image_id']) not in known:
                raise ValueError('Annotation references an unknown image')
            by_image[str(a['image_id'])].append(a)
        for im in data['images']:
            key = str(im['id'])
            if key in seen:
                raise ValueError('Repeated image ID across annotation files; choose one split scheme')
            seen.add(key)
            image = (root/im['file_name']).resolve()
            groups = [(field,str(im[field])) for field in ('camera_id','location','sequence_id','seq_id','video_id') if im.get(field) is not None]
            try:
                if not image.is_relative_to(root):
                    raise ValueError('Image path escapes source root')
                annotations = by_image[key]
                if not annotations and not (spec.get('empty_images_are_negative',False) or im.get('selection_role')=='negative'):
                    raise ValueError('Image without boxes is not a confirmed negative')
                rows = []
                for a in annotations:
                    cid = str(a['category_id'])
                    if cid not in categories or a.get('iscrowd') or a.get('ignore'):
                        raise ValueError('Unknown/crowd/ignored box; entire image excluded')
                    W,H = float(im['width']),float(im['height'])
                    if not math.isfinite(W+H) or min(W,H)<=0:
                        raise ValueError('Invalid image dimensions')
                    x,y,w,h = map(float,a['bbox'])
                    rows.append((cid,coords([f'{(x+w/2)/W:.12f}',f'{(y+h/2)/H:.12f}',f'{w/W:.12f}',f'{h/H:.12f}']),
                        {'format':'coco','bbox':a['bbox'],'width':W,'height':H,'annotation_file':str(annotation),'annotation_id':a.get('id')}))
                yield key,image,groups,bool(spec.get('require_groups') and not groups),rows,None
            except (ValueError,KeyError,TypeError,OverflowError) as exc:
                yield key,image,groups,False,[],str(exc)


def build(path, specs, refresh=False):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    names = [s['name'] for s in specs]
    if len(names)!=len(set(names)):
        raise ValueError('Source names must be unique')
    db = connect(path)
    results = []
    try:
        for spec in specs:
            if spec['kind'] not in ('yolo','coco'):
                raise ValueError('Source kind must be yolo or coco')
            digest = fingerprint(spec)
            old = db.execute('SELECT * FROM sources WHERE name=?',(spec['name'],)).fetchone()
            if old and old['fingerprint']==digest and not refresh:
                results.append({'source':spec['name'],'status':'unchanged'})
                continue
            with db:
                db.execute('DELETE FROM sources WHERE name=?',(spec['name'],))
                sid = db.execute('INSERT INTO sources(name,spec,fingerprint) VALUES(?,?,?)',
                    (spec['name'],json.dumps(spec),digest)).lastrowid
                cats,n,errors = {},0,0
                for key,image,groups,train_only,boxes,error in tqdm(raw_records(spec,cats),desc=f'Index {spec["name"]}',unit='img'):
                    try:
                        if error:
                            raise ValueError(error)
                        s = stamp(image)
                        iid = db.execute('INSERT INTO images(source,original_id,path,groups_json,train_only,stamp) VALUES(?,?,?,?,?,?)',
                            (sid,key,str(image),json.dumps(groups),int(train_only),json.dumps(s))).lastrowid
                        db.executemany('INSERT INTO boxes VALUES(?,?,?,?)',[(iid,cid,json.dumps(box),json.dumps(evidence)) for cid,box,evidence in boxes])
                        n += 1
                    except (OSError,ValueError) as exc:
                        errors += 1
                        db.execute('INSERT INTO errors VALUES(?,?,?)',(sid,str(image),str(exc)))
                db.executemany('INSERT INTO categories VALUES(?,?,?)',[(sid,cid,name) for cid,name in cats.items()])
                if fingerprint(spec)!=digest:
                    raise ValueError('Source annotations changed during indexing')
            results.append({'source':spec['name'],'status':'indexed','images':n,'errors':errors})
    finally:
        db.close()
    return results
