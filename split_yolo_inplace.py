"""Transactional in-place publishing for split_yolo.py."""
import json
import os
from pathlib import Path
import shutil

from filelock import FileLock
import yaml

from merge_yolo_ids import label_for, load_dataset
from search_worker import write_json
from split_yolo import collect_images, dataset_file, split_dataset, target_counts

WORK = '.split_yolo_inplace'


def inside(root, path):
    root, resolved = root.resolve(), path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f'Unsafe in-place target outside dataset: {path}')
    return path


def remove(path, root):
    """Verify resolved targets immediately before recursive removal."""
    inside(root, path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def clean_work(root):
    work = root/WORK
    for name in ('old', 'prepared'):
        remove(work/name, root)
    for name in ('transaction.json.tmp', 'transaction.json'):
        (work/name).unlink(missing_ok=True)
    work.rmdir()


def recover(root):
    """Recover an interrupted rename sequence under the dataset's exclusive lock."""
    work = root/WORK
    if not work.exists():
        return
    if work.is_symlink():
        raise ValueError(f'Refusing a symlink work directory: {work}')
    inside(root, work)
    if not any(work.iterdir()):
        work.rmdir()
        return
    journal = work/'transaction.json'
    if not journal.is_file():
        raise ValueError(f'Unrecognized work directory; retained untouched: {work}')
    state = json.loads(journal.read_text(encoding='utf-8'))
    if state.get('root') != str(root) or state.get('version') != 1:
        raise ValueError(f'Invalid in-place transaction: {journal}')
    # Validate ALL journal paths before any restoration/deletion.
    for name in state['old'] + state['new']:
        inside(root, root/name)
        inside(root, work/'old'/name)
        inside(root, work/'prepared'/name)
        if (root/name).resolve().is_relative_to(work.resolve()):
            raise ValueError('Transaction may not publish its own work directory')
    if state['phase'] == 'committed':
        if not all((root/name).exists() for name in state['new']):
            raise ValueError(f'Committed output changed; retained recovery backup: {work}')
        clean_work(root)
        return
    if state['phase'] == 'switching':
        for name in reversed(state['new']):
            prepared, target = work/'prepared'/name, root/name
            # A missing prepared path means its rename may have completed even
            # if the process was killed before a follow-up journal write.
            if not prepared.exists() and target.exists():
                remove(target, root)
        for name in state['old']:
            backup, target = work/'old'/name, root/name
            if backup.exists():
                if target.exists():
                    raise ValueError(f'Recovery target occupied; backup retained: {target}')
                target.parent.mkdir(parents=True,exist_ok=True)
                backup.rename(target)
    elif state['phase'] != 'building':
        raise ValueError(f'Unknown transaction phase: {state["phase"]}')
    clean_work(root)


def inventory(root, images):
    trees = {root/name for name in ('images','labels') if (root/name).exists()}
    known = set()
    for image in images:
        for path, component in ((image,'images'), (label_for(image),'labels')):
            inside(root,path)
            if path.is_symlink():
                raise ValueError(f'In-place mode does not replace symlinks: {path}')
            relative = path.relative_to(root)
            # A root/images tree can include deeper images components.
            tree = root.joinpath(*relative.parts[:relative.parts.index(component)+1])
            inside(root,tree)
            trees.add(tree)
            known.add(path)
    trees = sorted((p for p in trees if not any(q != p and p.is_relative_to(q) for q in trees)),key=str)
    extras = []
    for tree in trees:
        if tree.is_symlink() or not tree.is_dir():
            raise ValueError(f'Expected a real dataset directory: {tree}')
        for path in tree.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'In-place mode does not replace symlinks: {path}')
            if path.is_file() and path not in known and path.suffix.lower() != '.cache':
                extras.append(path)
    return trees, extras


def publish(root, state):
    """All source backups and staged replacements stay on the same filesystem."""
    work = root/WORK
    state['phase'] = 'switching'
    write_json(work/'transaction.json',state)
    for name in state['old']:
        target = work/'old'/name
        inside(root, root/name)
        target.parent.mkdir(parents=True,exist_ok=True)
        (root/name).rename(target)
    for name in state['new']:
        inside(root,root/name)
        (root/name).parent.mkdir(parents=True,exist_ok=True)
        (work/'prepared'/name).rename(root/name)
    state['phase'] = 'committed'
    write_json(work/'transaction.json',state)


def split_in_place(yaml_path, ratios=(8,1,1), seed=42, workers=16):
    root = Path(yaml_path).resolve().parent
    with FileLock(str(root/'.split_yolo.lock'),timeout=0):
        recover(root)
        source = load_dataset(yaml_path)
        images, ignored = collect_images(source)
        trees, extras = inventory(root,images)
        for name in {'data.yaml',source['yaml'].name,'SPLIT_REPORT.md','split_report.json'}:
            path = root/name
            if path.exists() and not path.is_file():
                raise ValueError(f'Refusing to replace non-file metadata path: {path}')
        work = root/WORK
        work.mkdir(exist_ok=False)
        state = {'version':1,'root':str(root),'phase':'building','old':[],'new':[]}
        write_json(work/'transaction.json',state)
        try:
            prepared = work/'prepared'
            report = split_dataset(source,prepared,ratios,seed,workers,False,collected=(images,ignored))
            # Preserve non-dataset files and orphan labels, rather than losing
            # them when an old images/labels directory is replaced. Drop caches.
            extra_trees = set()
            for extra in extras:
                relative = extra.relative_to(root)
                target = prepared/relative
                if target.exists():
                    raise ValueError(f'Preserved metadata conflicts with new dataset: {relative}')
                target.parent.mkdir(parents=True,exist_ok=True)
                os.link(extra,target)
                for tree in trees:
                    if extra.is_relative_to(tree) and tree not in (root/'images',root/'labels'):
                        extra_trees.add(tree.relative_to(root).as_posix())
            config = yaml.safe_load((prepared/'data.yaml').read_text(encoding='utf-8'))
            config['path'] = root.as_posix()
            config_text = yaml.safe_dump(config,allow_unicode=True,sort_keys=False)
            (prepared/'data.yaml').write_text(config_text,encoding='utf-8')
            yaml_names = sorted({'data.yaml',source['yaml'].name})
            for name in yaml_names:
                (prepared/name).write_text(config_text,encoding='utf-8')
            report.update(output=str(root),in_place=True,preserved_auxiliary_files=len(extras))
            with (prepared/'SPLIT_REPORT.md').open('a',encoding='utf-8') as stream:
                stream.write(f'\n## 原地更新\n\n已直接更新 `{root}`；无独立输出数据集目录。'
                             f'附属文件保留 {len(extras)} 个，旧缓存清除。当前统计见 `split_report.json`；其他原有报告为历史记录。\n')
            write_json(prepared/'split_report.json',report)
            (prepared/'report.json').unlink()
            state['new'] = ['images','labels',*sorted(extra_trees),*yaml_names,'SPLIT_REPORT.md','split_report.json']
            old = {p.relative_to(root).as_posix() for p in trees}
            old.update(name for name in state['new'] if (root/name).exists())
            # Collapse nested backup paths (e.g. metadata under old images/).
            state['old'] = sorted(name for name in old if not any(
                name != other and Path(name).is_relative_to(Path(other)) for other in old))
            for name in state['old']+state['new']:
                inside(root,root/name)
            publish(root,state)
        except BaseException:
            recover(root)
            raise
        recover(root)  # committed: discard old image links and temporary labels.
        # Empty old split parents carry no data and can be removed non-recursively.
        for tree in trees:
            parent = tree.parent
            if parent != root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        return report


def run_in_place_batch(datasets, ratios=(8,1,1), seed=42, workers=16):
    if workers < 1:
        raise ValueError('workers must be positive')
    target_counts(10,ratios)
    values = [Path(value).expanduser().resolve() for value in datasets]
    roots = [p.parent if p.suffix.lower() in ('.yaml','.yml') else p for p in values]
    if len(set(roots)) != len(roots) or any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise ValueError('In-place inputs must be distinct, non-nested dataset directories')
    results = []
    for value,root in zip(values,roots):
        try:
            # A hard kill during publication can temporarily move the YAML into
            # backup. Recover BEFORE looking up data.yaml so the same command works.
            if (root/WORK).exists():
                with FileLock(str(root/'.split_yolo.lock'),timeout=0):
                    recover(root)
            path = dataset_file(value)
            report = split_in_place(path,ratios,seed,workers)
            results.append({'source':str(path),'status':'ok','report':report})
            print(f'Updated in place: {path.parent}; {report["total_images"]} images; report: {path.parent / "SPLIT_REPORT.md"}',flush=True)
        except (ValueError,OSError,UnicodeError,yaml.YAMLError) as exc:
            results.append({'source':str(value),'status':'failed','error':str(exc)})
            print(f'Failed: {value}: {exc}',flush=True)
    return results
