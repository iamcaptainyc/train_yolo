"""Generic animal inventory, configurable preview and hierarchical YOLO sampling."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from filelock import FileLock
import yaml

from animal_index import build
from animal_sampling import export, plan, write_preview, refresh_report

ROOT = Path(__file__).parent


def area_ratio(value):
    try:
        ratio = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise argparse.ArgumentTypeError('Use a decimal or fraction, e.g. 0.0625 or 1/16') from exc
    if not 0 <= ratio < 1:
        raise argparse.ArgumentTypeError('LILA bbox area ratio must be >= 0 and < 1')
    return ratio


def read(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def specs_from_args(args):
    specs = []
    if args.sources:
        manifest = read(args.sources)
        base = args.sources.resolve().parent
        for entry in manifest['sources']:
            spec = dict(entry)
            for field in ('yaml','images'):
                if field in spec:
                    p = Path(spec[field]).expanduser()
                    spec[field] = str((base/p).resolve() if not p.is_absolute() else p.resolve())
            if 'annotations' in spec:
                spec['annotations'] = [str((base/Path(p)).resolve()) if not Path(p).is_absolute() else str(Path(p).resolve()) for p in spec['annotations']]
            specs.append(spec)
    if args.ap10k:
        candidates = [p for p in (args.ap10k,args.ap10k/'ap-10k',args.ap10k/'ap10k') if (p/'annotations').is_dir() and (p/'data').is_dir()]
        if len(candidates)!=1:
            raise ValueError('Expected one AP-10K root containing annotations/ and data/')
        root = candidates[0].resolve()
        specs.append({'name':'ap10k','family':'ap10k','kind':'coco','catalog_scope':'ap10k','images':str(root/'data'),
            'annotations':[str(root/'annotations'/f'ap10k-{s}-split{args.ap10k_split}.json') for s in ('train','val','test')]})
    if args.animals:
        path = args.animals.resolve()
        specs.append({'name':'animals','family':'animals','kind':'yolo','yaml':str(path/'data.yaml' if path.is_dir() else path)})
    if args.lila_source:
        matches = sorted(args.lila_source.resolve().glob('*/selection/selected_annotations.json'))
        if not matches:
            raise ValueError('No LILA */selection/selected_annotations.json found')
        for path in matches:
            root = path.parent.parent
            specs.append({'name':root.name,'catalog_scope':root.name,'family':'lila','kind':'coco',
                'images':str(root/'images'),'annotations':[str(path)],'require_groups':True})
    if not specs:
        raise ValueError('Provide --sources and/or --ap10k/--animals/--lila-source')
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    report = sub.add_parser('report',help='Refresh reports only; no image scan or sampling')
    report.add_argument('--dataset',type=Path,required=True)
    index = sub.add_parser('index',help='Build/update raw-source SQLite inventory')
    index.add_argument('--index',type=Path,required=True)
    index.add_argument('--sources',type=Path,help='Generic source manifest YAML')
    index.add_argument('--ap10k',type=Path)
    index.add_argument('--ap10k-split',type=int,choices=[1,2,3],default=1)
    index.add_argument('--animals',type=Path)
    index.add_argument('--lila-source',type=Path)
    index.add_argument('--refresh',action='store_true',help='Rebuild even unchanged source metadata; use after image changes')
    for command in ('preview','sample'):
        parser = sub.add_parser(command)
        parser.add_argument('--index',type=Path,required=True)
        parser.add_argument('--catalog',type=Path,default=ROOT/'configs/animal_catalog.json')
        parser.add_argument('--policy',type=Path,default=ROOT/'configs/camping19.yaml')
        parser.add_argument('--max-images',type=int,default=30000)
        parser.add_argument('--lila-min-area', '--lila-ratio', '--lila_ratio',
                            dest='lila_min_area', type=area_ratio, default=None,
                            help='LILA bbox/image area threshold, e.g. 1/16; overrides policy, not a source quota')
        parser.add_argument('--output',type=Path,required=True,help='New output directory (preview: reports only)')
        if command=='sample':
            parser.add_argument('--class-id-start',type=int,default=0)
            parser.add_argument('--repair-images',action=argparse.BooleanOptionalAction,default=True,
                                help='Check/repair each selected image while writing (default: enabled)')
    args = p.parse_args()
    try:
        if args.command=='report':
            refresh_report(args.dataset)
            print(args.dataset/'REPORT.md')
            return
        if args.command=='index':
            args.index.parent.mkdir(parents=True,exist_ok=True)
            with FileLock(str(args.index)+'.lock',timeout=0):
                print(json.dumps(build(args.index,specs_from_args(args),args.refresh),ensure_ascii=False,indent=2))
            return
        if not args.index.is_file():
            raise ValueError('Index does not exist; run index first')
        if args.output.exists() or args.max_images<1:
            raise ValueError('Use a new output directory and positive max-images')
        catalog,policy = read(args.catalog),read(args.policy)
        if args.lila_min_area is not None:
            policy['lila_min_area'] = args.lila_min_area
        v,t = policy.get('val_fraction',.1),policy.get('test_fraction',.1)
        if not (0<=v<1 and 0<=t<1 and v+t<1):
            raise ValueError('Invalid val/test fractions')
        with FileLock(str(args.index)+'.lock',timeout=0):
            if args.command=='sample':
                report = export(args.index,catalog,policy,args.max_images,args.output,args.class_id_start,
                                repair_images=args.repair_images)
            else:
                _,report,_ = plan(args.index,catalog,policy,args.max_images)
                args.output.mkdir(parents=True)
                write_preview(report,args.output/'REPORT.md')
                (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{args.command}: {report["selected_images"]} images; {args.output/"REPORT.md"}')
    except KeyboardInterrupt:
        p.exit(130,'\nInterrupted. Committed index sources remain valid; rerun index. Use new output for sample retry.\n')
    except (ValueError,OSError,KeyError,TypeError,yaml.YAMLError) as exc:
        p.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()
