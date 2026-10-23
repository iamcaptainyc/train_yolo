"""Extract up to 30,000 AP-10K/Animals-first images with the reviewed 68 camping classes."""
import argparse
from pathlib import Path
from fractions import Fraction

from merge_yolo_datasets import merge


def area_ratio(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError('Use a decimal or fraction, e.g. 0.0625 or 1/16') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ap10k', type=Path, required=True, help='AP-10K root with annotations/ and data/; one enclosing ap-10k/ level is also accepted')
    parser.add_argument('--ap10k-split', type=int, choices=[1,2,3], default=1, help='Choose ONE annotation split scheme; its train/val/test are pooled')
    parser.add_argument('--lila-min-area', type=area_ratio, default=1/16, help='Strict LILA bbox area ratio threshold; e.g. 1/16 or 0.0625')
    parser.add_argument('--lila', type=Path, default=Path('yolo_datasets/lila_camping'))
    parser.add_argument('--animals', type=Path, default=Path('yolo_datasets/animals-detection-images-dataset'))
    parser.add_argument('--lila-source', type=Path, default=Path('datasets/lila'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--flat-images', action='store_true', help='Put images and labels directly under each split, with unique source-prefixed filenames')
    parser.add_argument('--class-id-start', type=int, default=0, help='First animal class ID; preceding IDs are reserved in data.yaml')
    parser.add_argument('--max-images', type=int, default=30000, help='Hard total limit INCLUDING negatives; at most 30000')
    parser.add_argument('--class-target', type=int, default=1200, help='Soft image target per class, not a rare-class merging threshold')
    parser.add_argument('--negative-fraction', type=float, default=.05)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--test-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--group-map', type=Path)
    args = parser.parse_args()
    if args.class_id_start < 0:
        parser.error('--class-id-start must be a nonnegative integer')
    if args.output is None:
        suffix = f'_id{args.class_id_start}' if args.class_id_start else ''
        suffix += '_flat' if args.flat_images else ''
        args.output = Path('yolo_datasets')/f'camping68_ap10k_30k{suffix}'
    if not 1 <= args.max_images <= 30000:
        parser.error('--max-images must be between 1 and 30000')
    if not 0 <= args.lila_min_area < 1:
        parser.error('--lila-min-area must be >=0 and <1')
    if args.class_target < 1 or not 0 <= args.negative_fraction <= .2:
        parser.error('--class-target must be positive; --negative-fraction must be between 0 and 0.2')
    try:
        merge(args.lila, args.animals, args.output, workers=args.workers,
              ap10k=args.ap10k, ap10k_split=args.ap10k_split,
              min_class_images=0, max_class_images=0, cap_class_images=0,
              val_fraction=args.val_fraction, test_fraction=args.test_fraction,
              lila_source=args.lila_source, group_map=args.group_map, seed=args.seed,
              subset_options={'max_images': args.max_images, 'negative_fraction': args.negative_fraction,
                              'class_target': args.class_target, 'lila_min_area': args.lila_min_area,
                              **({'class_id_start': args.class_id_start} if args.class_id_start else {}),
                              **({'flat_images': True} if args.flat_images else {})})
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Rerun the same command to resume; source images were not modified.\n')
    except Exception as exc:
        parser.exit(1, f'Error: {exc}\nSee output/PROCESSING_REPORT.md and excluded_images.jsonl.\n')


if __name__ == '__main__':
    main()
