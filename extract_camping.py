"""Extract up to 30,000 Animals-first images with the reviewed 82 camping classes."""
import argparse
from pathlib import Path

from merge_yolo_datasets import merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lila', type=Path, default=Path('yolo_datasets/lila_camping'))
    parser.add_argument('--animals', type=Path, default=Path('yolo_datasets/animals-detection-images-dataset'))
    parser.add_argument('--lila-source', type=Path, default=Path('datasets/lila'))
    parser.add_argument('--output', type=Path, default=Path('yolo_datasets/camping82_30k'))
    parser.add_argument('--max-images', type=int, default=30000, help='Hard total limit INCLUDING negatives; at most 30000')
    parser.add_argument('--class-target', type=int, default=1200, help='Soft image target per class, not a rare-class merging threshold')
    parser.add_argument('--negative-fraction', type=float, default=.05)
    parser.add_argument('--val-fraction', type=float, default=.05)
    parser.add_argument('--test-fraction', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--group-map', type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_images <= 30000:
        parser.error('--max-images must be between 1 and 30000')
    if args.class_target < 1 or not 0 <= args.negative_fraction <= .2:
        parser.error('--class-target must be positive; --negative-fraction must be between 0 and 0.2')
    try:
        merge(args.lila, args.animals, args.output, workers=args.workers,
              min_class_images=0, max_class_images=0, cap_class_images=0,
              val_fraction=args.val_fraction, test_fraction=args.test_fraction,
              lila_source=args.lila_source, group_map=args.group_map, seed=args.seed,
              subset_options={'max_images': args.max_images, 'negative_fraction': args.negative_fraction,
                              'class_target': args.class_target})
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Rerun the same command to resume; source images were not modified.\n')
    except Exception as exc:
        parser.exit(1, f'Error: {exc}\nSee output/PROCESSING_REPORT.md and excluded_images.jsonl.\n')


if __name__ == '__main__':
    main()
