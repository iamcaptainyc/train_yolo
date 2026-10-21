"""Train, evaluate or predict with Ultralytics YOLO11 / YOLO26."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'val', 'predict'], default='train')
    p.add_argument('--model', default='yolo11s.pt', help='yolo11{n,s,m,l,x}.pt / yolo26{n,s,m,l,x}.pt or checkpoint path')
    p.add_argument('--data', type=Path, default=Path('datasets/animals_yolo/data.yaml'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default=None, help='0, 0,1, cpu; omitted = Ultralytics auto selection')
    p.add_argument('--workers', type=int, default=0, help='Windows-safe default; increase if loading is slow')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--project', default='runs/animals')
    p.add_argument('--name', default=None)
    p.add_argument('--resume', type=Path, help='Resume an interrupted run from weights/last.pt; keeps saved run settings')
    p.add_argument('--split', choices=['val', 'test'], default='test', help='Evaluation split in val mode')
    p.add_argument('--source', help='Image, directory, video or stream in predict mode')
    p.add_argument('--conf', type=float, default=0.25, help='Prediction confidence only; validation uses Ultralytics default')
    p.add_argument('--cache', choices=['false', 'ram', 'disk'], default='false')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.resume and args.mode != 'train':
        p.error('--resume is available only in train mode')
    if args.resume and not args.resume.is_file():
        p.error(f'Checkpoint does not exist: {args.resume}')
    if args.mode in ('train', 'val') and not args.resume and not args.data.is_file():
        p.error(f'Dataset YAML does not exist: {args.data}; run prepare_dataset.py first')
    if args.mode == 'predict' and not args.source:
        p.error('--source is required in predict mode')
    from ultralytics import YOLO
    import torch
    print(f'PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}')
    model = YOLO(str(args.resume) if args.resume else args.model, task='detect')
    device = {'device': args.device} if args.device is not None else {}
    if args.resume:
        model.train(resume=True, **device)
    elif args.mode == 'train':
        result = model.train(
            data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, workers=args.workers, seed=args.seed, deterministic=True,
            patience=args.patience, project=args.project,
            name=args.name or Path(args.model).stem, amp=args.amp,
            cache=False if args.cache == 'false' else args.cache,
            val=True, plots=True, **device,
        )
        print(f'Training output: {result.save_dir}')
        print('Evaluate weights/best.pt on the held-out test split with --mode val.')
    elif args.mode == 'val':
        metrics = model.val(data=str(args.data.resolve()), split=args.split,
                            imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                            project=args.project, name=args.name or f'eval_{args.split}',
                            plots=True, **device)
        print(metrics.results_dict)
        print(f'Evaluation output: {metrics.save_dir}')
    else:
        for result in model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                                    project=args.project, name=args.name or 'predict',
                                    save=True, save_txt=True, save_conf=True, stream=True, **device):
            print(f'{result.path}: {len(result.boxes)} detections')


if __name__ == '__main__':
    # Required for Windows multiprocessing data loaders.
    from training_guard import entrypoint
    raise SystemExit(entrypoint(main, __file__))
