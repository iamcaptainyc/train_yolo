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
    p.add_argument('--eval-batch', type=int, default=32, help='Post-training train/val/test evaluation batch size')
    p.add_argument('--device', default=None, help='0, 0,1, cpu; omitted = Ultralytics auto selection')
    p.add_argument('--workers', type=int, default=0, help='Windows-safe default; increase if loading is slow')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--project', default='runs/animals')
    p.add_argument('--name', default=None)
    p.add_argument('--resume', type=Path, help='Resume an interrupted run from weights/last.pt; keeps saved run settings')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Evaluation split in val mode')
    p.add_argument('--source', help='Image, directory, video or stream in predict mode')
    p.add_argument('--conf', type=float, default=0.25, help='Prediction confidence only; validation uses Ultralytics default')
    p.add_argument('--cache', choices=['false', 'ram', 'disk'], default='false')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fliplr', type=float, default=None,
                   help='Training horizontal flip probability; ignored in val/predict (new training default: 0.5; resume: saved value)')
    args = p.parse_args()
    if args.mode == 'train' and args.fliplr is not None:
        if not 0 <= args.fliplr <= 1:
            p.error('--fliplr must be between 0 and 1')
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
    from training_diagnostics import install_phase_logging
    install_phase_logging(model)
    device = {'device': args.device} if args.device is not None else {}
    if args.resume:
        saved_fliplr = (model.ckpt or {}).get('train_args', {}).get('fliplr', 0.5)
        if args.fliplr is not None and args.fliplr != saved_fliplr:
            p.error('--fliplr differs from the checkpoint. Resume keeps saved settings; '
                    'use --model CHECKPOINT with a new --name to start training with a different probability')
        result = model.train(resume=True, **device)
    elif args.mode == 'train':
        result = model.train(
            data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, workers=args.workers, seed=args.seed, deterministic=True,
            patience=args.patience, project=args.project,
            name=args.name or Path(args.model).stem, amp=args.amp,
            cache=False if args.cache == 'false' else args.cache,
            val=True, plots=True, fliplr=0.5 if args.fliplr is None else args.fliplr, **device,
        )
        print(f'Training output: {result.save_dir}')
    elif args.mode == 'val':
        from evaluation_artifacts import evaluate_logged
        # Let Ultralytics choose a unique evaluation directory, then place artifacts there.
        from ultralytics.cfg import get_cfg, get_save_dir
        directory = get_save_dir(get_cfg(overrides={'project':args.project,
                                 'name':args.name or f'eval_{args.split}', 'mode':'val', 'task':'detect'}))
        metrics = evaluate_logged(model, directory, args.split, args.model, data=str(args.data.resolve()),
                            imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                            project=str(directory.parent), name=directory.name, exist_ok=True, **device)
        print(metrics['metrics'])
        print(f'Evaluation output: {directory}')
    else:
        for result in model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                                    project=args.project, name=args.name or 'predict',
                                    save=True, save_txt=True, save_conf=True, stream=True, **device):
            print(f'{result.path}: {len(result.boxes)} detections')
    if args.mode == 'train':
        import yaml
        from evaluation_artifacts import evaluate_logged
        trainer = model.trainer
        directory = Path(trainer.save_dir)
        checkpoint = Path(trainer.best)
        evaluated = YOLO(str(checkpoint), task='detect')
        data = trainer.args.data
        config = yaml.safe_load(Path(data).read_text(encoding='utf-8-sig'))
        for split in ('val', 'test', 'train'):
            if not config.get(split):
                from search_worker import write_json
                write_json(directory/f'{split}_result.json', {'status':'skipped', 'split':split,
                                                             'reason':'Split is absent from dataset YAML'})
                continue
            evaluate_logged(evaluated, directory, split, checkpoint, data=str(data),
                imgsz=trainer.args.imgsz, batch=args.eval_batch, workers=args.workers,
                project=str(directory/'evaluation'), name=split, exist_ok=True, **device)
        print(f'Metrics, logs and confusion-matrix JSON: {directory}')


if __name__ == '__main__':
    # Required for Windows multiprocessing data loaders.
    from training_guard import entrypoint
    raise SystemExit(entrypoint(main, __file__))
