"""PyTorch inference for the official CFV ConvNeXt continuous-angle checkpoint."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re


CFV_URL = 'https://drive.usercontent.google.com/download?id=1l4hVCMvJFhIYEqHh5ij10V2f1cQyJhyf&export=download&confirm=t'
CFV_SIZE = 201_141_385
CFV_SHA256 = '26bd9e4eff8a4cfa3a264223ecced760f2fb3d1f29dc68f4e23b1b27babcd91d'
VEHICLE_CLASSES = ('truck', 'bus', 'car_head', 'car_side', 'car_tail')


def angle_class(degrees):
    if not math.isfinite(degrees):
        raise ValueError('Vehicle angle is not finite')
    angle = abs((degrees + 180) % 360 - 180)
    if angle <= 45:
        return 'car_head'
    if angle >= 135:
        return 'car_tail'
    return 'car_side'


def checkpoint_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch_cfv_weights(path, retries=5):
    from download_openimages import download

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    download(CFV_URL, path, retries, progress=True)
    if path.stat().st_size != CFV_SIZE or checkpoint_digest(path) != CFV_SHA256:
        raise ValueError(f'CFV checkpoint size or SHA-256 mismatch: {path}')
    return path


def convert_cfv_state(state):
    """Map the published timm ConvNeXt keys to the equivalent torchvision layers."""
    result = {}
    for key, tensor in state.items():
        key = key.removeprefix('module.')
        target = key
        if key.startswith('backbone.stem.'):
            target = key.replace('backbone.stem.', 'backbone.features.0.', 1)
        elif match := re.fullmatch(r'backbone.stages.([0-3]).downsample.([01]).(weight|bias)', key):
            stage, layer, parameter = match.groups()
            target = f'backbone.features.{2*int(stage)}.{layer}.{parameter}'
        elif match := re.fullmatch(r'backbone.stages.([0-3]).blocks.(\d+).(.+)', key):
            stage, block, suffix = match.groups()
            base = f'backbone.features.{2*int(stage)+1}.{block}.'
            if suffix == 'gamma':
                target = base + 'layer_scale'
                tensor = tensor.reshape(-1, 1, 1)
            else:
                replacements = {'conv_dw': 'block.0', 'norm': 'block.2',
                                'mlp.fc1': 'block.3', 'mlp.fc2': 'block.5'}
                layer, parameter = suffix.rsplit('.', 1)
                if layer not in replacements:
                    raise ValueError(f'Unsupported CFV checkpoint layer: {key}')
                target = base + replacements[layer] + '.' + parameter
        elif key.startswith('backbone.head.norm.'):
            target = key.replace('backbone.head.norm.', 'backbone.classifier.0.', 1)
        elif key.startswith('backbone.head.fc.'):
            target = key.replace('backbone.head.fc.', 'backbone.classifier.2.', 1)
        if target in result:
            raise ValueError(f'Duplicate CFV checkpoint parameter: {target}')
        result[target] = tensor
    return result


class CFVPredictor:
    def __init__(self, weights, device='cpu', batch=32):
        import torch
        from torch import nn
        from torchvision.models import convnext_small

        if batch < 1:
            raise ValueError('Pose batch size must be positive')
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise ValueError(f'CFV weights not found: {weights}')
        if device.isdecimal():
            device = f'cuda:{device}'
        self.device = torch.device(device)
        self.batch = batch
        self.digest = checkpoint_digest(weights)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = convnext_small(weights=None)
                self.backbone.classifier[2] = nn.Linear(768, 1024)
                self.class_prediction = nn.Linear(1024, 8)
                self.angle_prediction = nn.Linear(1032, 1)

            def forward(self, images):
                features = self.backbone(images)
                classes = self.class_prediction(features)
                return self.angle_prediction(torch.cat((features, classes), dim=1)).sigmoid().flatten()*360

        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('Expected the official CFV state dictionary')
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        self.model = Model()
        self.model.load_state_dict(convert_cfv_state(state), strict=True)
        self.model.to(self.device).eval()

    def predict(self, crops):
        import numpy as np
        import torch
        import cv2

        results = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch):
                tensors = []
                for crop in crops[start:start+self.batch]:
                    pixels = cv2.resize(np.asarray(crop.convert('RGB')), (224, 224), interpolation=cv2.INTER_LINEAR)
                    tensors.append(torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255))
                inputs = torch.stack(tensors).to(self.device)
                angles = self.model(inputs).cpu().tolist()
                if any(not math.isfinite(angle) or not 0 <= angle <= 360 for angle in angles):
                    raise ValueError('CFV returned invalid continuous angles')
                results.extend(angles)
        return results
