"""Shared preprocessing and models. Codex-generated; see ATTRIBUTION.md."""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from .generate import ROOT, BOX


def records(split):
    return [
        json.loads(s)
        for s in (ROOT / "manifest.jsonl").read_text().splitlines()
        if json.loads(s)["split"] == split
    ]


def preprocess(image, crop=False, gray=False, size=64, normalize_background=False):
    """Fixed known chart-box crop, NOT automatic plot detection on arbitrary images."""
    im = image.convert("RGB")
    if crop:
        im = im.crop(
            (
                round(im.width * BOX[0]),
                round(im.height * BOX[1]),
                round(im.width * BOX[2]),
                round(im.height * BOX[3]),
            )
        )
    if gray:
        im = im.convert("L").convert("RGB")
    if normalize_background:
        a = np.asarray(im)
        if np.median(a) < 128:
            im = Image.fromarray(255 - a)
    return (
        torch.from_numpy(
            np.asarray(im.resize((size, size), Image.Resampling.BILINEAR)).copy()
        )
        .permute(2, 0, 1)
        .float()
        / 255
    )


def load_split(split, crop=False, gray=False, size=64):
    rows = records(split)
    x = torch.stack(
        [preprocess(Image.open(ROOT / r["path"]), crop, gray, size) for r in rows]
    )
    y = torch.tensor([r["label"] if r["label"] is not None else -1 for r in rows])
    return x, y, rows


class SmallCNN(nn.Module):
    """Keep spatial 4x4 features: wing location distinguishes directional shapes."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.head(self.features(x))


class RobustCNN(nn.Module):
    """Higher-resolution shape reader, preserving spatial position of the wings."""
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 16, 32, 64]
        for a, b in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((6, 6)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 36, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 8))

    def forward(self, x):
        return self.head(self.features(x))


def device():
    if os.environ.get("OPTIONS_DEVICE"):
        return torch.device(os.environ["OPTIONS_DEVICE"])
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def augment(x):
    """Per-image brightness and contrast only: no flips, rotations or crops."""
    b = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.85, 1.15)
    c = torch.empty_like(b).uniform_(0.8, 1.2)
    means = x.mean((2, 3), keepdim=True)
    return ((x - means) * c + means).mul(b).clamp(0, 1)


def backbone(pretrained=False):
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    net = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    net.classifier = nn.Identity()
    return net.eval()


def imagenet_input(x):
    return (
        x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
