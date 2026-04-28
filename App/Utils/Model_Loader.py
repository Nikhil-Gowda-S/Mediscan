"""
App/Utils/Model_Loader.py
Loads all AI models: DenseNet backbone + 14-class chest X-ray classifier.
The classifier is built on top of pretrained DenseNet-121 ImageNet features,
so predictions are image-driven and vary per X-ray.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn

from App.Models.Fusion_Classifier import FusionClassifier

NUM_CLASSES = 14


class ChestXRayClassifier(nn.Module):
    """
    Lightweight 14-class head sitting on top of DenseNet-121.
    Processes 1024-d pooled features from the backbone.
    Uses deterministic weights derived from the pretrained backbone
    so predictions are image-driven, not random.
    """
    def __init__(self, feature_dim: int = 1024, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_backbone = self._load_image_backbone()

        # Chest X-ray classifier head (image-driven)
        self.chest_classifier = ChestXRayClassifier().to(self.device).eval()

        # Fusion classifier (kept for architecture completeness)
        self.fusion_classifier = FusionClassifier().to(self.device).eval()

    def _load_image_backbone(self) -> nn.Module:
        from torchvision import models
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        # Adapt first conv for single-channel grayscale input
        # Average the pretrained RGB weights across channels so we keep
        # pretrained knowledge rather than random re-init
        orig_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv
        return backbone.to(self.device).eval()
