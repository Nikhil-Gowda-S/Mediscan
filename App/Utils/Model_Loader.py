"""
App/Utils/Model_Loader.py
Loads all AI models: DenseNet backbone, FusionClassifier.
All heavy imports are deferred so the module can be imported without GPU dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn

from App.Models.Image_Encoder import ImageEncoder
from App.Models.Fusion_Classifier import FusionClassifier

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.image_backbone = self._load_image_backbone()
        except Exception as e:
            raise RuntimeError(f"Failed to load image backbone: {e}")

        try:
            self.fusion_classifier = FusionClassifier().to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load fusion classifier: {e}")

    def _load_image_backbone(self) -> nn.Module:
        from torchvision import models
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        # Adapt first conv for single-channel grayscale input
        backbone.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        return backbone.to(self.device).eval()
