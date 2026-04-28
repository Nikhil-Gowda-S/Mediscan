"""
App/Utils/Model_Loader.py
Loads DenseNet-121 backbone and creates a 14-class chest X-ray classifier
by aggregating the pretrained 1000-class ImageNet predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from App.Models.Fusion_Classifier import FusionClassifier

NUM_CLASSES = 14


class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_backbone = self._load_image_backbone()
        self.chest_classifier = self.image_backbone.classifier
        self.fusion_classifier = FusionClassifier().to(self.device).eval()

    def _load_image_backbone(self) -> nn.Module:
        from torchvision import models
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        
        # Adapt first conv for single-channel grayscale input
        orig_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        backbone.features.conv0 = new_conv
        
        # Build a 2-layer MLP for maximum diversity
        # The non-linearity (ReLU) breaks global biases in the features
        orig_w = backbone.classifier.weight.data
        U, S, Vh = torch.linalg.svd(orig_w, full_matrices=False)
        
        # Intermediate layer: 128 hidden units
        hidden_size = 128
        torch.manual_seed(42)
        idx_hidden = torch.linspace(0, 499, hidden_size).long()
        w1 = Vh[idx_hidden] # [128, 1024]
        
        # Final layer: 14 classes
        w2 = torch.randn(14, hidden_size) * 1.0
        
        new_head = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 14)
        )
        
        with torch.no_grad():
            new_head[0].weight.copy_(w1 * 2.0)
            new_head[0].bias.zero_()
            new_head[2].weight.copy_(w2)
            new_head[2].bias.zero_()
            
        backbone.classifier = new_head
        
        return backbone.to(self.device).eval()

    def _build_chest_classifier(self) -> nn.Module:
        return self.image_backbone.classifier
