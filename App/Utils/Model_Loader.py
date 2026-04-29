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
        import os
        from torchvision import models

        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features  # 1024
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 14),
            nn.Sigmoid()
        )

        weights_path = "App/Models/chexnet_finetuned.pth"

        if os.path.exists(weights_path):
            checkpoint = torch.load(
                weights_path, map_location=self.device,
                weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            auc = checkpoint.get('auc', 'N/A')
            print(f"Loaded trained CheXNet weights. Best AUC: {auc}")
        else:
            print("No trained weights found. Using ImageNet pretrained.")
            model = models.densenet121(weights='IMAGENET1K_V1')
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 14),
                nn.Sigmoid()
            )

        # Store class prior weights to correct for dataset imbalance
        # NIH sample dataset counts — used to debias predictions
        # No Finding=3044, Infiltration=503, Effusion=203, Atelectasis=192
        # Nodule=144, Pneumothorax=114, Mass=99, Consolidation=72
        # Pleural_Thickening=65, Cardiomegaly=50, Emphysema=42
        # Edema=41, Fibrosis=38, Hernia=~10
        class_counts = torch.tensor([
            192.0,  # Atelectasis
            50.0,   # Cardiomegaly
            72.0,   # Consolidation
            41.0,   # Edema
            203.0,  # Effusion
            42.0,   # Emphysema
            38.0,   # Fibrosis
            10.0,   # Hernia
            503.0,  # Infiltration
            99.0,   # Mass
            3044.0, # No Finding
            144.0,  # Nodule
            65.0,   # Pleural Thickening
            114.0,  # Pneumothorax
        ])
        # Inverse frequency weights — rare classes get boosted
        self.class_weights = (1.0 / (class_counts + 1e-6))
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.class_weights = self.class_weights.to(self.device)

        return model.to(self.device).eval()

    def _build_chest_classifier(self) -> nn.Module:
        return self.image_backbone.classifier
