"""
App/Components/Image_Uploader.py
Handles X-ray upload, preprocessing, and feature extraction.
Returns both pooled features (1024-d) and attention features (256-d),
plus the preprocessed tensor and original PIL image.
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax'
]


class ImageUploader:
    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone

    def process(self, image_file):
        """
        Returns:
            features      – 1024-d pooled backbone features  [1, 1024]
            attn_features – 256-d projected features          [1, 256]
            tensor        – preprocessed 1-ch tensor           [1,1,224,224]
            original_pil  – original PIL Image (RGB)
        """
        if hasattr(image_file, "seek"):
            image_file.seek(0)

        original_pil = Image.open(image_file).convert("RGB")

        img_gray = original_pil.convert("L")
        tensor = _TRANSFORM(img_gray).unsqueeze(0)

        device = next(self.backbone.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            spatial = self.backbone.features(tensor)          # [1, 1024, 7, 7]
            features = torch.flatten(
                F.adaptive_avg_pool2d(spatial, (1, 1)), 1
            )                                                  # [1, 1024]
            # Proper learned projection — not a slice
            if not hasattr(self, '_proj'):
                self._proj = torch.nn.Linear(1024, 256, bias=False).to(device)
                torch.nn.init.orthogonal_(self._proj.weight)
            with torch.no_grad():
                attn_features = self._proj(features)

        return features, attn_features, tensor, original_pil
