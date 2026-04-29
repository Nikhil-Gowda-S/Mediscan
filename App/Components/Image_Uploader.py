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
        if hasattr(image_file, "seek"):
            image_file.seek(0)

        original_pil = Image.open(image_file).convert("RGB")
        
        # Preserve aspect ratio then crop — critical for spatial feature integrity
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])
        
        tensor = transform(original_pil).unsqueeze(0)
        assert tensor.shape == torch.Size([1, 3, 224, 224]), \
            f"Unexpected tensor shape: {tensor.shape}. Expected [1, 3, 224, 224]"
        device = next(self.backbone.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            spatial = self.backbone.features(tensor)
            features = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(
                    spatial, (1,1)), 1)
            attn_features = features[:, :256]

        return features, attn_features, tensor, original_pil
