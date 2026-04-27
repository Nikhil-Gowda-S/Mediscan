"""
App/Components/Image_Uploader.py
Handles X-ray upload, preprocessing, and feature extraction.
"""
from __future__ import annotations

import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from App.Models.Image_Encoder import ImageEncoder

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumothorax'
]

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

class ImageUploader:
    def __init__(self, backbone: torch.nn.Module):
        self.model = ImageEncoder(backbone)

    def process(self, image_file):
        if hasattr(image_file, "seek"):
            image_file.seek(0)
            
        img = Image.open(image_file).convert("RGB")
        original_image = np.array(img)
        
        img_gray = img.convert("L")
        tensor = _TRANSFORM(img_gray).unsqueeze(0)

        device = next(self.model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            features, attn_features = self.model(tensor)

        return features, attn_features, tensor, original_image
