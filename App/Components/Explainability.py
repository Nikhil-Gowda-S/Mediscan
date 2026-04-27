"""
App/Components/Explainability.py
Generates Grad-CAM heatmap overlays on X-ray images.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

class GradCAMOverlay:
    def create_overlay(self, image_tensor, target_class, original_image, backbone):
        import torch
        from App.Models.Image_Encoder import ImageEncoder
        
        device = image_tensor.device
        encoder = ImageEncoder(backbone).to(device)
        encoder.eval()
        
        attributions = encoder.get_gradcam(image_tensor, target_class)
        
        if isinstance(original_image, np.ndarray):
            original = original_image
            if original.ndim == 2:
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        else:
            if hasattr(original_image, "seek"):
                original_image.seek(0)
            original = np.array(Image.open(original_image).convert("RGB"))

        H, W = original.shape[:2]
#snjs
        attr = attributions.detach().cpu()
        if attr.ndim == 4:
            attr = attr[0].mean(0)
        elif attr.ndim == 3:
            attr = attr.mean(0)
        attr = attr.numpy().astype(np.float32)
        attr = np.maximum(attr, 0)
        if attr.max() > 0:
            attr = attr / attr.max()

        heatmap_u8 = np.uint8(255 * cv2.resize(attr, (W, H)))
        heatmap_rgb = cv2.cvtColor(
            cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )

        alpha = 0.55
        overlay = cv2.addWeighted(original, alpha, heatmap_rgb, 1 - alpha, 0)
        return overlay
