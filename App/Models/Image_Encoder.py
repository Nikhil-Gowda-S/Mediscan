"""
App/Models/Image_Encoder.py
Wraps the DenseNet-121 backbone to extract spatial (1024-d) and
attention (256-d) feature vectors, and exposes a Grad-CAM interface.
"""
import torch
import torch.nn as nn

try:
    from captum.attr import LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

class ImageEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self._gradcam = None
        if CAPTUM_AVAILABLE:
            try:
                self.target_layer = backbone.features.denseblock4.denselayer16.conv2
            except AttributeError:
                self.target_layer = backbone.features[-1]
                
            try:
                self._gradcam = LayerGradCam(self, target_layer=self.target_layer)
            except Exception:
                self._gradcam = None

    def forward(self, x: torch.Tensor):
        spatial = self.backbone.features(x)
        features = torch.flatten(self.pool(spatial), 1)
        attn_feats = self.attention(features)
        return features, attn_feats

    def get_gradcam(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        if self._gradcam is None:
            raise RuntimeError("Captum not available")
        return self._gradcam.attribute(x, target_class)
