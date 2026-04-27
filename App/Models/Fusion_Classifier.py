"""
App/Models/Fusion_Classifier.py
Cross-modal fusion for image (256-d) + vitals (64-d).
Outputs class logits for 14 chest-disease categories.
"""
import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    NUM_CLASSES = 14

    def __init__(self):
        super().__init__()
        self.vitals_encoder = nn.Linear(4, 64)

        self.fusion = nn.Sequential(
            nn.Linear(320, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CLASSES)
        )

        self.mc_dropout = nn.Dropout(0.2)

    def forward(
        self,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        vital_features: torch.Tensor,
    ) -> torch.Tensor:
        vit_encoded = self.vitals_encoder(vital_features)
        fused = torch.cat([image_features, vit_encoded], dim=1)
        out = self.fusion(fused)
        out = self.mc_dropout(out)
        return out
