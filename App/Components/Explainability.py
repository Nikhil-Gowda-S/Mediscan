import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAMOverlay:
    def create_overlay(self, image_tensor, target_class, original_image, model):
        model.eval()
        gradients = []
        activations = []
        target_layer = model.features.denseblock4

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        try:
            img = image_tensor.clone().requires_grad_(True)
            features = model.features(img)
            out = F.adaptive_avg_pool2d(features, (1, 1))
            out = torch.flatten(out, 1)
            logits = model.classifier(out)
            model.zero_grad()
            score = logits[0, target_class % logits.shape[1]]
            score.backward(retain_graph=True)

            grad = gradients[0]
            act = activations[0]
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() + 1e-8)

            if isinstance(original_image, Image.Image):
                w, h = original_image.size
                orig_np = np.array(original_image.convert('RGB'))
            else:
                h, w = 224, 224
                orig_np = np.zeros((h, w, 3), dtype=np.uint8)

            if len(orig_np.shape) == 2:
                orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

            orig_np = cv2.resize(orig_np, (w, h))
            cam_resized = cv2.resize(cam, (w, h))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_np, 0.55, heatmap, 0.45, 0)
            return Image.fromarray(overlay)

        except Exception as e:
            raise RuntimeError(f"Grad-CAM failed: {e}")
        finally:
            fh.remove()
            bh.remove()
