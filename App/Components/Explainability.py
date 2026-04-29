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

        # Use full denseblock4 for rich 7x7 spatial activation maps
        # denselayer16.conv2 is 1x1 — produces poor heatmaps
        try:
            target_layer = model.features.denseblock4
        except AttributeError:
            target_layer = list(model.features.children())[-2]

        def forward_hook(module, input, output):
            activations.append(output.detach().clone())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach().clone())

        fh = target_layer.register_forward_hook(forward_hook)
        try:
            bh = target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            bh = target_layer.register_backward_hook(backward_hook)

        try:
            model.zero_grad()
            img = image_tensor.float().clone().detach().requires_grad_(True)
            
            # Must enable grad even if caller used no_grad context
            with torch.enable_grad():
                # Full forward pass through DenseNet manually
                features_out = model.features(img)
                pooled = F.adaptive_avg_pool2d(features_out, (1, 1))
                flat = torch.flatten(pooled, 1)
                logits = model.classifier(flat)

                # Backward on target class score
                score = logits[0, target_class % logits.shape[1]]
                score.backward()

            if not gradients or not activations:
                raise RuntimeError("Hooks did not capture gradients or activations")

            grad = gradients[0]       # [1, C, H, W]
            act = activations[0]      # [1, C, H, W]

            # Global average pool the gradients
            weights = grad.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
            
            # Weighted sum of activations
            cam = (weights * act).sum(dim=1).squeeze()  # [7, 7]
            # If squeeze removed too many dims, restore 2D
            if len(cam.shape) == 0:
                cam = cam.unsqueeze(0).unsqueeze(0)
            elif len(cam.shape) == 1:
                cam = cam.unsqueeze(0)
            cam_np = cam.detach().cpu().numpy()
            cam_np = np.maximum(cam_np, 0)  # ReLU in numpy — avoids tensor issues

            # Check if cam is all zeros
            if cam_np.max() == 0:
                # Fallback: use raw activation energy
                cam_np = act.squeeze().mean(dim=0).cpu().numpy()
                cam_np = np.maximum(cam_np, 0)

            # Normalize to 0-1
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max > cam_min:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.ones_like(cam_np) * 0.5

            # Get original image dimensions
            if isinstance(original_image, Image.Image):
                orig_w, orig_h = original_image.size
                orig_np = np.array(original_image.convert('RGB'))
            else:
                orig_h, orig_w = 224, 224
                orig_np = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

            if len(orig_np.shape) == 2:
                orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
            elif orig_np.shape[2] == 4:
                orig_np = orig_np[:, :, :3]

            orig_np = cv2.resize(orig_np, (orig_w, orig_h))

            # Resize CAM to match image
            cam_resized = cv2.resize(cam_np.astype(np.float32), (orig_w, orig_h))
            
            # Apply JET colormap
            cam_uint8 = np.uint8(255 * cam_resized)
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Blend with original
            orig_float = orig_np.astype(np.float32)
            heat_float = heatmap_rgb.astype(np.float32)
            blended = (0.55 * orig_float + 0.45 * heat_float).astype(np.uint8)

            return Image.fromarray(blended)

        except Exception as e:
            raise RuntimeError(f"Grad-CAM failed: {e}")
        finally:
            fh.remove()
            bh.remove()
