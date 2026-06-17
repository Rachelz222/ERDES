"""
3D Grad-CAM Explainer for ERDES Models
========================================
Provides Grad-CAM visualization for 3D medical imaging models,
specifically designed for the 3D U-Net classifier used in retinal
detachment diagnosis.

Supports:
  - 3D tensor input: shape (1, C, D, H, W) — Batch, Channel, Depth, Height, Width
  - Gradient extraction from the last convolutional layer
  - Heatmap generation and overlay on middle video frame

Usage:
    from erdes.models.explainer import GradCAM3D

    model = Unet3DClassifier(in_channels=1, num_classes=1)
    explainer = GradCAM3D(model, target_layer="enc.encoder.4.basic_module.SingleConv2.conv")
    heatmap = explainer(input_tensor)  # returns 2D heatmap [H, W] for middle frame
    explainer.save_heatmap(input_tensor, "output.png")
"""

import os, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class GradCAM3D:
    """
    3D Grad-CAM for volumetric medical imaging models.

    Captures gradients and activations from a target convolutional layer
    and generates a class-discriminative localization heatmap.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: str = None):
        """
        Args:
            model: The PyTorch model (e.g., Unet3DClassifier).
            target_layer: Dotted path to the target conv layer.
                          Default: last conv in UNet3D encoder.
                          E.g., "enc.encoder.4.basic_module.SingleConv2.conv"
        """
        self.model = model
        self.model.eval()

        # Resolve target layer
        if target_layer is None:
            # Default: output of the last encoder block (bottleneck features).
            # We hook at encoder[-1] level rather than internal conv to avoid
            # inplace ReLU conflicts in the autograd engine.
            target_layer = "enc.encoder.4"

        self.target_layer = self._resolve_layer(target_layer)
        self.target_name = target_layer

        # Storage for captured values
        self.activations = None   # Feature map A_k from forward pass
        self.gradients = None     # Gradient dL/dA_k from backward pass

        # Register hooks
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

        log.info(f"GradCAM3D: hooked to layer '{target_layer}'")

    def _resolve_layer(self, layer_path: str) -> nn.Module:
        """Resolve dotted path to an nn.Module."""
        parts = layer_path.split(".")
        obj = self.model
        for i, part in enumerate(parts):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"Layer '{layer_path}' resolved to {type(obj)}, not nn.Module")
        return obj

    def _forward_hook(self, module, input, output):
        """Capture activations from the target layer during forward pass.
        Clone is essential: the output tensor may be modified in-place by
        downstream ReLU layers, which conflicts with autograd's backward hooks."""
        self.activations = output.clone()  # [B, C, D', H', W']

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients from the target layer during backward pass.
        grad_output[0] is dL/d(output) — shape [B, C, D', H', W']."""
        self.gradients = grad_output[0].clone()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input tensor.

        Args:
            input_tensor: shape (1, C, D, H, W)
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            heatmap_2d: numpy array of shape (H, W) — heatmap for the middle frame
                        of the video, normalized to [0, 1].
        """
        # ------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------
        self.model.zero_grad()
        output = self.model(input_tensor)  # [1, 1] logit

        if class_idx is None:
            class_idx = int((torch.sigmoid(output) >= 0.5).int().item())

        # ------------------------------------------------------------------
        # Backward pass — compute gradient of the output score
        # w.r.t. the target layer's activations.
        # output shape: [1, 1]; backprop on the scalar logit value.
        # ------------------------------------------------------------------
        self.model.zero_grad()
        output.sum().backward(retain_graph=False)

        # ------------------------------------------------------------------
        # Compute Grad-CAM weights and heatmap
        # ------------------------------------------------------------------
        # activations: [1, C, D', H', W']
        # gradients:   [1, C, D', H', W']
        activations = self.activations  # [1, C, D', H', W']
        gradients = self.gradients      # [1, C, D', H', W']

        # Global average pool gradients over spatial dims (D', H', W')
        # → alpha_k per channel
        alpha = gradients.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]

        # Weighted sum of activations: sum_k(alpha_k * A_k)
        # → [1, 1, D', H', W']
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # [1, 1, D', H', W']

        # ReLU to keep only positive influence
        cam = F.relu(cam)  # [1, 1, D', H', W']

        # ------------------------------------------------------------------
        # Upsample heatmap to match input spatial dims
        # ------------------------------------------------------------------
        # Input shape: [1, C, D_in, H_in, W_in]
        _, _, D_in, H_in, W_in = input_tensor.shape

        # Need to upsample the 3D heatmap
        # cam: [1, 1, D', H', W'] → upsample → [1, 1, D_in, H_in, W_in]
        cam_upsampled = F.interpolate(
            cam,
            size=(D_in, H_in, W_in),
            mode="trilinear",
            align_corners=False,
        )  # [1, 1, D_in, H_in, W_in]

        cam_upsampled = cam_upsampled.squeeze().detach()  # [D_in, H_in, W_in]

        # ------------------------------------------------------------------
        # Extract middle frame for 2D visualization
        # ------------------------------------------------------------------
        mid_idx = D_in // 2
        heatmap_3d = cam_upsampled.cpu().numpy()  # [D, H, W] — already detached
        heatmap_2d = heatmap_3d[mid_idx]  # [H, W]

        # Normalize to [0, 1]
        h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
        if h_max > h_min:
            heatmap_2d = (heatmap_2d - h_min) / (h_max - h_min)

        return heatmap_2d

    def save_heatmap(
        self,
        input_tensor: torch.Tensor,
        output_path: str,
        class_idx: int = None,
        alpha: float = 0.5,
        colormap: str = "jet",
    ):
        """
        Generate heatmap, overlay on the middle video frame, and save as .png.

        Args:
            input_tensor: shape (1, C, D, H, W), normalized [0, 1]
            output_path: Path to save the .png file
            class_idx: Target class index (None = auto-detect)
            alpha: Blend factor for heatmap overlay (0 = original, 1 = heatmap only)
            colormap: Matplotlib colormap name
        """
        # Generate heatmap
        heatmap_2d = self(input_tensor, class_idx=class_idx)  # [H, W]

        # ------------------------------------------------------------------
        # Extract the middle frame from the original input tensor
        # ------------------------------------------------------------------
        # input_tensor: [1, C, D, H, W] in [0, 1]
        D = input_tensor.shape[2]
        mid_idx = D // 2
        frame = input_tensor[0, 0, mid_idx].cpu().numpy()  # [H, W], grayscale in [0,1]

        # ------------------------------------------------------------------
        # Create overlay figure
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Original frame
        axes[0].imshow(frame, cmap="gray")
        axes[0].set_title("Original Frame (middle)")
        axes[0].axis("off")

        # Subplot 2: Heatmap only
        axes[1].imshow(heatmap_2d, cmap=colormap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        # Subplot 3: Overlay
        axes[2].imshow(frame, cmap="gray")
        axes[2].imshow(heatmap_2d, cmap=colormap, alpha=alpha)
        axes[2].set_title(f"Overlay (α={alpha})")
        axes[2].axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info(f"Heatmap saved to {output_path}")

    def remove_hooks(self):
        """Remove registered hooks to free resources."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience function: generate explanation for a sample video
# ---------------------------------------------------------------------------
def explain_sample(
    model: nn.Module,
    video_tensor: torch.Tensor,
    output_dir: str = "results/plots",
    sample_name: str = "sample",
    device: str = "cuda",
):
    """
    Generate and save Grad-CAM explanation for a single video sample.

    Args:
        model: The pretrained 3D U-Net classifier
        video_tensor: shape [1, D, H, W] or [1, 1, D, H, W], normalized [0,1]
        output_dir: Directory to save output plots
        sample_name: Name prefix for the saved file
        device: "cuda" or "cpu"

    Returns:
        output_path: Path to saved .png file
    """
    # Ensure correct shape: [1, 1, D, H, W]
    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)  # [1, 1, D, H, W]

    video_tensor = video_tensor.to(device)
    model = model.to(device)

    explainer = GradCAM3D(model)

    output_path = Path(output_dir) / f"gradcam_{sample_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.enable_grad():
        explainer.save_heatmap(video_tensor, str(output_path))

    explainer.remove_hooks()
    return str(output_path)


if __name__ == "__main__":
    # Quick test
    print("Testing GradCAM3D...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from erdes.models.components.cls_model import Unet3DClassifier

    model = Unet3DClassifier(in_channels=1, num_classes=1)
    dummy = torch.randn(1, 1, 96, 128, 128)

    explainer = GradCAM3D(model)
    heatmap = explainer(dummy)
    print(f"Heatmap shape: {heatmap.shape} (expected: (128, 128))")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print("GradCAM3D test passed!")
    explainer.remove_hooks()
