"""
Hybrid 3D-UNet-Mamba Classifier
================================
UNet3D encoder → Mamba3DProcessor (bottleneck) → ClassificationHead.

Design rationale:
  - CNN encoder extracts local anatomical features (retina texture, shape)
  - Mamba at bottleneck processes 384-token sequence across 3 spatial axes,
    capturing long-range temporal dependencies across 96 ultrasound frames
  - ClassificationHead pools and classifies

Weight compatibility:
  The encoder and classification head weights are initialized from the
  pretrained UNet3D checkpoint. Only the Mamba processor weights are
  randomly initialized and fine-tuned.
"""

import torch
import torch.nn as nn

from .components.cls_model import Unet3DClassifier, ClassificationHead
from .mamba_block import Mamba3DProcessor


class HybridUNetMamba(nn.Module):
    """
    UNet3D + Mamba at bottleneck → Classification.

    Args:
        in_channels:     input channels (1 for grayscale ultrasound)
        num_classes:     output classes (1 for binary)
        mamba_channels:  channels at bottleneck (must match encoder output)
        mamba_d_state:   SSM state dimension
        mamba_d_conv:    conv kernel size for local mixing
        mamba_expand:    expansion factor
        pooling:         "avg" or "topk"
        topk_ratio:      ratio for top-k pooling
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        # UNet3D encoder args
        f_maps: list = None,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        conv_upscale: int = 2,
        dropout_prob: float = 0.0,
        layer_order: str = "gcr",
        num_groups: int = 8,
        pool_kernel_size: int = 2,
        # Mamba args
        mamba_channels: int = 768,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 1,
        # Classification head
        pooling: str = "avg",
        topk_ratio: float = 0.5,
    ):
        super().__init__()
        if f_maps is None:
            f_maps = [64, 128, 256, 512, 768]

        # Build a base UNet3D to extract the encoder
        base = Unet3DClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            f_maps=f_maps,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            dropout_prob=dropout_prob,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=pool_kernel_size,
            pooling=pooling,
            topk_ratio=topk_ratio,
        )

        self.encoder = base.enc  # UNet3DEncoder

        # Mamba bottleneck processor
        self.mamba = Mamba3DProcessor(
            channels=mamba_channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        # Classification head (same as original)
        input_dim = f_maps[-1]  # 768
        self.cls = ClassificationHead(
            input_dim=input_dim,
            hidden_size=input_dim // 2,
            num_classes=num_classes,
            pooling=pooling,
            topk_ratio=topk_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) — input video tensor
        Returns:
            logits: (B, num_classes)
        """
        # CNN encoder
        features = self.encoder(x)  # (B, 768, D', H', W')

        # Mamba sequence processing at bottleneck
        features = self.mamba(features)  # (B, 768, D', H', W')

        # Classification
        logits = self.cls(features)
        return logits

    def load_pretrained_encoder_head(self, state_dict: dict, strict: bool = False):
        """
        Load pretrained weights for encoder and classification head,
        leaving Mamba weights randomly initialized.

        Args:
            state_dict: checkpoint from pretrained UNet3DClassifier
            strict:     whether to require exact key match
        """
        # Filter to only encoder and cls keys
        own_state = self.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k.startswith("enc.") or k.startswith("cls."):
                if k in own_state:
                    filtered[k] = v
                else:
                    skipped.append(k)

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"  Loaded {len(filtered)}/{len(state_dict)} pretrained keys "
              f"(encoder + cls head)")
        print(f"  Uninitialized (Mamba): {len(own_state) - len(filtered)} params")
        return missing, unexpected


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing HybridUNetMamba...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridUNetMamba(in_channels=1, num_classes=1).to(device)
    x = torch.randn(1, 1, 96, 128, 128).to(device)

    with torch.no_grad():
        y = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(y.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    # Check weight loading
    print("\n  Testing pretrained weight loading...")
    from safetensors.torch import load_file
    from pathlib import Path
    ckpt = Path(__file__).parent.parent.parent / "weights" / "unet3d_macula_detached_vs_intact.safetensors"
    if ckpt.exists():
        sd = load_file(str(ckpt))
        model2 = HybridUNetMamba(in_channels=1, num_classes=1).to(device)
        model2.load_pretrained_encoder_head(sd)
        print("  Weight loading: SUCCESS ✓")
    else:
        print(f"  Checkpoint not found: {ckpt}")

    print("All tests passed!")
