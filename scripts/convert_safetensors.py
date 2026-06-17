"""Convert HuggingFace SafeTensors to Lightning .ckpt format."""
import os
import torch
from pathlib import Path
from safetensors.torch import load_file

os.environ["PYTHONIOENCODING"] = "utf-8"

WEIGHTS = Path(__file__).parent.parent / "weights"


def convert(hf_dir: str, output_name: str):
    """Load SafeTensors from HF dir and save as .ckpt."""
    safetensors_path = WEIGHTS / hf_dir / "model.safetensors"
    output_path = WEIGHTS / output_name

    if not safetensors_path.exists():
        print(f"Not found: {safetensors_path}")
        return False

    print(f"Loading: {safetensors_path}")
    state_dict = load_file(str(safetensors_path))
    print(f"  Keys: {len(state_dict)}")

    # Save as Lightning-compatible checkpoint
    checkpoint = {"state_dict": state_dict, "epoch": 50}
    torch.save(checkpoint, str(output_path))
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")
    return True


def main():
    # Convert HF downloads to standard ckpt names
    convert("hf_resnet3d_non_rd_vs_rd", "resnet3d_non_rd_vs_rd_best.ckpt")
    convert("hf_unet3d_macula", "unet3d_macula_detached_vs_intact_best.ckpt")


if __name__ == "__main__":
    main()
