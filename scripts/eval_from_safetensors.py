"""
Evaluate a SafeTensors checkpoint on a test CSV.
Bypasses Lightning checkpoint loading — loads weights directly.
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from erdes.models.components.factory import build_3d_architecture
from erdes.data.components.erdes_dataset import VideoDataset


def compute_metrics(preds, targets):
    """Compute all binary classification metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    tp = ((preds == 1) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()

    acc = (tp + tn) / len(targets) if len(targets) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    return {"acc": acc, "sens": sens, "spec": spec, "prec": prec, "f1": f1}


def evaluate(safetensors_path, model_name, test_csv, device="cuda", batch_size=4):
    """Load model from SafeTensors and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {safetensors_path}")
    print(f"Test CSV: {test_csv}")
    print(f"{'='*60}")

    # Load model architecture
    print(f"Building model: {model_name}")
    model = build_3d_architecture(model_name, num_classes=1)
    model = model.to(device)
    model.eval()

    # Load weights from SafeTensors
    print("Loading weights from SafeTensors...")
    state_dict = load_file(str(safetensors_path))
    # Auto-detect prefix: check if model expects 'model.' or 'enc.' prefix
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys:
        if any(k.startswith("model.") for k in model_keys) and not any(k.startswith("model.") for k in ckpt_keys):
            state_dict = {"model." + k: v for k, v in state_dict.items()}
        elif any(k.startswith("enc.") for k in model_keys) and not any(k.startswith("enc.") for k in ckpt_keys):
            state_dict = {"enc." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded {len(state_dict)} parameter tensors")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create dataset and dataloader
    size = (96, 128, 128)
    dataset = VideoDataset(
        csv_path=str(test_csv),
        size=size,
        data_root="data/erdes",
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"  Test samples: {len(dataset)}")

    # Run inference
    all_preds = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Batches"):
            videos = videos.to(device)
            logits = model(videos)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().view(-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.int().tolist())

    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets)

    print("\nResults:")
    print(f"  Accuracy:    {metrics['acc']:.4f}")
    print(f"  Sensitivity: {metrics['sens']:.4f}")
    print(f"  Specificity: {metrics['spec']:.4f}")
    print(f"  Precision:   {metrics['prec']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    return evaluate(
        safetensors_path=args.safetensors,
        model_name=args.model,
        test_csv=args.test_csv,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
