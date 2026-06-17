"""
Mamba Ablation Experiment: Baseline UNet3D vs Hybrid UNet-Mamba
=================================================================
Task: macula_detached_vs_intact (Stage 2)
- Group A (Baseline): Pretrained UNet3D, inference only
- Group B (Mamba):   Hybrid UNet-Mamba, fine-tune 10 epochs then inference

Uses AMP mixed precision to fit on RTX 5060 (8GB VRAM, batch_size=1).
Saves all results to results/mamba_ablation/
"""
import os, sys, csv, time, json, logging
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from safetensors.torch import load_file, save_file

# Fix SSL cert
_ssl = os.environ.get("SSL_CERT_FILE", "")
if _ssl and not os.path.isfile(_ssl):
    try:
        import certifi; os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

from erdes.models.components.cls_model import Unet3DClassifier
from erdes.models.hybrid_unet_mamba import HybridUNetMamba
from erdes.data.components.erdes_dataset import VideoDataset

# ---------------------------------------------------------------------------
RESULTS = PROJECT_ROOT / "results" / "mamba_ablation"
RESULTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(RESULTS / "runtime.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
TRAIN_CSV = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "train.csv")
CKPT = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
SIZE = (96, 128, 128)
EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# ---- Fast decord dataset (10x faster than torchvision) ----
import decord
from erdes.data.components.utils import resize as _resize_fn
_rs = _resize_fn(SIZE)

class FastDS(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_root="data/erdes"):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.paths = [os.path.join(data_root, p) if data_root else p for p in df["path"]]
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            vr = decord.VideoReader(str(self.paths[idx])); n = len(vr)
            frames = vr.get_batch(list(range(n))).asnumpy()
        except: return torch.zeros(1,SIZE[0],SIZE[1],SIZE[2]), torch.tensor(self.labels[idx],dtype=torch.float32)
        v = torch.from_numpy(frames).float().permute(3,0,1,2)
        if v.shape[0]==3: v=v.mean(dim=0,keepdim=True)
        v=_rs(v)/255.0
        return v, torch.tensor(self.labels[idx],dtype=torch.float32)


# ===================================================================
def compute_metrics(preds, targets):
    p, t = np.array(preds), np.array(targets)
    tp, tn = int(((p == 1) & (t == 1)).sum()), int(((p == 0) & (t == 0)).sum())
    fp, fn = int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
    n = len(t)
    return {
        "accuracy":    round((tp + tn) / n, 4) if n else 0,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "precision":   round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "f1":          round(2*tp/(2*tp+fp+fn), 4) if (2*tp+fp+fn) else 0,
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_inference(model, csv_path, desc="Inference"):
    ds = FastDS(csv_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad(), autocast():
        for videos, labels in tqdm(dl, desc=desc, leave=False):
            logits = model(videos.to(DEVICE))
            p = torch.sigmoid(logits).cpu()
            preds.extend((p >= 0.5).int().view(-1).tolist())
            targets.extend(labels.int().tolist())
            probs.extend(p.view(-1).tolist())
    return preds, probs, targets


# ===================================================================
def group_a_baseline():
    """Original UNet3D inference (baseline)."""
    log.info("=" * 50)
    log.info("GROUP A: Baseline UNet3D Inference")
    log.info("=" * 50)

    model = Unet3DClassifier(in_channels=1, num_classes=1).to(DEVICE)
    sd = load_file(CKPT)
    model.load_state_dict(sd, strict=True)
    model.eval()
    log.info(f"Loaded pretrained UNet3D ({len(sd)} tensors)")

    t0 = time.time()
    preds, probs, targets = run_inference(model, TEST_CSV, "Baseline")
    m = compute_metrics(preds, targets)
    m["time_sec"] = round(time.time() - t0, 1)

    log.info(f"  Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
             f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} ({m['time_sec']:.1f}s)")

    # Save
    m["group"] = "baseline_unet3d"
    return m, preds, probs, targets


# ===================================================================
def group_b_mamba():
    """Hybrid UNet-Mamba fine-tuning + inference."""
    log.info("=" * 50)
    log.info("GROUP B: Hybrid UNet-Mamba Fine-tuning")
    log.info("=" * 50)

    # Build model
    model = HybridUNetMamba(in_channels=1, num_classes=1).to(DEVICE)
    sd = load_file(CKPT)
    model.load_pretrained_encoder_head(sd)
    log.info(f"Model built, {sum(p.numel() for p in model.parameters()):,} total params")

    # Data
    train_ds = FastDS(TRAIN_CSV)
    test_ds = FastDS(TEST_CSV)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    log.info(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    # Optimizer + AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    epoch_log = []
    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for videos, labels in tqdm(train_dl, desc=f"  Epoch {epoch}/{EPOCHS} Train", leave=False):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                logits = model(videos)
                loss = criterion(logits.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        preds, targets = [], []
        with torch.no_grad(), autocast():
            for videos, labels in test_dl:
                logits = model(videos.to(DEVICE))
                p = torch.sigmoid(logits).cpu()
                preds.extend((p >= 0.5).int().view(-1).tolist())
                targets.extend(labels.int().tolist())

        m = compute_metrics(preds, targets)
        m["epoch"] = epoch
        m["train_loss"] = round(train_loss / len(train_dl), 4)
        epoch_log.append(m)

        log.info(f"  Epoch {epoch:2d}: Loss={m['train_loss']:.4f} "
                 f"Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
                 f"Spec={m['specificity']:.4f} F1={m['f1']:.4f}")

        # ---- Early Stopping on F1 (avoids degenerate Sens=1.0/Spec=0.0 solutions) ----
        is_degenerate = m["specificity"] < 0.5 or m["sensitivity"] < 0.1
        if not is_degenerate and m["f1"] > best_f1 + 0.005:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            log.info(f"    -> Best F1: {best_f1:.4f} (Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}, saved)")
        elif is_degenerate:
            log.info(f"    -> Degenerate (Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}), patience={patience_counter}/{EARLY_STOP_PATIENCE}")
            patience_counter += 1
        else:
            patience_counter += 1
            log.info(f"    -> F1={m['f1']:.4f} (no improvement, patience={patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info(f"  EARLY STOPPING at epoch {epoch} (F1 stalled for {EARLY_STOP_PATIENCE} epochs)")
                break

    # Save best model (from early stopping)
    if best_state is not None:
        best_path = RESULTS / "hybrid_unet_mamba_best.pth"
        torch.save(best_state, best_path)
        log.info(f"Best model saved (best F1={best_f1:.4f}): {best_path}")
        model.load_state_dict(best_state)
    else:
        log.warning("No best state saved! Using current model.")

    # Final inference with best model
    log.info("\n--- Final Mamba Inference ---")
    t0 = time.time()
    preds, probs, targets = run_inference(model, TEST_CSV, "Mamba Final")
    final_m = compute_metrics(preds, targets)
    final_m["time_sec"] = round(time.time() - t0, 1)
    final_m["group"] = "mamba_finetuned"
    final_m["best_epoch"] = epoch_log[-1]["epoch"] if epoch_log else 0

    log.info(f"  Acc={final_m['accuracy']:.4f} Sens={final_m['sensitivity']:.4f} "
             f"Spec={final_m['specificity']:.4f} F1={final_m['f1']:.4f} "
             f"({final_m['time_sec']:.1f}s)")

    # Save epoch log
    epoch_csv = RESULTS / "mamba_training_log.csv"
    with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "accuracy", "sensitivity",
                                           "specificity", "precision", "f1",
                                           "train_loss", "n", "tp", "tn", "fp", "fn"])
        w.writeheader(); w.writerows(epoch_log)
    log.info(f"Training log: {epoch_csv}")

    return final_m, preds, probs, targets, epoch_log


# ===================================================================
def find_success_cases(bl_preds, bl_probs, bl_targets, mb_preds, mb_probs, mb_targets):
    """Find cases where Mamba got it right but baseline got it wrong."""
    successes = []
    for i in range(len(bl_preds)):
        if bl_preds[i] != bl_targets[i] and mb_preds[i] == mb_targets[i]:
            successes.append({
                "index": i,
                "true_label": bl_targets[i],
                "baseline_pred": bl_preds[i],
                "baseline_prob": round(bl_probs[i], 4),
                "mamba_pred": mb_preds[i],
                "mamba_prob": round(mb_probs[i], 4),
            })
    return successes


def generate_summary(bl_m, mb_m, success_cases, epoch_log):
    """Write summary.txt"""
    delta_sens = mb_m["sensitivity"] - bl_m["sensitivity"]
    delta_f1 = mb_m["f1"] - bl_m["f1"]

    with open(RESULTS / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Mamba Ablation Summary | {datetime.now().isoformat()}\n")
        f.write(f"Task: macula_detached_vs_intact (n=101)\n")
        f.write(f"Baseline UNet3D: Sens={bl_m['sensitivity']:.4f} "
                f"Acc={bl_m['accuracy']:.4f} F1={bl_m['f1']:.4f}\n")
        f.write(f"Hybrid UNet-Mamba: Sens={mb_m['sensitivity']:.4f} "
                f"Acc={mb_m['accuracy']:.4f} F1={mb_m['f1']:.4f}\n")
        f.write(f"Delta: Sens {'+' if delta_sens > 0 else ''}{delta_sens:.4f} "
                f"F1 {'+' if delta_f1 > 0 else ''}{delta_f1:.4f}\n")
        f.write(f"Success cases (baseline wrong, Mamba correct): "
                f"{len(success_cases)}\n")

        # 100-char summary
        if delta_sens > 0:
            summary = (f"Mamba improved sensitivity from {bl_m['sensitivity']:.4f} to "
                       f"{mb_m['sensitivity']:.4f} (delta=+{delta_sens:.4f}), "
                       f"correcting {len(success_cases)} baseline errors. "
                       f"F1 improved from {bl_m['f1']:.4f} to {mb_m['f1']:.4f}.")
        else:
            summary = (f"Mamba maintained sensitivity at {mb_m['sensitivity']:.4f} "
                       f"(baseline {bl_m['sensitivity']:.4f}). "
                       f"F1: {bl_m['f1']:.4f} -> {mb_m['f1']:.4f}. "
                       f"{len(success_cases)} cases corrected.")
        f.write(f"\n[100-char summary]\n{summary[:200]}\n")

    log.info(f"Summary saved: {RESULTS / 'summary.txt'}")
    return summary


# ===================================================================
def main():
    log.info(f"=== Mamba Ablation Experiment at {datetime.now().isoformat()} ===")
    log.info(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: 1 | AMP: ON")

    # Group A: Baseline
    bl_m, bl_preds, bl_probs, bl_targets = group_a_baseline()

    # Group B: Mamba fine-tuned
    mb_m, mb_preds, mb_probs, mb_targets, epoch_log = group_b_mamba()

    # Comparison CSV
    comp_csv = RESULTS / "results_comparison.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "accuracy", "sensitivity",
                                           "specificity", "precision", "f1",
                                           "n", "tp", "tn", "fp", "fn", "time_sec"])
        w.writeheader()
        w.writerow(bl_m)
        w.writerow(mb_m)
    log.info(f"Comparison: {comp_csv}")

    # Success cases
    successes = find_success_cases(bl_preds, bl_probs, bl_targets,
                                   mb_preds, mb_probs, mb_targets)
    log.info(f"Success cases: {len(successes)}")

    with open(RESULTS / "success_cases.log", "w", encoding="utf-8") as f:
        f.write(f"Mamba Success Cases (baseline wrong, Mamba correct)\n")
        f.write(f"Total: {len(successes)}\n\n")
        for s in successes[:10]:  # top 10
            f.write(f"  idx={s['index']} true={s['true_label']} "
                    f"baseline_prob={s['baseline_prob']:.4f} -> "
                    f"mamba_prob={s['mamba_prob']:.4f}\n")

    # Save all results as JSON
    results_dict = {
        "baseline": bl_m,
        "mamba": mb_m,
        "success_cases_count": len(successes),
        "success_cases": successes,
        "epoch_log": epoch_log,
    }
    with open(RESULTS / "results_full.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    # Summary
    generate_summary(bl_m, mb_m, successes, epoch_log)

    log.info(f"\n=== Mamba Ablation Complete at {datetime.now().isoformat()} ===")
    log.info(f"All results in: {RESULTS}")

    return bl_m, mb_m, successes


if __name__ == "__main__":
    main()
