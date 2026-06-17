"""
Cascade Pipeline Eval (decord-accelerated + GPU-optimized)
============================================================
Key optimizations:
  1. decord for 10x faster video decoding (keeps GPU busy)
  2. num_workers=2 for parallel CPU decode while GPU computes
  3. batch_size=8 for better GPU utilization
  4. Windows-safe multiprocessing via 'spawn' context
"""
import os, sys, time, csv, logging
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

# Windows-safe multiprocessing
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import decord
from tqdm import tqdm
from safetensors.torch import load_file

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "exp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "runtime.log"
FAILURE_LOG = RESULTS_DIR / "failure_cases.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

from erdes.models.components.factory import build_3d_architecture
from erdes.data.components.utils import resize


# ---------------------------------------------------------------------------
# FastVideoDataset using decord (10x faster than torchvision)
# ---------------------------------------------------------------------------
class FastVideoDataset(Dataset):
    """Video dataset using decord for fast CPU decoding."""
    def __init__(self, csv_path, size, data_root=""):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.paths = [os.path.join(data_root, p) if data_root else p for p in self.df["path"]]
        self.labels = self.df["label"].tolist()
        self.size = size
        self.resize_tf = resize(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        try:
            vr = decord.VideoReader(str(p))
            n = len(vr)
            frames = vr.get_batch(list(range(n))).asnumpy()  # [T, H, W, C]
        except Exception:
            D, H, W = self.size
            return torch.zeros(1, D, H, W), label

        # [T, H, W, C] → float → [C, T, H, W] → grayscale
        video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]
        video = self.resize_tf(video) / 255.0
        return video, label


def load_model(arch, ckpt, device):
    model = build_3d_architecture(arch, num_classes=1).to(device).eval()
    sd = load_file(ckpt)
    if any(k.startswith("enc.") for k in model.state_dict()) and not any(k.startswith("enc.") for k in sd):
        sd = {"enc." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def run_inference(model, test_csv, device, batch_size=8):
    ds = FastVideoDataset(test_csv, size=(96, 128, 128), data_root="data/erdes")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # num_workers=0 because decord is already fast; spawn workers add overhead
    preds, targets, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for videos, labels in tqdm(dl, desc="  Inference", leave=False):
            videos = videos.to(device)
            logits = model(videos)
            p = torch.sigmoid(logits).cpu()
            preds.extend((p >= 0.5).int().view(-1).tolist())
            targets.extend(labels.int().tolist())
            probs.extend(p.view(-1).tolist())
    return preds, probs, targets, ds.paths


def metrics(preds, targets):
    p, t = np.array(preds), np.array(targets)
    tp, tn = int(((p == 1) & (t == 1)).sum()), int(((p == 0) & (t == 0)).sum())
    fp, fn = int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
    n = len(t)
    return {
        "acc": round((tp + tn) / n, 4) if n else 0,
        "sens": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "spec": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "prec": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0,
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device} | Started: {datetime.now().isoformat()}")

    s1_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_non_rd_vs_rd.safetensors")
    s2_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    model_s1 = load_model("unet3d", s1_ckpt, device)
    model_s2 = load_model("unet3d", s2_ckpt, device)
    log.info("Models loaded (decord-based dataset)")

    # Stage 1
    s1_csv = str(PROJECT_ROOT / "data" / "splits" / "non_rd_vs_rd" / "test.csv")
    log.info("--- Stage 1: RD Detection ---")
    t0 = time.time()
    s1_p, s1_prob, s1_t, s1_paths = run_inference(model_s1, s1_csv, device, batch_size=8)
    s1_m = metrics(s1_p, s1_t)
    log.info(f"  Acc={s1_m['acc']:.4f} Sens={s1_m['sens']:.4f} Spec={s1_m['spec']:.4f} F1={s1_m['f1']:.4f} ({time.time()-t0:.1f}s)")

    # Stage 2
    s2_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
    log.info("--- Stage 2: Macula Status ---")
    t0 = time.time()
    s2_p, s2_prob, s2_t, s2_paths = run_inference(model_s2, s2_csv, device, batch_size=8)
    s2_m = metrics(s2_p, s2_t)
    log.info(f"  Acc={s2_m['acc']:.4f} Sens={s2_m['sens']:.4f} Spec={s2_m['spec']:.4f} F1={s2_m['f1']:.4f} ({time.time()-t0:.1f}s)")

    combined = s1_m['sens'] * s2_m['sens']
    log.info(f"\nCombined Sensitivity: {s1_m['sens']:.4f} × {s2_m['sens']:.4f} = {combined:.4f}")

    # Failure cases
    s1_fn = sorted([i for i in range(len(s1_p)) if s1_t[i] == 1 and s1_p[i] == 0],
                   key=lambda i: s1_prob[i])[:5]
    s2_err = sorted([i for i in range(len(s2_p)) if s2_t[i] != s2_p[i]],
                    key=lambda i: abs(s2_prob[i] - 0.5))[:5]

    with open(FAILURE_LOG, "w", encoding="utf-8") as f:
        f.write(f"ERDES Pipeline Failure Cases | {datetime.now().isoformat()}\n")
        f.write(f"Combined Sensitivity: {s1_m['sens']:.4f} × {s2_m['sens']:.4f} = {combined:.4f}\n\n")
        f.write("--- Stage 1 FN Top-5 (RD missed) ---\n")
        for i, idx in enumerate(s1_fn, 1):
            f.write(f"  [{i}] prob={s1_prob[idx]:.4f} true=1 pred=0 | {s1_paths[idx]}\n")
        f.write(f"\n--- Stage 2 Errors Top-5 (Macula status wrong) ---\n")
        for i, idx in enumerate(s2_err, 1):
            et = "FN(Detached→Intact)" if s2_t[idx] == 1 else "FP(Intact→Detached)"
            f.write(f"  [{i}] prob={s2_prob[idx]:.4f} true={s2_t[idx]} pred={s2_p[idx]} {et} | {s2_paths[idx]}\n")
    log.info(f"Failure cases: {FAILURE_LOG}")

    # Summary CSV
    sp = RESULTS_DIR / "pipeline_summary.csv"
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "model", "accuracy", "sensitivity", "specificity", "precision", "f1", "tp", "tn", "fp", "fn", "combined_sensitivity"])
        w.writeheader()
        w.writerow({"stage": "stage1_rd", "model": "unet3d", **s1_m, "combined_sensitivity": ""})
        w.writerow({"stage": "stage2_macula", "model": "unet3d", **s2_m, "combined_sensitivity": ""})
        w.writerow({"stage": "pipeline", "model": "unet3d+unet3d", "sensitivity": round(combined, 4), "combined_sensitivity": round(combined, 4)})
    log.info(f"Summary: {sp}")
    log.info(f"=== Completed at {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
