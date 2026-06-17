"""
Sampling Ratio Ablation (decord-accelerated)
=============================================
Tests r ∈ {0.3, 0.5, 0.7, 1.0} on Stage 2 UNet3D.
Uses decord for 10x faster video loading.
"""
import os, sys, time, csv, logging
from pathlib import Path
from datetime import datetime

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(f"=== Sampling Ablation (decord) started at {datetime.now().isoformat()} ===")

from erdes.models.components.cls_model import Unet3DClassifier
from erdes.data.components.utils import resize


class FastVideoDataset(Dataset):
    def __init__(self, csv_path, size, data_root=""):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.paths = [os.path.join(data_root, p) if data_root else p for p in self.df["path"]]
        self.labels = self.df["label"].tolist()
        self.resize_tf = resize(size)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            vr = decord.VideoReader(str(self.paths[idx]))
            n = len(vr)
            frames = vr.get_batch(list(range(n))).asnumpy()
        except:
            return torch.zeros(1, 96, 128, 128), torch.tensor(self.labels[idx], dtype=torch.float32)

        video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)
        video = self.resize_tf(video) / 255.0
        return video, torch.tensor(self.labels[idx], dtype=torch.float32)


def compute_metrics(preds, targets):
    p, t = np.array(preds), np.array(targets)
    tp, tn = int(((p == 1) & (t == 1)).sum()), int(((p == 0) & (t == 0)).sum())
    fp, fn = int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
    n = len(t)
    return {
        "accuracy": round((tp + tn) / n, 4) if n else 0,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0,
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    test_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
    state_dict = load_file(ckpt)
    log.info(f"Loaded weights: {len(state_dict)} tensors")

    ds = FastVideoDataset(test_csv, size=(96, 128, 128), data_root="data/erdes")
    log.info(f"Test samples: {len(ds)}")

    ratios = [0.3, 0.5, 0.7, 1.0]
    results = []

    for r in ratios:
        log.info(f"\n{'='*50}\nTesting r = {r}\n{'='*50}")
        t0 = time.time()

        pooling = "avg" if r >= 1.0 else "topk"
        model = Unet3DClassifier(in_channels=1, num_classes=1, pooling=pooling, topk_ratio=r if r < 1.0 else 0.5)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()
        torch.cuda.empty_cache()

        bs = 8 if pooling == "avg" else 4
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        preds, targets = [], []
        with torch.no_grad():
            for videos, labels in tqdm(dl, desc="  Inference", leave=False):
                videos = videos.to(device)
                probs = torch.sigmoid(model(videos))
                preds.extend((probs >= 0.5).int().cpu().view(-1).tolist())
                targets.extend(labels.int().tolist())

        m = compute_metrics(preds, targets)
        m["ratio"] = r
        m["time_sec"] = round(time.time() - t0, 1)
        log.info(f"  r={r:.1f}: Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
                 f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} ({m['time_sec']:.1f}s)")
        results.append(m)
        del model, dl
        torch.cuda.empty_cache()

    out = RESULTS_DIR / "sampling_ablation_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ratio", "accuracy", "sensitivity", "specificity", "precision", "f1", "n", "tp", "tn", "fp", "fn", "time_sec"])
        w.writeheader(); w.writerows(results)
    log.info(f"\nSaved: {out}")

    log.info("\n" + "=" * 60)
    log.info(f"{'Ratio':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'Time':>8}")
    for r in results:
        log.info(f"{r['ratio']:>8.1f} {r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f} {r['f1']:>8.4f} {r['time_sec']:>7.1f}s")
    best = max(results, key=lambda x: x["f1"])
    log.info(f"\nBest F1: r={best['ratio']:.1f} (F1={best['f1']:.4f})")
    log.info(f"=== Completed at {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
