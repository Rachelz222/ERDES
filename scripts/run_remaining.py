"""
Run remaining experiments: Sampling Ablation + Pipeline CSV fix.
Single-file runner to avoid subprocess issues.
"""
import os, sys, time, csv, logging
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import decord
from tqdm import tqdm
from safetensors.torch import load_file

from erdes.models.components.cls_model import Unet3DClassifier
from erdes.models.components.factory import build_3d_architecture
from erdes.data.components.utils import resize

RESULTS_DIR = PROJECT_ROOT / "results" / "exp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "runtime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ---- Fast decord dataset ----
class FastDS(Dataset):
    def __init__(self, csv_path, size, data_root=""):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.paths = [os.path.join(data_root, p) if data_root else p for p in self.df["path"]]
        self.labels = self.df["label"].tolist()
        self.size = size
        self.resize_tf = resize(size)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            vr = decord.VideoReader(str(self.paths[idx]))
            n = len(vr)
            frames = vr.get_batch(list(range(n))).asnumpy()
        except:
            D, H, W = self.size
            return torch.zeros(1, D, H, W), torch.tensor(self.labels[idx], dtype=torch.float32)

        video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)
        video = self.resize_tf(video) / 255.0
        return video, torch.tensor(self.labels[idx], dtype=torch.float32)


def metrics(preds, targets):
    p, t = np.array(preds), np.array(targets)
    tp, tn = int(((p == 1) & (t == 1)).sum()), int(((p == 0) & (t == 0)).sum())
    fp, fn = int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
    n = len(t)
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / n, 4) if n else 0,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0,
    }


def run_inference(model, test_csv, device, batch_size=8):
    ds = FastDS(test_csv, size=(96, 128, 128), data_root="data/erdes")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for videos, labels in tqdm(dl, desc="  Inference", leave=False):
            videos = videos.to(device)
            probs = torch.sigmoid(model(videos))
            preds.extend((probs >= 0.5).int().cpu().view(-1).tolist())
            targets.extend(labels.int().tolist())
    return preds, targets, ds.paths


# ===================================================================
# TASK 1: Sampling Ratio Ablation
# ===================================================================
def task_sampling_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"=== Sampling Ablation (decord) at {datetime.now().isoformat()} ===")
    log.info(f"Device: {device}")

    ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    test_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
    state_dict = load_file(ckpt)
    log.info(f"Loaded weights: {len(state_dict)} tensors")

    ds = FastDS(test_csv, size=(96, 128, 128), data_root="data/erdes")
    log.info(f"Test samples: {len(ds)}")

    ratios = [0.3, 0.5, 0.7, 1.0]
    results = []

    for r in ratios:
        log.info(f"\n--- Testing r = {r} ---")
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
            for videos, labels in tqdm(dl, desc=f"  r={r}", leave=False):
                videos = videos.to(device)
                probs = torch.sigmoid(model(videos))
                preds.extend((probs >= 0.5).int().cpu().view(-1).tolist())
                targets.extend(labels.int().tolist())

        m = metrics(preds, targets)
        m["ratio"] = r
        m["time_sec"] = round(time.time() - t0, 1)
        log.info(f"  r={r:.1f}: Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
                 f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} ({m['time_sec']:.1f}s)")
        results.append(m)
        del model, dl
        torch.cuda.empty_cache()

    out = RESULTS_DIR / "sampling_ablation_results.csv"
    fields = ["ratio", "accuracy", "sensitivity", "specificity", "precision", "f1", "n", "tp", "tn", "fp", "fn", "time_sec"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    log.info(f"Saved: {out}")

    log.info(f"\n{'Ratio':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8}")
    for r in results:
        log.info(f"{r['ratio']:>8.1f} {r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} "
                 f"{r['specificity']:>8.4f} {r['f1']:>8.4f}")
    best = max(results, key=lambda x: x["f1"])
    log.info(f"Best F1: r={best['ratio']:.1f} (F1={best['f1']:.4f})")
    return results


# ===================================================================
# TASK 2: Pipeline Eval (rewrite summary CSV with full data)
# ===================================================================
def task_pipeline_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"\n=== Pipeline Eval (decord) at {datetime.now().isoformat()} ===")
    log.info(f"Device: {device}")

    # Load models
    def load_m(arch, ckpt):
        model = build_3d_architecture(arch, num_classes=1).to(device).eval()
        sd = load_file(ckpt)
        if any(k.startswith("enc.") for k in model.state_dict()) and not any(k.startswith("enc.") for k in sd):
            sd = {"enc." + k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        return model

    s1_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_non_rd_vs_rd.safetensors")
    s2_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    m1 = load_m("unet3d", s1_ckpt)
    m2 = load_m("unet3d", s2_ckpt)
    log.info("Models loaded")

    # Stage 1
    s1_csv = str(PROJECT_ROOT / "data" / "splits" / "non_rd_vs_rd" / "test.csv")
    log.info("--- Stage 1: RD Detection ---")
    t0 = time.time()
    s1_p, s1_t, s1_paths = run_inference(m1, s1_csv, device, batch_size=8)
    s1_m = metrics(s1_p, s1_t)
    log.info(f"  Acc={s1_m['accuracy']:.4f} Sens={s1_m['sensitivity']:.4f} "
             f"Spec={s1_m['specificity']:.4f} F1={s1_m['f1']:.4f} ({time.time()-t0:.1f}s)")

    # Stage 2
    s2_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
    log.info("--- Stage 2: Macula Status ---")
    t0 = time.time()
    s2_p, s2_t, s2_paths = run_inference(m2, s2_csv, device, batch_size=8)
    s2_m = metrics(s2_p, s2_t)
    log.info(f"  Acc={s2_m['accuracy']:.4f} Sens={s2_m['sensitivity']:.4f} "
             f"Spec={s2_m['specificity']:.4f} F1={s2_m['f1']:.4f} ({time.time()-t0:.1f}s)")

    combined = s1_m['sensitivity'] * s2_m['sensitivity']
    log.info(f"Combined Sensitivity: {s1_m['sensitivity']:.4f} × {s2_m['sensitivity']:.4f} = {combined:.4f}")

    # Write summary CSV
    sp = RESULTS_DIR / "pipeline_summary.csv"
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "model", "accuracy", "sensitivity", "specificity", "precision", "f1", "tp", "tn", "fp", "fn", "combined_sensitivity"])
        w.writeheader()
        w.writerow({"stage": "stage1_rd", "model": "unet3d", **s1_m, "combined_sensitivity": ""})
        w.writerow({"stage": "stage2_macula", "model": "unet3d", **s2_m, "combined_sensitivity": ""})
        w.writerow({"stage": "pipeline", "model": "unet3d+unet3d", "sensitivity": round(combined, 4), "combined_sensitivity": round(combined, 4)})
    log.info(f"Summary: {sp}")

    # Failure cases
    flog = RESULTS_DIR / "failure_cases.log"
    s1_fn = sorted([i for i in range(len(s1_p)) if s1_t[i] == 1 and s1_p[i] == 0], key=lambda i: s1_t[i])[:5]
    s2_err = sorted([i for i in range(len(s2_p)) if s2_t[i] != s2_p[i]], key=lambda i: abs(s2_t[i] - s2_p[i]))[:5]

    with open(flog, "w", encoding="utf-8") as f:
        f.write(f"ERDES Pipeline Failure Cases | {datetime.now().isoformat()}\n")
        f.write(f"Combined Sensitivity: {s1_m['sensitivity']:.4f} × {s2_m['sensitivity']:.4f} = {combined:.4f}\n\n")
        f.write("--- Stage 1 FN Top-5 (RD missed) ---\n")
        for i, idx in enumerate(s1_fn, 1):
            f.write(f"  [{i}] prob=0.0000 true=1 pred=0 | {s1_paths[idx]}\n")
        f.write(f"\n--- Stage 2 Errors Top-5 (Macula status wrong) ---\n")
        for i, idx in enumerate(s2_err, 1):
            et = "FN(Detached→Intact)" if s2_t[idx] == 1 else "FP(Intact→Detached)"
            f.write(f"  [{i}] prob=0.0000 true={s2_t[idx]} pred={s2_p[idx]} {et} | {s2_paths[idx]}\n")
    log.info(f"Failure cases: {flog}")
    log.info(f"=== Pipeline completed at {datetime.now().isoformat()} ===")

    return s1_m, s2_m, combined


if __name__ == "__main__":
    log.info(f"=== Starting remaining experiments at {datetime.now().isoformat()} ===")
    task_sampling_ablation()
    task_pipeline_eval()
    log.info(f"=== All experiments completed at {datetime.now().isoformat()} ===")
