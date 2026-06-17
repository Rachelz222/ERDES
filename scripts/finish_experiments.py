"""
Finish missing experiments: sampling ablation r=0.7/1.0 + pipeline summary.
All results computed from actual inference runs — no hardcoded values.
"""
import os, sys, csv, time, logging
from pathlib import Path
from datetime import datetime
os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

from erdes.models.components.cls_model import Unet3DClassifier
from erdes.data.components.erdes_dataset import VideoDataset
from erdes.models.components.factory import build_3d_architecture

RESULTS = PROJECT_ROOT / "results" / "exp"
RESULTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(RESULTS / "runtime.log", mode="a", encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)


def compute_metrics(preds, targets):
    p, t = np.array(preds), np.array(targets)
    tp = int(((p == 1) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    n = len(t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "accuracy": round((tp + tn) / n, 4), "sensitivity": round(sens, 4),
        "specificity": round(spec, 4), "precision": round(prec, 4),
        "f1": round(f1, 4), "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_inference(model, test_csv, device, batch_size=4):
    ds = VideoDataset(csv_path=test_csv, size=(96, 128, 128), data_root="data/erdes")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for videos, labels in tqdm(dl, desc="  Inference", leave=False):
            logits = model(videos.to(device))
            probs = torch.sigmoid(logits).cpu()
            preds.extend((probs >= 0.5).int().view(-1).tolist())
            targets.extend(labels.int().tolist())
    return preds, targets


# ============================================================
log.info(f"=== Finishing experiments at {datetime.now().isoformat()} ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {device}")

ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
test_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
sd = load_file(ckpt)

# ---- TASK 1: Sampling ablation r=0.7 and r=1.0 ----
results = []
for r in [0.7, 1.0]:
    log.info(f"\n=== Sampling r={r} ===")
    t0 = time.time()
    pooling = "avg" if r >= 1.0 else "topk"
    model = Unet3DClassifier(in_channels=1, num_classes=1, pooling=pooling,
                             topk_ratio=r if r < 1.0 else 0.5)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    torch.cuda.empty_cache()

    bs = 4 if pooling == "avg" else 2  # smaller batch for topk to avoid OOM
    preds, targets = run_inference(model, test_csv, device, batch_size=bs)
    m = compute_metrics(preds, targets)
    m["ratio"], m["time_sec"] = r, round(time.time() - t0, 1)
    log.info(f"  r={r:.1f}: Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
             f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} ({m['time_sec']:.1f}s)")
    results.append(m)
    del model; torch.cuda.empty_cache()

# Also write the full ablation CSV with all 4 ratios
# r=0.3 and r=0.5 were computed in earlier runs — let me re-run for completeness
log.info("\n=== Re-running r=0.3 and r=0.5 for completeness ===")
for r in [0.3, 0.5]:
    t0 = time.time()
    pooling = "topk"
    model = Unet3DClassifier(in_channels=1, num_classes=1, pooling=pooling, topk_ratio=r)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    torch.cuda.empty_cache()
    preds, targets = run_inference(model, test_csv, device, batch_size=2)
    m = compute_metrics(preds, targets)
    m["ratio"], m["time_sec"] = r, round(time.time() - t0, 1)
    log.info(f"  r={r:.1f}: Acc={m['accuracy']:.4f} Sens={m['sensitivity']:.4f} "
             f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} ({m['time_sec']:.1f}s)")
    results.insert(0, m)  # prepend to keep sorted order
    del model; torch.cuda.empty_cache()

# Sort by ratio
results.sort(key=lambda x: x["ratio"])

out = RESULTS / "sampling_ablation_results.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["ratio", "accuracy", "sensitivity", "specificity",
                                       "precision", "f1", "n", "tp", "tn", "fp", "fn", "time_sec"])
    w.writeheader(); w.writerows(results)
log.info(f"\nSaved: {out}")

log.info(f"{'Ratio':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'F1':>8}")
for r in results:
    log.info(f"{r['ratio']:>8.1f} {r['accuracy']:>8.4f} {r['sensitivity']:>8.4f} "
             f"{r['specificity']:>8.4f} {r['f1']:>8.4f}")
best = max(results, key=lambda x: x["f1"])
log.info(f"Best F1: r={best['ratio']:.1f} (F1={best['f1']:.4f})")

# ---- TASK 2: Pipeline summary from actual inference ----
log.info("\n=== Pipeline Summary ===")

def load_model(arch, ckpt_path):
    model = build_3d_architecture(arch, num_classes=1).to(device).eval()
    sd = load_file(ckpt_path)
    if any(k.startswith("enc.") for k in model.state_dict()) and not any(k.startswith("enc.") for k in sd):
        sd = {"enc." + k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

s1_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_non_rd_vs_rd.safetensors")
s2_ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")

log.info("Loading Stage 1 model...")
m1 = load_model("unet3d", s1_ckpt)
log.info("Loading Stage 2 model...")
m2 = load_model("unet3d", s2_ckpt)

# Stage 1
t0 = time.time()
s1_csv = str(PROJECT_ROOT / "data" / "splits" / "non_rd_vs_rd" / "test.csv")
s1_p, s1_t = run_inference(m1, s1_csv, device, batch_size=4)
s1_m = compute_metrics(s1_p, s1_t)
log.info(f"  Stage 1: Acc={s1_m['accuracy']:.4f} Sens={s1_m['sensitivity']:.4f} "
         f"Spec={s1_m['specificity']:.4f} F1={s1_m['f1']:.4f} ({time.time()-t0:.1f}s)")

# Stage 2
t0 = time.time()
s2_csv = str(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
s2_p, s2_t = run_inference(m2, s2_csv, device, batch_size=4)
s2_m = compute_metrics(s2_p, s2_t)
log.info(f"  Stage 2: Acc={s2_m['accuracy']:.4f} Sens={s2_m['sensitivity']:.4f} "
         f"Spec={s2_m['specificity']:.4f} F1={s2_m['f1']:.4f} ({time.time()-t0:.1f}s)")

combined = round(s1_m["sensitivity"] * s2_m["sensitivity"], 4)
log.info(f"  Combined Sensitivity: {s1_m['sensitivity']:.4f} × {s2_m['sensitivity']:.4f} = {combined:.4f}")

# Write pipeline CSV
sp = RESULTS / "pipeline_summary.csv"
with open(sp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["stage", "model", "accuracy", "sensitivity",
                                       "specificity", "precision", "f1", "tp", "tn", "fp", "fn",
                                       "combined_sensitivity"])
    w.writeheader()
    s1_row = {"stage": "stage1_rd", "model": "unet3d", **s1_m, "combined_sensitivity": ""}
    s2_row = {"stage": "stage2_macula", "model": "unet3d", **s2_m, "combined_sensitivity": ""}
    pl_row = {"stage": "pipeline", "model": "unet3d+unet3d", "sensitivity": combined,
              "combined_sensitivity": combined}
    w.writerow(s1_row); w.writerow(s2_row); w.writerow(pl_row)
log.info(f"Saved: {sp}")

# Failure cases
flog = RESULTS / "failure_cases.log"
s1_fn = sorted([i for i in range(len(s1_p)) if s1_t[i] == 1 and s1_p[i] == 0],
               key=lambda i: s1_t[i])[:5]
s2_err = sorted([i for i in range(len(s2_p)) if s2_t[i] != s2_p[i]],
                key=lambda i: abs(s2_t[i] - s2_p[i]))[:5]

with open(flog, "w", encoding="utf-8") as f:
    f.write(f"ERDES Pipeline Failure Cases | {datetime.now().isoformat()}\n")
    f.write(f"Combined Sensitivity: {s1_m['sensitivity']:.4f} × {s2_m['sensitivity']:.4f} = {combined:.4f}\n\n")
    f.write("--- Stage 1 FN Top-5 (RD missed) ---\n")
    for rank, idx in enumerate(s1_fn, 1):
        f.write(f"  [{rank}] true=1 pred=0 | {s1_t[idx]}\n")
    f.write(f"\n--- Stage 2 Errors Top-5 (Macula status wrong) ---\n")
    for rank, idx in enumerate(s2_err, 1):
        et = "FN(Detached→Intact)" if s2_t[idx] == 1 else "FP(Intact→Detached)"
        f.write(f"  [{rank}] true={s2_t[idx]} pred={s2_p[idx]} {et}\n")
log.info(f"Failure cases: {flog}")

log.info(f"\n=== ALL EXPERIMENTS COMPLETE at {datetime.now().isoformat()} ===")
