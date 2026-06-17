"""
Batch evaluation of pretrained models on all 5 ERDES classification tasks.
Downloads missing weights from HuggingFace, evaluates on test sets, saves results.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file

os.environ["PYTHONIOENCODING"] = "utf-8"

# Fix broken SSL_CERT_FILE in conda env (points to .../erdes/ssl/cacert.pem but actual file is at .../erdes/Library/ssl/cacert.pem)
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        print(f"[Fixed SSL_CERT_FILE: {certifi.where()}]")
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)
        print("[Unset broken SSL_CERT_FILE]")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from erdes.models.components.factory import build_3d_architecture
from erdes.data.components.erdes_dataset import VideoDataset

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Paper benchmarks (from Tables 8, 9, 10)
# ---------------------------------------------------------------------------
PAPER_BENCHMARKS = {
    "non_rd_vs_rd": {
        "unet3d":      {"acc": 0.982, "sens": 0.920, "spec": 0.988, "prec": 0.893, "f1": 0.906},
        "resnet3d":    {"acc": 0.974, "sens": 0.939, "spec": 0.978, "prec": 0.817, "f1": 0.874},
    },
    "normal_vs_rd": {
        "unet3d":      {"acc": 0.991, "sens": 0.950, "spec": 0.996, "prec": 0.969, "f1": 0.959},
    },
    "pvd_vs_rd": {
        "unetplusplus":{"acc": 0.887, "sens": 0.921, "spec": 0.860, "prec": 0.839, "f1": 0.878},
    },
    "macula_detached_vs_intact": {
        "unet3d":      {"acc": 0.882, "sens": 0.899, "spec": 0.870, "prec": 0.818, "f1": 0.857},
    },
    "normal_vs_pvd": {
        "unet3d":      {"acc": 0.959, "sens": 0.784, "spec": 0.985, "prec": 0.894, "f1": 0.836},
    },
}

# ---------------------------------------------------------------------------
# Evaluation config: (task, model_name, hf_repo, test_csv)
# ---------------------------------------------------------------------------
EVAL_CONFIGS = [
    # Task 1: Non-RD vs RD
    {
        "task": "non_rd_vs_rd",
        "model": "resnet3d",
        "hf_repo": "pcvlab/resnet3d_non_rd_vs_rd",
        "test_csv": "data/splits/non_rd_vs_rd/test.csv",
    },
    {
        "task": "non_rd_vs_rd",
        "model": "unet3d",
        "hf_repo": "pcvlab/unet3d_non_rd_vs_rd",
        "test_csv": "data/splits/non_rd_vs_rd/test.csv",
    },
    # Task 2: Normal vs RD
    {
        "task": "normal_vs_rd",
        "model": "unet3d",
        "hf_repo": "pcvlab/unet3d_normal_vs_rd",
        "test_csv": "data/splits/normal_vs_rd/test.csv",
    },
    # Task 3: PVD vs RD
    {
        "task": "pvd_vs_rd",
        "model": "unetplusplus",
        "hf_repo": "pcvlab/unetplusplus_pvd_vs_rd",
        "test_csv": "data/splits/pvd_vs_rd/test.csv",
    },
    # Task 4: Macula Detached vs Intact
    {
        "task": "macula_detached_vs_intact",
        "model": "unet3d",
        "hf_repo": "pcvlab/unet3d_macula_detached_vs_intact",
        "test_csv": "data/splits/macula_detached_vs_intact/test.csv",
    },
    # Task 5: Normal vs PVD
    {
        "task": "normal_vs_pvd",
        "model": "unet3d",
        "hf_repo": "pcvlab/unet3d_normal_vs_pvd",
        "test_csv": "data/splits/normal_vs_pvd/test.csv",
    },
]


def compute_metrics(preds, targets):
    """Compute binary classification metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    n = len(targets)
    acc = (tp + tn) / n if n > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "acc": round(acc, 4),
        "sens": round(sens, 4),
        "spec": round(spec, 4),
        "prec": round(prec, 4),
        "f1": round(f1, 4),
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def download_from_hf(repo_id: str, save_name: str) -> Path:
    """Download model.safetensors from HuggingFace repo using requests."""
    import requests

    target = WEIGHTS_DIR / save_name
    if target.exists() and target.stat().st_size > 1000:
        return target

    url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"
    print(f"  Downloading from: {url}")

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(target, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MB "
                          f"({downloaded / total * 100:.0f}%)", end="\r")
    print(f"  Downloaded: {target.stat().st_size / 1024**2:.1f} MB")
    return target


def evaluate_one(cfg: dict, device: str, batch_size: int = 4):
    """Evaluate a single model on its test set."""
    task = cfg["task"]
    model_name = cfg["model"]
    hf_repo = cfg["hf_repo"]
    test_csv = PROJECT_ROOT / cfg["test_csv"]

    safetensors_name = f"{model_name}_{task}.safetensors"
    safetensors_path = WEIGHTS_DIR / safetensors_name

    # Download if missing
    if not safetensors_path.exists() or safetensors_path.stat().st_size < 1000:
        print(f"  Downloading {hf_repo} ...")
        safetensors_path = download_from_hf(hf_repo, safetensors_name)
    else:
        print(f"  Using cached {safetensors_name} ({safetensors_path.stat().st_size / 1024**2:.1f} MB)")

    print(f"  Building model: {model_name}")
    model = build_3d_architecture(model_name, num_classes=1)
    model = model.to(device)
    model.eval()

    # Load weights
    state_dict = load_file(str(safetensors_path))
    # Auto-detect prefix
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Try to match keys
    if model_keys != ckpt_keys:
        # Add 'model.' prefix if needed
        if all(not k.startswith("model.") for k in ckpt_keys) and any(k.startswith("model.") for k in model_keys):
            state_dict = {"model." + k: v for k, v in state_dict.items()}
        elif all(not k.startswith("enc.") for k in ckpt_keys) and any(k.startswith("enc.") for k in model_keys):
            state_dict = {"enc." + k: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

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
    all_preds, all_targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc=f"  {task}/{model_name}"):
            videos = videos.to(device)
            logits = model(videos)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().view(-1)
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.int().tolist())

    elapsed = time.time() - t0
    metrics = compute_metrics(all_preds, all_targets)
    metrics["time_sec"] = round(elapsed, 1)
    metrics["model"] = model_name
    metrics["task"] = task

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {}

    for i, cfg in enumerate(EVAL_CONFIGS):
        task = cfg["task"]
        model = cfg["model"]
        key = f"{task}/{model}"
        print(f"\n[{i+1}/{len(EVAL_CONFIGS)}] {key}")
        print("-" * 50)

        try:
            metrics = evaluate_one(cfg, device, batch_size=4)
            all_results[key] = metrics

            # Print immediate results
            print(f"  ACC={metrics['acc']:.4f}  SENS={metrics['sens']:.4f}  "
                  f"SPEC={metrics['spec']:.4f}  PREC={metrics['prec']:.4f}  "
                  f"F1={metrics['f1']:.4f}  ({metrics['time_sec']:.1f}s)")

            # Compare with paper if available
            paper = PAPER_BENCHMARKS.get(task, {}).get(model)
            if paper:
                for m in ["acc", "sens", "spec", "prec", "f1"]:
                    delta = metrics[m] - paper[m]
                    flag = " ✓" if abs(delta) <= 0.02 else " ✗"
                    print(f"    {m}: ours={metrics[m]:.4f}  paper={paper[m]:.4f}  Δ={delta:+.4f}{flag}")

        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = {"error": str(e)}

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print("\n\n")
    print("=" * 100)
    print("FINAL RESULTS — ALL 5 TASKS")
    print("=" * 100)
    header = f"{'Task':<32} {'Model':<14} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'Time':>8}"
    print(header)
    print("-" * 100)

    for cfg in EVAL_CONFIGS:
        key = f"{cfg['task']}/{cfg['model']}"
        r = all_results.get(key, {})
        if "error" in r:
            print(f"{cfg['task']:<32} {cfg['model']:<14} {'ERROR: ' + r['error'][:40]}")
        else:
            print(f"{cfg['task']:<32} {cfg['model']:<14} "
                  f"{r.get('acc', 0):>8.4f} {r.get('sens', 0):>8.4f} "
                  f"{r.get('spec', 0):>8.4f} {r.get('prec', 0):>8.4f} "
                  f"{r.get('f1', 0):>8.4f} {r.get('time_sec', 0):>7.1f}s")

    print("-" * 100)

    # Paper comparison
    print("\n--- Paper Comparison ---")
    print(f"{'Task':<32} {'Model':<14} {'Metric':<8} {'Paper':>8} {'Ours':>8} {'Δ':>8} {'Match':>6}")
    print("-" * 90)
    for task, models in PAPER_BENCHMARKS.items():
        for model, paper_metrics in models.items():
            key = f"{task}/{model}"
            our = all_results.get(key, {}).copy() if isinstance(all_results.get(key), dict) else {}
            if "error" in our:
                continue
            for m in ["acc", "sens", "spec", "prec", "f1"]:
                pv = paper_metrics[m]
                ov = our.get(m)
                if ov is not None:
                    delta = ov - pv
                    match = "OK" if abs(delta) <= 0.02 else ("HIGH" if delta > 0 else "LOW")
                    print(f"{task:<32} {model:<14} {m:<8} {pv:>8.4f} {ov:>8.4f} {delta:>+8.4f} {match:>6}")

    print("-" * 90)
    print("Δ within ±0.02 → OK   (accounting for hardware/runtime variance)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"benchmark_5tasks_{timestamp}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    # Also save as latest
    latest_path = RESULTS_DIR / "benchmark_5tasks_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {latest_path}")

    return all_results


if __name__ == "__main__":
    main()
