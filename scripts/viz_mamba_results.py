"""
Mamba Ablation Visualizations
===============================
1. metrics_plot.png — bar chart comparing Group A (Baseline UNet3D) vs Group B (Mamba)
2. gradcam_success_cases.png — Grad-CAM for 3 cases where Mamba corrected baseline errors
"""
import os, sys, json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = PROJECT_ROOT / "results" / "mamba_ablation"
PLOTS = RESULTS

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})


# ===================================================================
def plot_comparison():
    """Bar chart: Group A vs Group B for Acc/Sens/Spec/F1."""
    with open(RESULTS / "results_full.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    bl = data["baseline"]
    mb = data["mamba"]

    metrics = ["accuracy", "sensitivity", "specificity", "f1"]
    labels = ["Accuracy", "Sensitivity", "Specificity", "F1 Score"]
    bl_vals = [bl[m] for m in metrics]
    mb_vals = [mb[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bl_vals, width, label="Baseline UNet3D",
                   color="#3498db", edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width/2, mb_vals, width, label="Hybrid UNet-Mamba",
                   color="#e74c3c", edgecolor="white", linewidth=1)

    # Annotate values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    # Delta annotations
    for i, (bl_v, mb_v) in enumerate(zip(bl_vals, mb_vals)):
        delta = mb_v - bl_v
        color = "#27ae60" if delta >= 0 else "#e74c3c"
        ax.annotate(f"d={delta:+.4f}", (x[i], max(bl_v, mb_v) + 0.04),
                    ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title(f"Mamba Ablation: Baseline vs Hybrid UNet-Mamba\n"
                 f"({data['success_cases_count']} cases corrected)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS / "metrics_plot.png")
    plt.close(fig)
    print("metrics_plot.png saved")


# ===================================================================
def plot_gradcam_success_cases():
    """Grad-CAM for top-3 success cases: Baseline vs Mamba on same video."""
    import torch
    import decord
    from safetensors.torch import load_file
    from erdes.models.components.cls_model import Unet3DClassifier
    from erdes.models.hybrid_unet_mamba import HybridUNetMamba
    from erdes.models.explainer import GradCAM3D
    from erdes.data.components.utils import resize

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(RESULTS / "results_full.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    successes = data.get("success_cases", [])
    if not successes:
        print("No success cases found, skipping Grad-CAM")
        return

    # Load models
    bl_model = Unet3DClassifier(in_channels=1, num_classes=1).to(device).eval()
    ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    sd = load_file(ckpt)
    bl_model.load_state_dict(sd, strict=True)

    mb_model = HybridUNetMamba(in_channels=1, num_classes=1).to(device).eval()
    mb_sd = torch.load(RESULTS / "hybrid_unet_mamba_best.pth", map_location=device)
    mb_model.load_state_dict(mb_sd, strict=False)

    # Get video paths from test CSV
    import pandas as pd
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "macula_detached_vs_intact" / "test.csv")
    data_root = str(PROJECT_ROOT / "data" / "erdes")
    resize_tf = resize((96, 128, 128))

    def load_video(rel_path):
        abs_path = os.path.join(data_root, rel_path)
        try:
            vr = decord.VideoReader(abs_path)
            n = len(vr)
            frames = vr.get_batch(list(range(n))).asnumpy()
        except:
            return None
        video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)
        video = resize_tf(video) / 255.0
        return video.unsqueeze(0)  # [1, 1, D, H, W]

    # Top 3 success cases
    top3 = successes[:3]
    n = len(top3)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(18, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Grad-CAM: Baseline UNet3D vs Hybrid UNet-Mamba on Success Cases\n"
                 "(Cases where Mamba corrected the baseline error)",
                 fontsize=14, fontweight="bold")

    for row, case in enumerate(top3):
        idx = case["index"]
        path = test_df.iloc[idx]["path"]
        true_label = case["true_label"]

        video = load_video(path)
        if video is None:
            for c in range(4):
                axes[row][c].text(0.5, 0.5, "Video unavailable", ha="center", va="center")
            continue

        video = video.to(device)
        D_mid = video.shape[2] // 2
        frame = video[0, 0, D_mid].cpu().numpy()

        # Baseline Grad-CAM
        bl_expl = GradCAM3D(bl_model)
        try:
            with torch.enable_grad():
                bl_hm = bl_expl(video)
        except:
            bl_hm = np.zeros_like(frame)
        bl_expl.remove_hooks()

        # Mamba Grad-CAM
        mb_expl = GradCAM3D(mb_model)
        try:
            with torch.enable_grad():
                mb_hm = mb_expl(video)
        except:
            mb_hm = np.zeros_like(frame)
        mb_expl.remove_hooks()

        bprob = case["baseline_prob"]
        mprob = case["mamba_prob"]

        # Column 1: Baseline original
        axes[row][0].imshow(frame, cmap="gray")
        axes[row][0].set_title(f"Baseline UNet3D (prob={bprob:.3f})")
        axes[row][0].axis("off")

        # Column 2: Baseline Grad-CAM
        axes[row][1].imshow(frame, cmap="gray")
        axes[row][1].imshow(bl_hm, cmap="jet", alpha=0.5)
        axes[row][1].set_title(f"Baseline Grad-CAM\n(label={true_label}, pred={case['baseline_pred']})")
        axes[row][1].axis("off")

        # Column 3: Mamba original
        axes[row][2].imshow(frame, cmap="gray")
        axes[row][2].set_title(f"Mamba UNet (prob={mprob:.3f})")
        axes[row][2].axis("off")

        # Column 4: Mamba Grad-CAM
        axes[row][3].imshow(frame, cmap="gray")
        axes[row][3].imshow(mb_hm, cmap="jet", alpha=0.5)
        axes[row][3].set_title(f"Mamba Grad-CAM\n(label={true_label}, pred={case['mamba_pred']})")
        axes[row][3].axis("off")

    plt.tight_layout()
    fig.savefig(PLOTS / "gradcam_success_cases.png")
    plt.close(fig)
    print("gradcam_success_cases.png saved")


# ===================================================================
if __name__ == "__main__":
    print("Generating Mamba ablation visualizations...")
    try:
        plot_comparison()
    except Exception as e:
        print(f"Comparison plot failed: {e}")
    try:
        plot_gradcam_success_cases()
    except Exception as e:
        print(f"Grad-CAM plot failed: {e}")
    print(f"Plots saved to: {PLOTS}")
