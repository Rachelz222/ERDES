"""
ERDES Experiment Visualizations
=================================
6 plots from 4 experiments, saved to results/plots/
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PLOTS = PROJECT_ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "non_rd": "#2ecc71", "rd": "#e74c3c",
    "normal": "#3498db", "pvd": "#f39c12",
    "macula_intact": "#1abc9c", "macula_detached": "#e74c3c",
    "stage1": "#3498db", "stage2": "#e74c3c",
}


# ===================================================================
# 1b: Motion Score by 4 Subtypes (Box Plot)
# ===================================================================
def plot_1b():
    df = pd.read_csv(PROJECT_ROOT / "results" / "stats" / "motion_scores.csv")
    # Map subtypes to readable names
    subtype_map = {
        "normal": "Normal", "pvd": "PVD",
        "macula_intact": "Macula\nIntact", "macula_detached": "Macula\nDetached",
    }
    df["subtype_label"] = df["subtype"].map(subtype_map).fillna(df["subtype"])
    order = ["Normal", "PVD", "Macula\nIntact", "Macula\nDetached"]
    data = [df[df["subtype_label"] == s]["motion_score"].values for s in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.3})
    palette = ["#3498db", "#f39c12", "#1abc9c", "#e74c3c"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers and values
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.plot(i + 1, mean_val, "D", color="black", markersize=8, zorder=5)
        ax.annotate(f"{mean_val:.4f}", (i + 1.25, mean_val), fontsize=9, va="center")

    ax.set_ylabel("Motion Score (mean |frame diff|)")
    ax.set_title("Motion Score by Diagnostic Subtype\n(Higher = more inter-frame motion)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / "1b_motion_by_subtype.png")
    plt.close(fig)
    print("1b: Motion by subtype → done")


# ===================================================================
# 1c: Brightness × Contrast × Motion 3D Scatter
# ===================================================================
def plot_1c():
    df = pd.read_csv(PROJECT_ROOT / "results" / "stats" / "motion_scores.csv")
    # Downsample for readability (max 2000 points)
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: brightness vs contrast, colored by diagnostic_class
    for cls, color, label in [("non_rd", "#2ecc71", "Non-RD"), ("rd", "#e74c3c", "RD")]:
        subset = df[df["diagnostic_class"] == cls]
        axes[0].scatter(subset["brightness"], subset["contrast"],
                        c=color, label=label, alpha=0.4, s=8, edgecolors="none")
    axes[0].set_xlabel("Brightness (mean intensity)")
    axes[0].set_ylabel("Contrast (std intensity)")
    axes[0].set_title("Brightness vs Contrast by Class")
    axes[0].legend(markerscale=3)
    axes[0].grid(alpha=0.2)

    # Panel 2: contrast vs motion, colored by diagnostic_class
    for cls, color, label in [("non_rd", "#2ecc71", "Non-RD"), ("rd", "#e74c3c", "RD")]:
        subset = df[df["diagnostic_class"] == cls]
        axes[1].scatter(subset["contrast"], subset["motion_score"],
                        c=color, label=label, alpha=0.4, s=8, edgecolors="none")
    axes[1].set_xlabel("Contrast (std intensity)")
    axes[1].set_ylabel("Motion Score")
    axes[1].set_title("Contrast vs Motion Score by Class")
    axes[1].legend(markerscale=3)
    axes[1].grid(alpha=0.2)

    plt.suptitle("Video-Level Features: RD vs Non-RD Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS / "1c_feature_scatter.png")
    plt.close(fig)
    print("1c: Feature scatter → done")


# ===================================================================
# 2a: Sampling Ratio Ablation (Multi-Metric Line Chart)
# ===================================================================
def plot_2a():
    df = pd.read_csv(PROJECT_ROOT / "results" / "exp" / "sampling_ablation_results.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    ratios = df["ratio"].values

    metrics_config = [
        ("accuracy",    "Accuracy",    "#2ecc71", "o-"),
        ("sensitivity", "Sensitivity", "#3498db", "s--"),
        ("specificity", "Specificity", "#e74c3c", "^-."),
        ("f1",          "F1 Score",    "#9b59b6", "D:"),
    ]
    for col, label, color, style in metrics_config:
        ax.plot(ratios, df[col].values, style, color=color, label=label,
                linewidth=2, markersize=10, markerfacecolor="white")

    ax.set_xlabel("Sampling Ratio r")
    ax.set_ylabel("Score")
    ax.set_title("Sampling Ratio Ablation on Macula Status Classification\n(UNet3D, Stage 2)")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:.1f}" for r in ratios])
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / "2a_sampling_ablation.png")
    plt.close(fig)
    print("2a: Sampling ablation → done")


# ===================================================================
# 3a: Pipeline Sankey / Flow Diagram
# ===================================================================
def plot_3a():
    """Manual Sankey-style flow diagram for the two-stage pipeline."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Metrics from pipeline eval (latest run)
    s1_tp, s1_tn, s1_fp, s1_fn = 92, 966, 11, 8   # Stage 1 (n=1077)
    s2_tp, s2_tn, s2_fp, s2_fn = 36, 54, 7, 4      # Stage 2 (n=101)
    s1_n = s1_tp + s1_tn + s1_fp + s1_fn
    s2_n = s2_tp + s2_tn + s2_fp + s2_fn

    def draw_box(x, y, w, h, color, text, subtext=""):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y + 0.05, text, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        if subtext:
            ax.text(x, y - 0.35, subtext, ha="center", va="center", fontsize=8, color="white", alpha=0.9)

    def draw_arrow(x1, y1, x2, y2, color, width, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                    connectionstyle="arc3,rad=0"))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.15, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    # ---- Stage 1 boxes ----
    draw_box(2, 7.5, 2.5, 1.2, "#2ecc71", f"Non-RD (True Negative)", f"n={s1_tn}")
    draw_box(2, 5.5, 2.5, 1.2, "#e74c3c", f"RD (True Positive)", f"n={s1_tp}")
    draw_box(6, 8.5, 2.0, 0.9, "#e67e22", f"Non-RD → RD (FP)", f"n={s1_fp}")
    draw_box(6, 4.5, 2.0, 0.9, "#c0392b", f"RD → Non-RD (FN)", f"n={s1_fn}")

    # Stage 1 title
    ax.text(3.5, 9.5, "STAGE 1: RD Detection (n=1077)", ha="center", fontsize=14, fontweight="bold", color="#2c3e50")
    ax.text(3.5, 9.1, f"Sensitivity={s1_tp/(s1_tp+s1_fn):.3f}  Specificity={s1_tn/(s1_tn+s1_fp):.3f}",
            ha="center", fontsize=10, color="#7f8c8d")

    # Arrows from correct predictions
    draw_arrow(2, 7.0, 7, 3.8, "#95a5a6", 1.5, "Stop (correct)")
    draw_arrow(2, 6.0, 8, 3.8, "#95a5a6", 1.5, "")

    # ---- Stage 2 boxes ----
    draw_box(8, 5.5, 2.5, 1.2, "#1abc9c", f"Macula Intact (TN)", f"n={s2_tn}")
    draw_box(8, 2.5, 2.5, 1.2, "#e74c3c", f"Macula Detached (TP)", f"n={s2_tp}")
    draw_box(8, 1.0, 2.0, 0.8, "#e67e22", f"FP", f"n={s2_fp}")
    draw_box(8, 0.0, 2.0, 0.8, "#c0392b", f"FN", f"n={s2_fn}")

    ax.text(8, 7.0, "STAGE 2: Macula Status (n=101)", ha="center", fontsize=13, fontweight="bold", color="#2c3e50")
    ax.text(8, 6.6, f"Sensitivity={s2_tp/(s2_tp+s2_fn):.3f}  Specificity={s2_tn/(s2_tn+s2_fp):.3f}",
            ha="center", fontsize=9, color="#7f8c8d")

    # Combined sensitivity
    combined = (s1_tp/(s1_tp+s1_fn)) * (s2_tp/(s2_tp+s2_fn))
    ax.text(5, 0.5, f"Combined Sensitivity = {s1_tp/(s1_tp+s1_fn):.4f} × {s2_tp/(s2_tp+s2_fn):.4f} = {combined:.4f}",
            ha="center", fontsize=14, fontweight="bold", color="#c0392b",
            bbox=dict(boxstyle="round", facecolor="#fdf2f2", edgecolor="#c0392b", alpha=0.9))

    ax.set_title("Two-Stage Diagnostic Pipeline Flow", fontsize=16, fontweight="bold", pad=25)
    plt.tight_layout()
    fig.savefig(PLOTS / "3a_pipeline_flow.png")
    plt.close(fig)
    print("3a: Pipeline flow → done")


# ===================================================================
# 4a & 4b: Grad-CAM Heatmaps
# ===================================================================
def plot_gradcam():
    """Generate Grad-CAM visualizations for 4 subtype samples."""
    import torch
    import decord
    from safetensors.torch import load_file
    from erdes.models.components.cls_model import Unet3DClassifier
    from erdes.models.explainer import GradCAM3D
    from erdes.data.components.utils import resize

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Grad-CAM using: {device}")

    # Load model
    ckpt = str(PROJECT_ROOT / "weights" / "unet3d_macula_detached_vs_intact.safetensors")
    model = Unet3DClassifier(in_channels=1, num_classes=1).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)
    print("  Model loaded")

    resize_tf = resize((96, 128, 128))
    data_root = PROJECT_ROOT / "data" / "erdes"

    # Pick 1 representative sample per subtype
    samples = [
        ("Normal",        "Non_Retinal_Detachment/Normal/164267_00013.mp4", 0),
        ("PVD",           "Non_Retinal_Detachment/Posterior_Vitreous_Detachment/825315_00047.mp4", 0),
        ("Macula Intact", "Retinal_Detachment/Macula_Intact/TD/102769_00001.mp4", 0),
        ("Macula Detached","Retinal_Detachment/Macula_Detached/TD/405744_00005.mp4", 1),
    ]

    def load_video(rel_path):
        abs_path = str(data_root / rel_path)
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

    # ---- 4a: Grid of 4 subtypes ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Grad-CAM Heatmaps by Diagnostic Subtype (UNet3D, Stage 2)", fontsize=16, fontweight="bold")

    for idx, (name, path, label) in enumerate(samples):
        row = idx // 2
        col_base = (idx % 2) * 2

        video = load_video(path)
        if video is None:
            for c in range(2):
                axes[row][col_base + c].text(0.5, 0.5, "Video unavailable", ha="center", va="center")
                axes[row][col_base + c].set_title(f"{name} - N/A")
            continue

        video = video.to(device)
        D_mid = video.shape[2] // 2

        # Original frame
        frame = video[0, 0, D_mid].cpu().numpy()
        axes[row][col_base].imshow(frame, cmap="gray")
        axes[row][col_base].set_title(f"{name} — Original (frame {D_mid})")
        axes[row][col_base].axis("off")

        # Grad-CAM
        explainer = GradCAM3D(model)
        try:
            with torch.enable_grad():
                heatmap = explainer(video)
            axes[row][col_base + 1].imshow(frame, cmap="gray")
            axes[row][col_base + 1].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[row][col_base + 1].set_title(f"{name} — Grad-CAM Overlay")
        except Exception as e:
            axes[row][col_base + 1].text(0.5, 0.5, f"Error: {e}", ha="center", va="center", fontsize=8)
            axes[row][col_base + 1].set_title(f"{name} — Failed")
        axes[row][col_base + 1].axis("off")
        explainer.remove_hooks()

    plt.tight_layout()
    fig.savefig(PLOTS / "4a_gradcam_subtypes.png")
    plt.close(fig)
    print("4a: Grad-CAM subtypes → done")

    # ---- 4b: Correct vs Incorrect comparison ----
    # We need a sample that the model got WRONG.
    # From failure_cases.log: 405744_00015.mp4 was FN (label=1, pred=0)
    # We need a CORRECT one too. Let's use a clear TP case.
    wrong_sample = ("FN: Detached→Intact", "Retinal_Detachment/Macula_Detached/TD/405744_00015.mp4", 1)
    correct_sample = ("TP: Detached (correct)", "Retinal_Detachment/Macula_Detached/Bilateral/755384_00051.mp4", 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Grad-CAM: Correct vs Incorrect Prediction Comparison", fontsize=15, fontweight="bold")

    for idx, (name, path, label) in enumerate([correct_sample, wrong_sample]):
        col_base = idx * 2
        video = load_video(path)
        if video is None:
            continue
        video = video.to(device)
        D_mid = video.shape[2] // 2
        frame = video[0, 0, D_mid].cpu().numpy()

        axes[col_base].imshow(frame, cmap="gray")
        axes[col_base].set_title(f"{name}\nOriginal Frame {D_mid}")
        axes[col_base].axis("off")

        explainer = GradCAM3D(model)
        try:
            with torch.enable_grad():
                heatmap = explainer(video)
            axes[col_base + 1].imshow(frame, cmap="gray")
            axes[col_base + 1].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[col_base + 1].set_title(f"{name}\nGrad-CAM Overlay")
        except:
            axes[col_base + 1].text(0.5, 0.5, "Error", ha="center", va="center")
        axes[col_base + 1].axis("off")
        explainer.remove_hooks()

    plt.tight_layout()
    fig.savefig(PLOTS / "4b_gradcam_correct_vs_wrong.png")
    plt.close(fig)
    print("4b: Grad-CAM correct vs wrong → done")


# ===================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("ERDES Experiment Visualizations")
    print("=" * 50)
    plot_1b()
    plot_1c()
    plot_2a()
    plot_3a()
    plot_gradcam()
    print(f"\nAll plots saved to: {PLOTS}")
