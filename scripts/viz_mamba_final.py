"""
Mamba final visualizations: convergence curve + baseline comparison
"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = PROJECT / "results" / "mamba_ablation"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

# ======== 1. Convergence curve ========
df = pd.read_csv(PLOTS / "mamba_training_log.csv")

fig, ax1 = plt.subplots(figsize=(11, 5))
color1, color2 = "#e74c3c", "#3498db"

# Loss on left y-axis
ax1.plot(df["epoch"], df["train_loss"], "o-", color=color1, linewidth=2, markersize=8,
         markerfacecolor="white", label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

# F1 on right y-axis
ax2 = ax1.twinx()
ax2.plot(df["epoch"], df["f1"], "s--", color=color2, linewidth=2, markersize=8,
         markerfacecolor="white", label="Val F1 Score")

# Baseline F1 reference
baseline_f1 = 0.8675
ax2.axhline(y=baseline_f1, color="#2ecc71", linestyle=":", linewidth=2, alpha=0.8,
            label=f"Baseline F1 = {baseline_f1:.4f}")

# Best F1 marker
best_idx = df["f1"].idxmax()
best_epoch = df.loc[best_idx, "epoch"]
best_f1 = df.loc[best_idx, "f1"]
ax2.annotate(f"Best: epoch {int(best_epoch)}\nF1={best_f1:.4f}",
             xy=(best_epoch, best_f1), xytext=(best_epoch + 0.8, best_f1 + 0.02),
             arrowprops=dict(arrowstyle="->", color="#2c3e50"),
             fontsize=10, fontweight="bold", color="#2c3e50")

ax2.set_ylabel("F1 Score", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0.0, 1.0)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

ax1.set_xticks(df["epoch"].astype(int))
ax1.set_title("Hybrid UNet-Mamba Training Convergence\n(macula_detached_vs_intact, 360 train / 101 test)")
ax1.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(PLOTS / "mamba_convergence.png")
plt.close(fig)
print("Convergence curve saved")

# ======== 2. Baseline vs Mamba comparison bar chart ========
comp = pd.read_csv(PLOTS / "results_comparison.csv")

metrics = ["accuracy", "sensitivity", "specificity", "f1"]
labels = ["Accuracy", "Sensitivity", "Specificity", "F1 Score"]
bl_vals = [comp[comp["group"] == "baseline_unet3d"][m].values[0] for m in metrics]
mb_vals = [comp[comp["group"] == "mamba_finetuned"][m].values[0] for m in metrics]

x = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, bl_vals, width, label="Baseline UNet3D",
               color="#3498db", edgecolor="white", linewidth=1.2)
bars2 = ax.bar(x + width/2, mb_vals, width, label="Hybrid UNet-Mamba",
               color="#e74c3c", edgecolor="white", linewidth=1.2)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

for i, (bl, mb) in enumerate(zip(bl_vals, mb_vals)):
    delta = mb - bl
    color = "#27ae60" if delta >= 0 else "#e74c3c"
    sign = "+" if delta >= 0 else ""
    ax.annotate(f"{sign}{delta:.4f}", (x[i], max(bl, mb) + 0.04),
                ha="center", fontsize=11, color=color, fontweight="bold")

ax.set_ylabel("Score")
ax.set_title("Mamba Ablation: Baseline UNet3D vs Hybrid UNet-Mamba\n"
             "(UNet3D: 25M params | Hybrid: 38M params, Mamba bottleneck at 768-dim)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.82, 0.95)
ax.legend(loc="lower left", framealpha=0.9)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig.savefig(PLOTS / "mamba_vs_baseline.png")
plt.close(fig)
print("Baseline comparison saved")
print("DONE")
