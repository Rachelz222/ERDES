"""
ERDES dataset class statistics and visualizations.

Produces:
    - Console table of global and per-task split distributions
    - Figure 1 (pie): subtype distribution (Normal/PVD/Macula Detached/Macula Intact)
    - Figure 2 (bars): per-task train/val/test class counts

Usage:  python scripts/dataset_statistics.py
Output: scripts/dataset_stats_overview.png
"""
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_CSV = os.path.join(ROOT, "data", "erdes", "erdes_metadata.csv")
SPLITS_DIR = os.path.join(ROOT, "data", "splits")
OUT_PNG = os.path.join(ROOT, "scripts", "dataset_stats_overview.png")

TASK_NAMES = [
    "non_rd_vs_rd",
    "normal_vs_rd",
    "pvd_vs_rd",
    "macula_detached_vs_intact",
    "normal_vs_pvd",
]

TASK_LABELS = {
    "non_rd_vs_rd": {0: "Non-RD (Normal+PVD)", 1: "RD"},
    "normal_vs_rd": {0: "Normal", 1: "RD"},
    "pvd_vs_rd": {0: "PVD", 1: "RD"},
    "macula_detached_vs_intact": {0: "Macula Detached", 1: "Macula Intact"},
    "normal_vs_pvd": {0: "Normal", 1: "PVD"},
}

SUBTYPE_NAMES = {
    "normal": "Normal",
    "pvd": "PVD",
    "macula_detached": "Macula Detached",
    "macula_intact": "Macula Intact",
}

SUBTYPE_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22"]


def print_global_stats(df):
    """Print global dataset statistics from metadata."""
    total = len(df)
    patients = df["clip_id"].str.extract(r"^(\d+)_")[0].nunique()

    print("=" * 60)
    print(f"  ERDES Dataset Overview")
    print("=" * 60)
    print(f"  Total clips:  {total}")
    print(f"  Total patients: {patients}")
    print()

    # By diagnostic_class (rd vs non_rd)
    rd_counts = df["diagnostic_class"].value_counts()
    print(f"  RD vs Non-RD:")
    print(f"    Non-RD: {rd_counts.get('non_rd', 0):5d}  ({rd_counts.get('non_rd', 0)/total*100:5.1f}%)")
    print(f"    RD:     {rd_counts.get('rd', 0):5d}  ({rd_counts.get('rd', 0)/total*100:5.1f}%)")
    print()

    # By subtype
    print(f"  By subtype:")
    subtype_counts = df["subtype"].value_counts()
    for st in ["normal", "pvd", "macula_detached", "macula_intact"]:
        cnt = subtype_counts.get(st, 0)
        name = SUBTYPE_NAMES.get(st, st)
        print(f"    {name:<20s}: {cnt:5d}  ({cnt/total*100:5.1f}%)")
    print()

    # By anatomical subclass (within RD)
    print(f"  RD anatomical subclass breakdown:")
    rd_df = df[df["diagnostic_class"] == "rd"]
    anat_counts = rd_df["anatomical_subclass"].value_counts()
    for ak in sorted(anat_counts.index):
        cnt = anat_counts[ak]
        print(f"    {ak:<6s}: {cnt:4d}  ({cnt/len(rd_df)*100:5.1f}% of RD)")
    print()

    # Per patient
    print(f"  Per patient:")
    df["patient_id"] = df["clip_id"].str.extract(r"^(\d+)_")[0]
    patient_clips = df.groupby("patient_id").size()
    for pid, cnt in patient_clips.items():
        subtypes_in = df[df["patient_id"] == pid]["subtype"].value_counts()
        subtype_str = ", ".join(f"{SUBTYPE_NAMES.get(k, k)}:{v}" for k, v in subtypes_in.items())
        print(f"    Patient {pid}: {cnt:4d} clips  ({subtype_str})")
    print()


def print_split_stats():
    """Print per-task train/val/test class distribution."""
    print("=" * 60)
    print(f"  Per-Task Split Distributions")
    print("=" * 60)

    summary_rows = []

    for task in TASK_NAMES:
        task_dir = os.path.join(SPLITS_DIR, task)
        print(f"\n  [{task}]")
        print(f"    {'Split':<6s} {'Count 0':>8s} {'Count 1':>8s} {'Total':>7s} {'Ratio 0':>8s} {'Ratio 1':>8s}")

        for split in ["train", "val", "test"]:
            path = os.path.join(task_dir, f"{split}.csv")
            df = pd.read_csv(path)
            cnt0 = int((df["label"] == 0).sum())
            cnt1 = int((df["label"] == 1).sum())
            total = cnt0 + cnt1
            r0 = cnt0 / total * 100
            r1 = cnt1 / total * 100
            print(f"    {split:<6s} {cnt0:8d} {cnt1:8d} {total:7d} {r0:7.1f}% {r1:7.1f}%")
            summary_rows.append({
                "task": task, "split": split,
                "cnt0": cnt0, "cnt1": cnt1, "total": total,
            })

    print()
    return summary_rows


def plot_figure1_pie(df):
    """Figure 1: Donut pie chart of 4 subtypes with RD/Non-RD inner ring."""
    subtype_counts = df["subtype"].value_counts()
    order = ["normal", "pvd", "macula_detached", "macula_intact"]
    sizes = [subtype_counts.get(k, 0) for k in order]
    labels = [f"{SUBTYPE_NAMES[k]}\n{n}" for k, n in zip(order, sizes)]
    colors = SUBTYPE_COLORS

    # RD vs Non-RD for inner ring
    rd_count = subtype_counts.get("macula_detached", 0) + subtype_counts.get("macula_intact", 0)
    non_rd_count = subtype_counts.get("normal", 0) + subtype_counts.get("pvd", 0)
    inner_sizes = [non_rd_count, rd_count]
    inner_colors = ["#d5f5e3", "#fadbd8"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Outer ring: subtypes
    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12, fontweight="bold"),
    )

    # Inner ring: RD vs Non-RD
    ax.pie(
        inner_sizes, colors=inner_colors,
        startangle=90,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.5),
        radius=0.72,
    )

    # Percent labels on outer ring
    total = sum(sizes)
    for i, (w, s) in enumerate(zip(wedges, sizes)):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.15 * np.cos(np.deg2rad(ang))
        y = 1.15 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{s/total*100:.1f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color=colors[i])

    # Center text
    ax.text(0, 0.06, f"Total\n{total}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.12, "clips", ha="center", va="center", fontsize=10, color="gray")

    ax.set_title("ERDES Dataset: Subtype Distribution", fontsize=16, fontweight="bold", pad=25)
    plt.tight_layout()
    return fig


def plot_figure2_bars(summary_rows):
    """Figure 2: Grouped bar chart of per-task train/val/test class counts."""
    fig, ax = plt.subplots(figsize=(16, 7))

    n_tasks = len(TASK_NAMES)
    n_splits = 3
    n_classes = 2
    bar_width = 0.12
    group_width = bar_width * n_classes * n_splits + bar_width * 1.2
    split_names = ["train", "val", "test"]
    class_colors = ["#3498db", "#e74c3c"]
    hatch_styles = ["", "//"]

    x_positions = np.arange(n_tasks) * (group_width + 0.25)

    for ti, task in enumerate(TASK_NAMES):
        for si, split in enumerate(split_names):
            rows = [r for r in summary_rows if r["task"] == task and r["split"] == split]
            if not rows:
                continue
            r = rows[0]
            counts = [r["cnt0"], r["cnt1"]]

            for ci in range(n_classes):
                offset = (si * n_classes + ci) * bar_width + si * bar_width * 0.3
                x = x_positions[ti] + offset
                bar = ax.bar(x, counts[ci], bar_width,
                             color=class_colors[ci],
                             edgecolor="white", linewidth=0.5,
                             hatch=hatch_styles[ci] if si == 1 else "",
                             alpha=0.85 if si != 0 else 1.0)

                # Annotate count
                if counts[ci] > 0:
                    ax.text(x, counts[ci] + max(1, r["total"] * 0.015),
                            str(counts[ci]), ha="center", va="bottom",
                            fontsize=7, fontweight="bold", rotation=90)

    # X-axis labels
    xtick_positions = x_positions + (n_splits * n_classes * bar_width + n_splits * bar_width * 0.3) / 2 - bar_width / 2
    ax.set_xticks(xtick_positions)
    task_display = [t.replace("_", "\n") for t in TASK_NAMES]
    ax.set_xticklabels(task_display, fontsize=9, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for ci in range(n_classes):
        legend_elements.append(Patch(facecolor=class_colors[ci], label=f"Class {ci}"))
    for si, sn in enumerate(split_names):
        legend_elements.append(Patch(facecolor="white", edgecolor="black",
                                     hatch=hatch_styles[si] if si == 1 else "",
                                     alpha=0.85 if si != 0 else 1.0,
                                     label=f"  [{sn}]"))
    ax.legend(handles=legend_elements, loc="upper right", ncols=3, fontsize=8,
              title="Class / Split", title_fontsize=9)

    # Split boundary lines between task groups
    for i in range(n_tasks - 1):
        boundary = (x_positions[i] + x_positions[i + 1]) / 2
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_ylabel("Number of Clips", fontsize=12)
    ax.set_title("ERDES Dataset: Per-Task Train/Val/Test Class Distribution", fontsize=16, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


OUT_PIE = os.path.join(ROOT, "scripts", "dataset_stats_pie.png")
OUT_SPLITS = os.path.join(ROOT, "scripts", "dataset_stats_splits.png")


def plot_and_save_pie(df):
    """Figure 1: Donut pie of subtype distribution, saved separately."""
    fig, ax = plt.subplots(figsize=(9, 7))

    subtype_counts = df["subtype"].value_counts()
    order = ["normal", "pvd", "macula_detached", "macula_intact"]
    sizes = [subtype_counts.get(k, 0) for k in order]
    pie_colors = SUBTYPE_COLORS
    total = sum(sizes)

    # Legend labels with name, count, percentage
    legend_labels = [
        f"{SUBTYPE_NAMES[k]}: {n} ({n/total*100:.1f}%)"
        for k, n in zip(order, sizes)
    ]

    # Outer ring: no built-in labels, use autopct on wedges
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=pie_colors,
        startangle=90, pctdistance=0.78,
        autopct=lambda pct: f"{pct:.1f}%",
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11, fontweight="bold", color="white"),
    )

    # Inner ring: RD vs Non-RD
    rd_count = subtype_counts.get("macula_detached", 0) + subtype_counts.get("macula_intact", 0)
    non_rd_count = subtype_counts.get("normal", 0) + subtype_counts.get("pvd", 0)
    inner_wedges, _ = ax.pie(
        [non_rd_count, rd_count], colors=["#d5f5e3", "#fadbd8"],
        startangle=90,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.5),
        radius=0.72)

    # Center text
    ax.text(0, 0.06, f"Total\n{total}", ha="center", va="center",
            fontsize=15, fontweight="bold")
    ax.text(0, -0.13, "clips", ha="center", va="center", fontsize=9, color="gray")

    # Legend: outer ring wedges + inner ring wedges
    inner_legend = [
        f"Non-RD (Normal+PVD): {non_rd_count} ({non_rd_count/total*100:.1f}%)",
        f"RD (Macula Detached+Intact): {rd_count} ({rd_count/total*100:.1f}%)",
    ]
    all_wedges = list(wedges) + [inner_wedges[0], inner_wedges[1]]
    all_labels = legend_labels + inner_legend
    ax.legend(
        all_wedges, all_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=10,
        title="Categories",
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    ax.set_title("ERDES Dataset: Subtype Distribution", fontsize=15, fontweight="bold", pad=22)

    fig.tight_layout()
    fig.savefig(OUT_PIE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PIE}")


def plot_and_save_splits(summary_rows):
    """Figure 2: Per-task horizontal grouped bars — one subplot per task."""
    from matplotlib.patches import Patch

    class_colors = ["#3498db", "#e74c3c"]
    class_labels = ["Class 0", "Class 1"]
    split_names = ["train", "val", "test"]
    n_tasks = len(TASK_NAMES)

    fig, axes = plt.subplots(1, n_tasks, figsize=(26, 6.5), sharex=False)
    fig.subplots_adjust(wspace=0.35)

    global_max = 0
    for task in TASK_NAMES:
        for split in split_names:
            rows = [r for r in summary_rows if r["task"] == task and r["split"] == split]
            if rows:
                global_max = max(global_max, max(rows[0]["cnt0"], rows[0]["cnt1"]))

    for ti, (task, ax) in enumerate(zip(TASK_NAMES, axes)):
        # Build data for this task
        data = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
        for split in split_names:
            rows = [r for r in summary_rows if r["task"] == task and r["split"] == split]
            if rows:
                data[split] = [rows[0]["cnt0"], rows[0]["cnt1"]]

        y_labels = []
        for split in split_names:
            for ci in range(2):
                y_labels.append(f"{split}\n({class_labels[ci]})")

        y_pos = np.arange(6)
        bar_height = 0.35
        # layout: train0, train1, val0, val1, test0, test1
        values = []
        for split in split_names:
            values.extend(data[split])

        bars = ax.barh(y_pos, values, height=bar_height,
                       color=[class_colors[0], class_colors[1]] * 3,
                       edgecolor="white", linewidth=0.5)

        # Annotations — place after each bar, with intelligent offset
        for j, (v, b) in enumerate(zip(values, bars)):
            offset = max(global_max * 0.01, 5)
            ax.text(v + offset, b.get_y() + b.get_height() / 2,
                    str(v), va="center", ha="left",
                    fontsize=8.5, fontweight="bold",
                    color=class_colors[j % 2])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7.5)
        ax.set_xlim(0, global_max * 1.22)
        title = task.replace("_", "\n")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("Clips", fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Draw horizontal separators between splits
        for sep_y in [1.5, 3.5]:
            ax.axhline(sep_y, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    # Shared legend
    legend_elements = [
        Patch(facecolor=class_colors[0], label="Class 0"),
        Patch(facecolor=class_colors[1], label="Class 1"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncols=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("ERDES Dataset: Per-Task Train/Val/Test Class Distribution",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_SPLITS, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_SPLITS}")


def main():
    # Load metadata
    df = pd.read_csv(METADATA_CSV)

    # Console stats
    print_global_stats(df)
    summary_rows = print_split_stats()

    # Save two separate figures
    plot_and_save_pie(df)
    plot_and_save_splits(summary_rows)


if __name__ == "__main__":
    main()
