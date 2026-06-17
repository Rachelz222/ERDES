"""
Run eval on downloaded checkpoints and compare with paper benchmarks.
"""
import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent

# Paper benchmarks (from Tables 8, 9, 10)
PAPER = {
    ("non_rd_vs_rd", "resnet3d"): {"acc": 0.974, "sens": 0.939, "spec": 0.978, "prec": 0.817, "f1": 0.874},
    ("non_rd_vs_rd", "unet3d"):   {"acc": 0.982, "sens": 0.920, "spec": 0.988, "prec": 0.893, "f1": 0.906},
    ("macula_detached_vs_intact", "unet3d"): {"acc": 0.882, "sens": 0.899, "spec": 0.870, "prec": 0.818, "f1": 0.857},
}

# Checkpoints to evaluate
CHECKPOINTS = [
    {"task": "non_rd_vs_rd", "model": "resnet3d", "ckpt": "weights/resnet3d_non_rd_vs_rd_best.ckpt"},
    {"task": "non_rd_vs_rd", "model": "unet3d", "ckpt": "weights/unet3d_non_rd_vs_rd_best.ckpt"},
    {"task": "macula_detached_vs_intact", "model": "unet3d", "ckpt": "weights/unet3d_macula_detached_vs_intact_best.ckpt"},
]


def parse_metrics(output: str) -> dict:
    """Parse test metrics from eval.py output."""
    metrics = {}
    # Lightning test output looks like: "test/acc": 0.982, ...
    patterns = {
        "acc": r"test/acc[^=]*?([\d.]+)",
        "prec": r"test/precision[^=]*?([\d.]+)",
        "sens": r"test/sensitivity[^=]*?([\d.]+)",
        "spec": r"test/specificity[^=]*?([\d.]+)",
        "f1": r"test/f1[^=]*?([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def run_eval(task, model, ckpt_path):
    """Run eval.py for a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {task} / {model}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    cmd = (
        f"python erdes/eval.py "
        f"experiment={task}/{model} "
        f"trainer=gpu "
        f"ckpt_path={ckpt_path} "
        f"data.batch_size=4 "
        f"data.num_workers=2"
    )
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace"
        )
        combined = result.stdout + "\n" + result.stderr
        # Print last 40 lines
        lines = combined.strip().split("\n")
        for line in lines[-40:]:
            print(f"  | {line}")
        return result.returncode == 0, combined
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        return False, ""
    except Exception as e:
        print(f"  [ERROR: {e}]")
        return False, ""


def main():
    print("=" * 70)
    print("ERDES Paper Benchmark Comparison")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}

    for cfg in CHECKPOINTS:
        ckpt_path = PROJECT_ROOT / cfg["ckpt"]
        if not ckpt_path.exists():
            print(f"\n[Skipping] {cfg['ckpt']} not found")
            continue

        print(f"\nCheckpoint size: {ckpt_path.stat().st_size / 1024**2:.1f} MB")
        success, output = run_eval(cfg["task"], cfg["model"], str(ckpt_path))
        metrics = parse_metrics(output)
        results[(cfg["task"], cfg["model"])] = {"success": success, "metrics": metrics}

    # Print comparison table
    print("\n\n")
    print("=" * 90)
    print("PAPER BENCHMARK VS OUR EVALUATION")
    print("=" * 90)
    header = f"{'Task':<30} {'Model':<12} {'Metric':<6} {'Paper':>7} {'Ours':>7} {'Delta':>7} {'Match':>6}"
    print(header)
    print("-" * 90)

    for (task, model), paper_metrics in PAPER.items():
        our = results.get((task, model), {})
        our_metrics = our.get("metrics", {})
        for metric in ["acc", "sens", "spec", "prec", "f1"]:
            paper_val = paper_metrics.get(metric, 0)
            our_val = our_metrics.get(metric, None)
            if our_val is not None:
                delta = our_val - paper_val
                match_flag = "OK" if abs(delta) < 0.02 else ("HIGH" if delta > 0 else "LOW")
                print(f"{task:<30} {model:<12} {metric:<6} {paper_val:>7.4f} {our_val:>7.4f} {delta:>+7.4f} {match_flag:>6}")
            else:
                print(f"{task:<30} {model:<12} {metric:<6} {paper_val:>7.4f} {'N/A':>7} {'-':>7} {'-':>6}")

    print("-" * 90)
    print("\nNote: Delta within ±0.02 is considered matching (accounting for hardware/runtime variance).")

    return results


if __name__ == "__main__":
    main()
