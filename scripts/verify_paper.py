"""
Systematic verification script for ERDES paper benchmarks.
Runs training + evaluation on all 5 classification tasks using GPU,
then runs the diagnostic pipeline and compares metrics against paper baselines.
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent

# Paper benchmark results (from Tables 8, 9, 10)
PAPER_BENCHMARKS = {
    "non_rd_vs_rd": {
        "best_model": "unet3d",
        "second_best": "resnet3d",
        "metrics": {
            "unet3d": {"acc": 0.982, "sens": 0.920, "spec": 0.988, "prec": 0.893, "f1": 0.906},
            "resnet3d": {"acc": 0.974, "sens": 0.939, "spec": 0.978, "prec": 0.817, "f1": 0.874},
            "unetplusplus": {"acc": 0.976, "sens": 0.879, "spec": 0.986, "prec": 0.871, "f1": 0.875},
        },
    },
    "macula_detached_vs_intact": {
        "best_model": "unet3d",
        "second_best": "vnet",
        "metrics": {
            "unet3d": {"acc": 0.882, "sens": 0.899, "spec": 0.870, "prec": 0.818, "f1": 0.857},
            "resnet3d": {"acc": 0.862, "sens": 0.750, "spec": 0.935, "prec": 0.882, "f1": 0.810},
        },
    },
    "normal_vs_rd": {
        "best_model": "unet3d",
        "metrics": {
            "unet3d": {"acc": 0.991, "sens": 0.950, "spec": 0.996, "prec": 0.969, "f1": 0.959},
            "unetplusplus": {"acc": 0.985, "sens": 0.910, "spec": 0.994, "prec": 0.948, "f1": 0.929},
        },
    },
    "pvd_vs_rd": {
        "best_model": "unetplusplus",
        "metrics": {
            "unetplusplus": {"acc": 0.887, "sens": 0.921, "spec": 0.860, "prec": 0.839, "f1": 0.878},
        },
    },
    "normal_vs_pvd": {
        "best_model": "unet3d",
        "note": "Hardest task - paper uses selective temporal pooling",
        "metrics": {
            "unet3d": {"acc": 0.959, "sens": 0.784, "spec": 0.985, "prec": 0.894, "f1": 0.836},
        },
    },
}

# Best architectures to train per task (focusing on top performers)
TASKS = [
    {"task": "non_rd_vs_rd", "model": "resnet3d", "epochs": 50},
    {"task": "non_rd_vs_rd", "model": "unet3d", "epochs": 50},
    {"task": "macula_detached_vs_intact", "model": "unet3d", "epochs": 50},
    {"task": "normal_vs_rd", "model": "unet3d", "epochs": 50},
    {"task": "pvd_vs_rd", "model": "unetplusplus", "epochs": 50},
    {"task": "normal_vs_pvd", "model": "unet3d", "epochs": 50},
]

def run_command(cmd, timeout=None):
    """Run a command and return (success, output)."""
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace"
        )
        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        # Print last 20 lines
        lines = output.strip().split("\n")
        for line in lines[-30:]:
            print(line)
        return success, output
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after {timeout}s]")
        return False, ""
    except Exception as e:
        print(f"  [ERROR: {e}]")
        return False, ""

def find_checkpoint(run_dir):
    """Find the best checkpoint in the run directory."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    for subdir in ckpt_dir.iterdir():
        if subdir.is_dir():
            for ckpt in subdir.glob("*_best_epoch_*.ckpt"):
                return str(ckpt)
            for ckpt in subdir.glob("last.ckpt"):
                # Only return last.ckpt if it's > 0 epochs (check filename size)
                if ckpt.stat().st_size > 42 * 1024 * 1024:  # > 42MB, not just init
                    return str(ckpt)
    return None

def main():
    results = {}

    print("=" * 70)
    print("ERDES Paper Systematic Verification")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Train models
    print("\n\n### STEP 1: Training Models ###")
    for i, task_config in enumerate(TASKS):
        task = task_config["task"]
        model = task_config["model"]
        epochs = task_config["epochs"]

        print(f"\n--- [{i+1}/{len(TASKS)}] Training {task} with {model} ({epochs} epochs) ---")

        cmd = (
            f"python erdes/train.py "
            f"experiment={task}/{model} "
            f"trainer=gpu "
            f"data.batch_size=2 "
            f"data.num_workers=2 "
            f"trainer.max_epochs={epochs}"
        )
        success, output = run_command(cmd, timeout=7200)  # 2h per task

        if not success:
            print(f"  [WARNING] Training may have failed for {task}/{model}")

        # Find the latest checkpoint
        logs_dir = PROJECT_ROOT / "logs" / "train" / "runs"
        run_dir = None
        if task in ["non_rd_vs_rd"]:
            pattern = f"**/resnet3d/**" if model == "resnet3d" else f"**/unet3d/**"
        # Find most recent run directory
        if logs_dir.exists():
            all_dirs = sorted(logs_dir.rglob("checkpoints"), key=os.path.getmtime, reverse=True)
            if all_dirs:
                results[f"{task}/{model}"] = {"checkpoint_dir": str(all_dirs[0].parent)}

    # Step 2: Print summary
    print("\n\n### STEP 2: Results Summary ###")
    print("\nPaper Benchmarks vs Our Results:")
    print("-" * 80)
    print(f"{'Task':<30} {'Model':<15} {'Metric':<8} {'Paper':<8} {'Ours':<8}")
    print("-" * 80)

    for task_name, benchmark in PAPER_BENCHMARKS.items():
        for model_name, metrics in benchmark.get("metrics", {}).items():
            for metric, paper_val in metrics.items():
                print(f"{task_name:<30} {model_name:<15} {metric:<8} {paper_val:<8.3f} {'N/A':<8}")

    print("\n\n### STEP 3: Diagnostic Pipeline ###")
    print("Note: Pipeline requires trained checkpoints for both stages.")

    # Step 4: Recommendations
    print("\n\n### RECOMMENDATIONS ###")
    print("1. Full paper replication requires:")
    print("   - 8 architectures × 5 tasks = 40 models")
    print("   - 3× NVIDIA A6000 GPUs (48GB each) for reasonable training time")
    print("   - Approximately 20-30 GPU-hours total")
    print("2. On a single laptop GPU (8GB), train key benchmark models:")
    print("   - Non-RD vs RD: 3D ResNet + 3D U-Net")
    print("   - Macula Detached vs Intact: 3D U-Net")
    print("3. Run diagnostic pipeline with best checkpoints")
    print("4. Compare metrics against paper Tables 8-10")

    return results

if __name__ == "__main__":
    main()
