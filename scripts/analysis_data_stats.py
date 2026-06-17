"""
Data Statistics & Motion Score Analysis (decord-accelerated)
==============================================================
Uses decord (10x faster than torchvision) + ThreadPoolExecutor.
Motion Score = mean(|frame_{t+1} - frame_t|) on ROI grayscale.
"""
import os, sys, time, logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import decord
from tqdm import tqdm

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "stats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "runtime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(f"=== Motion Score Analysis (decord) started at {datetime.now().isoformat()} ===")

from erdes.data.components.utils import resize
SIZE = (96, 128, 128)
resize_tf = resize(SIZE)


def process_one(args: tuple) -> dict:
    """Load video via decord, resize, compute motion/brightness/contrast."""
    abs_path, rel_path, meta_row = args
    try:
        vr = decord.VideoReader(str(abs_path))
        n = len(vr)
        if n < 2:
            return None
        # Decode all frames: [T, H, W, C] uint8
        frames = vr.get_batch(list(range(n))).asnumpy()
    except Exception:
        return None

    # [T, H, W, C] → float → [C, T, H, W] grayscale
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)  # [C, T, H, W]
    if video.shape[0] == 3:
        video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]
    video = resize_tf(video) / 255.0

    D = video.shape[1]
    frames_t = video.squeeze(0)  # [D, H, W]

    ms = torch.abs(frames_t[1:] - frames_t[:-1]).mean().item() if D >= 2 else 0.0
    brightness = float(video.mean().item())
    contrast = float(video.std().item())

    return {
        "clip_id": Path(abs_path).stem,
        "file_path": rel_path,
        "diagnostic_class": str(meta_row.get("diagnostic_class", "unknown")),
        "subtype": str(meta_row.get("subtype", "unknown")),
        "frame_count": D,
        "motion_score": round(ms, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
    }


def main():
    df_meta = pd.read_csv(PROJECT_ROOT / "data" / "erdes" / "erdes_metadata.csv")
    log.info(f"Metadata: {len(df_meta)} entries")

    # Collect unique test paths
    splits_dir = PROJECT_ROOT / "data" / "splits"
    all_test = set()
    for td in sorted(splits_dir.iterdir()):
        if td.is_dir() and (td / "test.csv").exists():
            for _, row in pd.read_csv(td / "test.csv").iterrows():
                all_test.add(str(PROJECT_ROOT / "data" / "erdes" / row["path"]))
    test_paths = sorted(all_test)
    log.info(f"Test videos: {len(test_paths)}")

    # Build tasks
    tasks = []
    for p in test_paths:
        if not os.path.isfile(p):
            continue
        rel = str(Path(p).relative_to(PROJECT_ROOT / "data" / "erdes"))
        cid = Path(p).stem
        rows = df_meta[df_meta["clip_id"].astype(str) == cid]
        if len(rows) == 0:
            rows = df_meta[df_meta["file_path"].str.contains(rel.replace("\\", "/"), na=False)]
        mr = rows.iloc[0].to_dict() if len(rows) > 0 else {"diagnostic_class": "unknown", "subtype": "unknown"}
        tasks.append((p, rel, mr))
    log.info(f"Valid: {len(tasks)}")

    # Parallel with decord
    workers = min(8, os.cpu_count() or 4)
    log.info(f"Threads: {workers}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Computing"):
            try:
                r = fut.result()
                if r: results.append(r)
            except:
                pass
    elapsed = time.time() - t0
    rate = len(results) / elapsed if elapsed > 0 else 0
    log.info(f"Done: {len(results)}/{len(tasks)} in {elapsed:.1f}s ({rate:.1f} videos/s)")

    df_out = pd.DataFrame(results)
    out = RESULTS_DIR / "motion_scores.csv"
    df_out.to_csv(out, index=False)
    log.info(f"Saved: {out}")

    log.info("\n=== Motion Score by Class ===")
    for cls_name in sorted(df_out["diagnostic_class"].unique()):
        s = df_out[df_out["diagnostic_class"] == cls_name]
        log.info(f"  {cls_name:<15}: n={len(s):<5} motion={s['motion_score'].mean():.6f}±{s['motion_score'].std():.6f}")
    log.info(f"=== Completed at {datetime.now().isoformat()} ===")
    return df_out


if __name__ == "__main__":
    main()
