"""Extract short clips from ERDES diagnostic categories using torchvision."""
import os, glob
import torchvision.io

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "erdes")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_clips")
os.makedirs(OUT, exist_ok=True)

targets = {
    "Normal": "Non_Retinal_Detachment/Normal",
    "PVD": "Non_Retinal_Detachment/Posterior_Vitreous_Detachment",
    "Macula_Detached": "Retinal_Detachment/Macula_Detached/Bilateral",
    "Macula_Intact": "Retinal_Detachment/Macula_Intact/TD",
}

for name, subdir in targets.items():
    mp4s = sorted(glob.glob(os.path.join(ROOT, subdir, "*.mp4")))
    if not mp4s:
        print(f"Not found: {subdir}")
        continue

    src = mp4s[0]
    vframes, _, info = torchvision.io.read_video(src, output_format="TCHW")

    # Take first 90 frames (~3 sec at 30fps) or fewer if clip is shorter
    max_frames = min(90, vframes.shape[0])
    clip = vframes[:max_frames]

    # Downsample spatially to 256x256 for smaller file
    clip_resized = torchvision.transforms.functional.resize(
        clip, [256, 256], antialias=True
    )

    dst = os.path.join(OUT, f"{name}.mp4")
    fps = info.get("video_fps", 27.0)
    torchvision.io.write_video(dst, clip_resized.permute(0, 2, 3, 1), fps=fps, video_codec="libx264")
    size_kb = os.path.getsize(dst) / 1024
    print(f"{name}: {max_frames} frames @ {fps:.0f}fps, {size_kb:.0f} KB → {dst}")
print("Done.")
