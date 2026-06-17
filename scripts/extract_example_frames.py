"""Extract one middle frame from each ERDES diagnostic category."""
import os, glob
import torchvision.io
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "erdes")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_frames")
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
    vframes, _, info = torchvision.io.read_video(mp4s[0], output_format="TCHW")
    total = vframes.shape[0]
    mid = total // 2
    frame = vframes[mid].float().mean(dim=0)
    img = Image.fromarray(frame.numpy().astype("uint8"))
    out_path = os.path.join(OUT, f"{name}.png")
    img.save(out_path)
    fps = info.get("video_fps", 0)
    print(f"{name}: frame {mid}/{total} @ {fps:.1f}fps → {out_path}")
print("Done.")
