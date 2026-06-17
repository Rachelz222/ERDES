"""Download pretrained checkpoints from Zenodo with resume support."""
import os
import sys
import requests
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# All files available on Zenodo record 18821031
FILES = {
    # Core pipeline checkpoints
    "unet3d_macula_detached_vs_intact_best.ckpt": "286.4MB",
    "resnet3d_non_rd_vs_rd_best.ckpt": "810.2MB",
    "unet3d_non_rd_vs_rd_best.ckpt": "286.4MB",
}

ZENODO_BASE = "https://zenodo.org/records/18821031/files"


def download_file(filename, expected_mb=None):
    """Download with progress display."""
    url = f"{ZENODO_BASE}/{filename}?download=1"
    out_path = WEIGHTS_DIR / filename

    print(f"\nDownloading: {filename}")
    print(f"URL: {url}")
    print(f"Target: {out_path}")

    headers = {}
    if out_path.exists():
        headers["Range"] = f"bytes={out_path.stat().st_size}-"
        print(f"Resuming from {out_path.stat().st_size / 1024**2:.1f} MB")

    response = requests.get(url, stream=True, headers=headers, timeout=30)

    if response.status_code == 416:
        print("File already fully downloaded!")
        return True

    total_size = int(response.headers.get("content-length", 0))
    mode = "ab" if "Range" in headers else "wb"

    last_report = 0
    with open(out_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                current = out_path.stat().st_size / 1024**2
                if current - last_report >= 10:  # Report every 10MB
                    if total_size:
                        print(f"  {current:.1f} / {total_size/1024**2:.1f} MB ({current/total_size*100:.0f}%)")
                    else:
                        print(f"  {current:.1f} MB downloaded")
                    last_report = current

    final_size = out_path.stat().st_size / 1024**2
    print(f"Complete: {final_size:.1f} MB")

    if expected_mb:
        # Check if size is close to expected
        if final_size < expected_mb * 0.9:
            print(f"  WARNING: File smaller than expected ({expected_mb}MB expected)")

    return True


def main():
    print("=" * 60)
    print("Downloading ERDES Pretrained Checkpoints")
    print("=" * 60)

    for filename, size in FILES.items():
        out_path = WEIGHTS_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            size_mb = out_path.stat().st_size / 1024**2
            print(f"\n{filename}: {size_mb:.1f} MB (already exists)")
            resp = input("Re-download? [y/N]: ").strip().lower()
            if resp != "y":
                continue
            out_path.unlink()

        # Extract expected size in MB
        expected_mb = float(size.replace("MB", ""))
        download_file(filename, expected_mb=expected_mb)

    # List final results
    print("\n" + "=" * 60)
    print("Downloaded files:")
    total = 0
    for f in sorted(WEIGHTS_DIR.glob("*.ckpt")):
        mb = f.stat().st_size / 1024**2
        total += mb
        print(f"  {f.name}: {mb:.1f} MB")
    print(f"  Total: {total:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
