"""Extract all images from essay.pdf, grouped by page."""
import fitz, os

PDF = "essay.pdf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures")
os.makedirs(OUT, exist_ok=True)

doc = fitz.open(PDF)
print(f"Total pages: {len(doc)}")

for page_idx in range(len(doc)):
    page = doc[page_idx]
    images = page.get_images(full=True)
    if images:
        print(f"\nPage {page_idx+1}: {len(images)} image(s)")
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            w, h = base_image["width"], base_image["height"]
            fname = os.path.join(OUT, f"p{page_idx+1:02d}_img{img_idx+1:02d}.{ext}")
            with open(fname, "wb") as f:
                f.write(base_image["image"])
            print(f"  → {fname}  ({w}x{h})")

doc.close()
print("\nDone.")
