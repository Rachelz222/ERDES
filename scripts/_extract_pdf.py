import fitz
doc = fitz.open("d:/projects/ERDES/essay.pdf")
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"\n===== PAGE {i+1} =====\n")
    print(text)
