import fitz
from pathlib import Path

doc_path = Path("wisconsin.pdf")
if not doc_path.exists():
    print("File not found.")
    exit(1)

doc = fitz.open(str(doc_path))
total_images = 0
pages_with_images = 0

for page in doc:
    images = page.get_images(full=True)
    if images:
        total_images += len(images)
        pages_with_images += 1

print(f"Total images found via get_images: {total_images}")
print(f"Pages with images: {pages_with_images}")
doc.close()
