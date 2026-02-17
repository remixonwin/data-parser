import hashlib
import uuid


class ImageExtractor:
    def __init__(self, **kwargs):
        pass

    def __init__(self, min_size=0, **kwargs):
        self.min_size = min_size

    def extract(self, raw, doc_id="unknown"):
        raw_images = raw.get("raw_images", [])
        if not raw_images:
            return []

        extracted_images = []
        hashes = set()

        for img in raw_images:
            image_bytes = img.get("image_bytes") or img.get(
                "data"
            )  # Check both for robustness
            if not image_bytes:
                continue

            # Size filtering - exclude only if BOTH dimensions are too small
            # This keeps wide but short images (banners) and tall but narrow images (sidebars)
            width = img.get("width", 0)
            height = img.get("height", 0)
            if width < self.min_size and height < self.min_size:
                continue

            # Content-based hashing for deduplication
            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash in hashes:
                continue

            hashes.add(img_hash)

            # Generate a clean image_id
            short_hash = img_hash[slice(0, 8)]
            image_id = f"{doc_id}_{len(extracted_images):04d}_{short_hash}"

            extracted_images.append(
                {
                    "image_id": image_id,
                    "doc_id": doc_id,
                    "image_bytes": image_bytes,
                    "format": img.get("format", "png"),
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 0,
                    "page": img.get("page", 0),
                    "bbox": img.get("bbox", []),
                    "content_hash": img_hash,
                }
            )

        return extracted_images
