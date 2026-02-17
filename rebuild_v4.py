#!/usr/bin/env python3
"""
Re-extract wisconsin.pdf with:
1. Fixed image filter (OR logic)
2. BLIP model for unique captions (more reliable than LLM API for vision)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.WARNING)

from doc_parser_engine.core import DocParserEngine

print("=" * 70)
print("RE-EXTRACTING WITH BLIP FOR UNIQUE CAPTIONS")
print("=" * 70)
print("Using BLIP model (proven to produce unique captions)")

# Use BLIP for reliable unique captions
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip-image-captioning-large",  # BLIP - reliable for unique captions
    image_min_size=30,  # Include almost all images
    verbose=True,
)

doc_path = Path("wisconsin.pdf")
print(f"\nParsing {doc_path}...")
doc = engine.parse(doc_path)

print(f"\n📊 EXTRACTION RESULTS:")
print(f"  Title: {doc.title}")
print(f"  Word count: {doc.word_count}")
print(f"  Images extracted: {doc.image_count}")
print(f"  Pages: {doc.page_count}")

# Check all images
captions = [img.get("caption", "") for img in doc.images]
unique_captions = set(captions)

print(f"\n🖼️ ALL IMAGES ({len(doc.images)} total):")
print("-" * 70)
for i, img in enumerate(doc.images):
    caption = img.get("caption", "N/A")[:50]
    page = img.get("page")
    size = f"{img.get('width')}x{img.get('height')}"
    print(f"{i + 1:2d}. Page {page}: {size} | {caption}...")

print(f"\n📝 Caption Quality:")
print(f"  Total images: {len(captions)}")
print(f"  Unique captions: {len(unique_captions)}")
print(f"  Diversity: {len(unique_captions) / len(captions) * 100:.1f}%")

# Push to HF
hf_token = os.getenv("HF_TOKEN")
hub_repo = "Remixonwin/wisconsin-motorists-handbook-dataset"

print(f"\n🚀 Pushing to {hub_repo}...")
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"\n✅ Done!")
print(f"   URL: https://huggingface.co/datasets/{hub_repo}")
