#!/usr/bin/env python3
"""
Re-extract wisconsin.pdf with:
1. Lower min_size (50) to capture more images
2. HF BLIP model for unique per-image captions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Suppress excessive logging
import logging

logging.basicConfig(level=logging.WARNING)

from doc_parser_engine.core import DocParserEngine

print("=" * 70)
print("RE-EXTRACTING WISCONSIN PDF WITH PROPER SETTINGS")
print("=" * 70)

# Initialize engine with lower min_size and HF model for unique captions
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip-image-captioning-large",  # HF model for unique captions
    image_min_size=50,  # Lower threshold to get more images
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
print(f"\n🖼️ ALL IMAGES WITH CAPTIONS:")
print("-" * 70)
for i, img in enumerate(doc.images):
    caption = img.get("caption", "N/A")
    # Show first 80 chars of caption
    cap_preview = caption[:80] + "..." if len(caption) > 80 else caption
    print(
        f"{i + 1:2d}. Page {img.get('page')}: {img.get('width')}x{img.get('height')} [{img.get('category')}]"
    )
    print(f"     {cap_preview}")

# Check for unique captions
captions = [img.get("caption", "") for img in doc.images]
unique_captions = set(captions)
print(f"\n📝 Caption Quality:")
print(f"  Total images: {len(captions)}")
print(f"  Unique captions: {len(unique_captions)}")

# Push to HF
hf_token = os.getenv("HF_TOKEN")
hub_repo = "Remixonwin/wisconsin-motorists-handbook-dataset"

print(f"\n🚀 Pushing updated dataset to {hub_repo}...")
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"\n✅ Dataset updated successfully!")
print(f"   URL: https://huggingface.co/datasets/{hub_repo}")
