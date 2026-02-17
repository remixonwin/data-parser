#!/usr/bin/env python3
"""
Re-extract wisconsin.pdf with all images and proper captioning.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from doc_parser_engine.core import DocParserEngine

# Initialize engine - lower min_size to get all images, use HF model for better captions
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip2-opt-2.7b",  # Use HF model for unique captions
    image_min_size=50,  # Lower threshold to get more images
    verbose=True,
)

doc_path = Path("wisconsin.pdf")
print(f"Parsing {doc_path} with all images...")
doc = engine.parse(doc_path)

print(f"\n📊 EXTRACTION RESULTS:")
print(f"  Title: {doc.title}")
print(f"  Word count: {doc.word_count}")
print(f"  Images extracted: {doc.image_count}")
print(f"  Pages: {doc.page_count}")

# Show all images with their captions
print(f"\n🖼️ ALL IMAGES WITH CAPTIONS:")
print("-" * 70)
for i, img in enumerate(doc.images):
    print(
        f"{i + 1}. Page {img.get('page')}: {img.get('width')}x{img.get('height')} [{img.get('category')}]"
    )
    print(f"   Caption: {img.get('caption', 'N/A')[:100]}...")
    print()

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

print(f"\n✅ Dataset updated!")
print(f"   URL: https://huggingface.co/datasets/{hub_repo}")
