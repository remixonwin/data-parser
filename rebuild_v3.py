#!/usr/bin/env python3
"""
Re-extract wisconsin.pdf with fixes:
1. Image filter now uses OR logic (keeps wide/short images)
2. LLM API properly sends images for unique captions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.WARNING)

from doc_parser_engine.core import DocParserEngine

print("=" * 70)
print("RE-EXTRACTING WITH FIXES")
print("=" * 70)
print("Fix 1: Image filter - OR logic (keeps wide/short images)")
print("Fix 2: LLM API - sends actual images for unique captions")

# Initialize engine with LLM API for fast processing
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",  # Use LLM API for fast processing
    image_min_size=50,  # Lower threshold
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
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
    caption = img.get("caption", "N/A")[:60]
    print(
        f"{i + 1:2d}. Page {img.get('page')}: {img.get('width')}x{img.get('height')} | {caption}..."
    )

print(f"\n📝 Caption Quality:")
print(f"  Total images: {len(captions)}")
print(f"  Unique captions: {len(unique_captions)}")

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
