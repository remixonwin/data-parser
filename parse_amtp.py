#!/usr/bin/env python3
"""
Parse amtp_ch1.pdf and upload to HF
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.INFO)

from doc_parser_engine.core import DocParserEngine

print("=" * 70)
print("Parsing amtp_ch1.pdf")
print("=" * 70)

engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip-image-captioning-large",
    image_min_size=30,
    verbose=True,
)

doc = engine.parse(Path("amtp_ch1.pdf"))

print(f"\n📊 RESULTS:")
print(f"   Title: {doc.title}")
print(f"   Words: {doc.word_count}")
print(f"   Pages: {doc.page_count}")
print(f"   Images: {doc.image_count}")
print(f"   Paragraphs: {len(doc.paragraphs)}")

# Upload
print(f"\n📤 Uploading...")
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,
    push_to_hub=True,
    hub_repo="Remixonwin/amtp-ch1-dataset",
    hub_token=os.getenv("HF_TOKEN"),
)

print(f"\n✅ Done! https://huggingface.co/datasets/Remixonwin/amtp-ch1-dataset")
