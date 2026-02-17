#!/usr/bin/env python3
"""
End-to-End Test for amtp_ch1.pdf
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.WARNING)

from doc_parser_engine.core import DocParserEngine
from datasets import load_dataset

print("=" * 70)
print("E2E TEST: amtp_ch1.pdf")
print("=" * 70)

# STEP 1: Parse
print("\n📄 STEP 1: Parsing amtp_ch1.pdf...")

engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip-image-captioning-large",
    image_min_size=30,
    verbose=False,
)

doc = engine.parse(Path("amtp_ch1.pdf"))

print(f"   Title: {doc.title}")
print(f"   Words: {doc.word_count}, Pages: {doc.page_count}")
print(f"   Images: {doc.image_count}")

# STEP 2: Upload
print("\n📤 STEP 2: Uploading to HuggingFace...")

hf_token = os.getenv("HF_TOKEN")
hub_repo = "Remixonwin/amtp-ch1-dataset"

dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"   Uploaded to: https://huggingface.co/datasets/{hub_repo}")

# STEP 3: Download & Verify
print("\n📥 STEP 3: Downloading dataset...")

dataset = load_dataset(hub_repo)
train = dataset["train"]
row = train[0]

print(f"   Rows: {len(train)}")

# STEP 4: Quality Check
print("\n🔍 STEP 4: Quality Verification...")
print("-" * 70)

print(f"📄 Document Info:")
print(f"   Title: {row['title']}")
print(f"   Authors: {row['authors']}")
print(f"   Word count: {row['word_count']}")
print(f"   Page count: {row['page_count']}")

images = row["images"]
print(f"\n🖼️ Image Quality:")
print(f"   Total images: {len(images.get('image_id', []))}")
unique_caps = len(set(images.get("caption", [])))
print(f"   Unique captions: {unique_caps}")

paragraphs = row["paragraphs"]
texts = paragraphs.get("text", [])
print(f"\n📝 Text Quality:")
print(f"   Total paragraphs: {len(texts)}")
non_empty = sum(1 for t in texts if t and len(t.strip()) > 0)
print(f"   Non-empty: {non_empty}")
print(f"   First paragraph: {texts[0][:100] if texts else 'N/A'}...")

sections = row["sections"]
print(f"\n📑 Structure:")
print(f"   Sections: {len(sections.get('title', []))}")

# HF Convention Check
print("\n✅ HF Convention Check:")
checks = [
    ("Required metadata", all(row.get(f) for f in ["doc_id", "title", "word_count"])),
    ("Has train split", "train" in dataset),
    ("Valid features", train.features.get("doc_id") is not None),
]
for name, passed in checks:
    print(f"   {'✅' if passed else '❌'} {name}")

print("\n" + "=" * 70)
print("✅ E2E TEST COMPLETE")
print("=" * 70)
print(f"Dataset: https://huggingface.co/datasets/{hub_repo}")
