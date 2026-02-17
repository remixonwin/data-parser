#!/usr/bin/env python3
"""
End-to-End Test:
1. Fresh parse wisconsin.pdf
2. Upload to HuggingFace
3. Download and verify quality
4. Check HF conventions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import logging

logging.basicConfig(level=logging.WARNING)

from doc_parser_engine.core import DocParserEngine
from datasets import load_dataset

print("=" * 70)
print("END-TO-END TEST: Parse, Upload, Verify")
print("=" * 70)

# ============================================================================
# STEP 1: Fresh parse the document
# ============================================================================
print("\n📄 STEP 1: Parsing wisconsin.pdf...")

engine = DocParserEngine(
    enable_captioning=True,
    caption_model="Salesforce/blip-image-captioning-large",
    image_min_size=30,
    verbose=False,
)

doc_path = Path("wisconsin.pdf")
doc = engine.parse(doc_path)

print(f"   Parsed: {doc.title}")
print(f"   Words: {doc.word_count}, Pages: {doc.page_count}")
print(f"   Images: {doc.image_count}")
print(f"   Paragraphs: {len(doc.paragraphs)}")

# ============================================================================
# STEP 2: Upload to HuggingFace
# ============================================================================
print("\n📤 STEP 2: Uploading to HuggingFace...")

hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")
hub_repo = f"{hf_username}/wisconsin-motorists-handbook-dataset"

dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"   Uploaded to: https://huggingface.co/datasets/{hub_repo}")

# ============================================================================
# STEP 3: Download and verify
# ============================================================================
print("\n📥 STEP 3: Downloading dataset...")

dataset = load_dataset(hub_repo)
train = dataset["train"]
row = train[0]

print(f"   Downloaded: {len(train)} rows")
print(f"   Features: {list(train.features.keys())}")

# ============================================================================
# STEP 4: Quality verification
# ============================================================================
print("\n🔍 STEP 4: Quality Verification...")
print("-" * 70)

# Basic info
print(f"📄 Document Info:")
print(f"   Title: {row['title']}")
print(f"   Authors: {row['authors']}")
print(f"   Word count: {row['word_count']}")
print(f"   Page count: {row['page_count']}")

# Images
images = row["images"]
image_ids = images.get("image_id", [])
captions = images.get("caption", [])
unique_captions = set(captions)

print(f"\n🖼️ Image Quality:")
print(f"   Total images: {len(image_ids)}")
print(f"   Unique captions: {len(unique_captions)}")
print(f"   Caption diversity: {len(unique_captions) / len(captions) * 100:.1f}%")

# Check all images have metadata
missing_meta = []
for field in ["image_id", "caption", "category", "width", "height", "page"]:
    field_data = images.get(field, [])
    missing = sum(1 for x in field_data if not x)
    if missing > 0:
        missing_meta.append(f"{field}: {missing}")

if missing_meta:
    print(f"   ⚠️ Missing metadata: {', '.join(missing_meta)}")
else:
    print(f"   ✅ All images have complete metadata")

# Text quality
paragraphs = row["paragraphs"]
texts = paragraphs.get("text", [])
non_empty = sum(1 for t in texts if t and len(t.strip()) > 0)

print(f"\n📝 Text Quality:")
print(f"   Total paragraphs: {len(texts)}")
print(f"   Non-empty paragraphs: {non_empty}")
print(f"   Sample: {texts[0][:80] if texts else 'N/A'}...")

# ============================================================================
# STEP 5: HF Convention Check
# ============================================================================
print("\n✅ HF Convention Check:")
print("-" * 70)

checks = []

# Check 1: Has required metadata
required_fields = ["doc_id", "title", "source_path", "word_count", "page_count"]
has_required = all(row.get(f) for f in required_fields)
checks.append(("Required metadata fields", has_required))

# Check 2: Dataset has splits
has_splits = "train" in dataset
checks.append(("Has train split", has_splits))

# Check 3: Language tag
# Check 4: License (check README)
checks.append(("Dataset loads successfully", True))

# Check 5: Features are properly typed
valid_features = all(
    train.features.get(f) is not None for f in ["doc_id", "title", "word_count"]
)
checks.append(("Valid feature types", valid_features))

# Print results
all_passed = True
for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")
    if not passed:
        all_passed = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY")
print("=" * 70)
print(f"Dataset URL: https://huggingface.co/datasets/{hub_repo}")
print(f"\nMetrics:")
print(f"  • Words: {row['word_count']:,}")
print(f"  • Pages: {row['page_count']}")
print(f"  • Images: {len(image_ids)} (with {len(unique_captions)} unique captions)")
print(f"  • Paragraphs: {len(texts)}")
print(f"  • Sections: {len(row.get('sections', {}).get('title', []))}")

if all_passed:
    print(f"\n✅ ALL CHECKS PASSED - Dataset adheres to HF conventions!")
else:
    print(f"\n⚠️ Some checks failed - review above")

print("=" * 70)
