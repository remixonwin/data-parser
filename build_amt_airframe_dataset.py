#!/usr/bin/env python3
"""
Build and push amt_airframe.pdf dataset to HuggingFace Hub.
Uses LLM API for faster captioning.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_parser_engine.core import DocParserEngine

# Initialize engine with LLM API for faster captioning
print("Initializing DocParserEngine...")
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",  # Use API-based captioning
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
    verbose=True,
)

# Parse the PDF
doc_path = Path("amt_airframe.pdf")
print(f"\n{'='*60}")
print(f"Parsing {doc_path}...")
print(f"{'='*60}")
doc = engine.parse(doc_path)

print(f"\n{'='*60}")
print(f"Parsed document:")
print(f"{'='*60}")
print(f"  Title: {doc.title}")
print(f"  Word count: {doc.word_count}")
print(f"  Images: {doc.image_count}")
print(f"  Pages: {doc.page_count}")
print(f"  Sections: {len(doc.sections)}")
print(f"  Paragraphs: {len(doc.paragraphs)}")
print(f"  Tables: {len(doc.tables)}")

# Show some sample content - sections may be dicts
if doc.sections:
    print(f"\nFirst 3 sections:")
    for i, section in enumerate(doc.sections[:3]):
        if isinstance(section, dict):
            title = section.get('title', 'N/A')
            content = section.get('content', '')[:100]
        else:
            title = getattr(section, 'title', 'N/A')
            content = getattr(section, 'content', '')[:100]
        print(f"  {i+1}. {title}: {content}...")

if doc.images:
    print(f"\nImages found: {len(doc.images)}")
    for i, img in enumerate(doc.images[:3]):
        if isinstance(img, dict):
            print(f"  Image {i+1}: {img.get('category', 'N/A')}, page {img.get('page_number', 'N/A')}")
        else:
            print(f"  Image {i+1}: {getattr(img, 'category', 'N/A')}, page {getattr(img, 'page_number', 'N/A')}")
else:
    print(f"\nNo images found in the PDF")

# Show paragraphs sample
if doc.paragraphs:
    print(f"\nFirst 3 paragraphs:")
    for i, para in enumerate(doc.paragraphs[:3]):
        if isinstance(para, dict):
            content = para.get('content', '')[:150]
        else:
            content = getattr(para, 'content', '')[:150]
        print(f"  {i+1}. {content}...")

# Get HuggingFace credentials
hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")
hub_repo = f"{hf_username}/amt-airframe-handbook-dataset"

# Build and push dataset
print(f"\n{'='*60}")
print(f"Building dataset and pushing to {hub_repo}...")
print(f"{'='*60}")
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=True,  # Include images in dataset
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"\n✅ Dataset pushed successfully!")
print(f"   URL: https://huggingface.co/datasets/{hub_repo}")
print(f"\nDataset info:")
print(f"  Number of rows: {len(dataset)}")
print(f"  Columns: {dataset.column_names}")
