#!/usr/bin/env python3
"""
Build and push amt_airframe.pdf dataset to HuggingFace Hub.
Uses render_pages=True to capture vector graphics as images.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_parser_engine.core import DocParserEngine

# Initialize engine with render_pages=True to capture vector graphics
print("Initializing DocParserEngine with render_pages=True...")
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",  # Use API-based captioning
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
    render_pages=True,    # NEW: Enable page rendering to capture vector graphics
    render_dpi=150,      # DPI for rendered images
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

# Show some sample images
if doc.images:
    print(f"\n--- Sample Images (first 5) ---")
    for i, img in enumerate(doc.images[:5]):
        if isinstance(img, dict):
            img_id = img.get('image_id', 'N/A')
            width = img.get('width', 'N/A')
            height = img.get('height', 'N/A')
            page = img.get('page', 'N/A')
            caption = img.get('caption', '')[:50] if img.get('caption') else 'No caption'
        else:
            img_id = getattr(img, 'image_id', 'N/A')
            width = getattr(img, 'width', 'N/A')
            height = getattr(img, 'height', 'N/A')
            page = getattr(img, 'page', 'N/A')
            caption = getattr(img, 'caption', '')[:50] if hasattr(img, 'caption') and img.caption else 'No caption'
        print(f"  {i+1}. ID: {img_id}, Size: {width}x{height}, Page: {page}")
        print(f"      Caption: {caption}")

# Show sections sample
if doc.sections:
    print(f"\n--- Sample Sections (first 3) ---")
    for i, section in enumerate(doc.sections[:3]):
        if isinstance(section, dict):
            title = section.get('title', 'N/A')
            content = section.get('content', '')[:100]
        else:
            title = getattr(section, 'title', 'N/A')
            content = getattr(section, 'content', '')[:100]
        print(f"  {i+1}. {title}: {content}...")

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

# Quality summary
print(f"\n{'='*60}")
print("QUALITY SUMMARY")
print(f"{'='*60}")
print(f"✅ Text: {doc.word_count} words across {doc.page_count} pages")
print(f"✅ Paragraphs: {len(doc.paragraphs) if isinstance(doc.paragraphs, list) else 'structured'}")
print(f"✅ Images: {doc.image_count} images (including rendered pages)")
print(f"✅ Tables: {doc.table_count}")
print(f"\nDataset URL: https://huggingface.co/datasets/{hub_repo}")
