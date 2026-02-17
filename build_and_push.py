#!/usr/bin/env python3
"""
Build and push wisconsin.pdf dataset to HuggingFace Hub.
Uses LLM API for faster captioning.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from doc_parser_engine.core import DocParserEngine

# Initialize engine with LLM API for faster captioning
engine = DocParserEngine(
    enable_captioning=True,
    caption_model="api",  # Use API-based captioning
    llm_api_base="http://0.0.0.0:7543",
    llm_model="gpt-4o",
    verbose=True,
)

# Parse the PDF
doc_path = Path("wisconsin.pdf")
print(f"Parsing {doc_path}...")
doc = engine.parse(doc_path)

print(f"\nParsed document:")
print(f"  Title: {doc.title}")
print(f"  Word count: {doc.word_count}")
print(f"  Images: {doc.image_count}")
print(f"  Pages: {doc.page_count}")
print(f"  Sections: {len(doc.sections)}")
print(f"  Paragraphs: {len(doc.paragraphs)}")

# Get HuggingFace credentials
hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")
hub_repo = f"{hf_username}/wisconsin-motorists-handbook-dataset"

# Build and push dataset
print(f"\nBuilding dataset and pushing to {hub_repo}...")
dataset = engine.to_hf_dataset(
    [doc],
    schema="full",
    include_images=False,  # Skip images to avoid large uploads
    push_to_hub=True,
    hub_repo=hub_repo,
    hub_token=hf_token,
)

print(f"\n✅ Dataset pushed successfully!")
print(f"   URL: https://huggingface.co/datasets/{hub_repo}")
