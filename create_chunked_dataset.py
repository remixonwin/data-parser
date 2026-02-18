"""
Create a chunked version of the amt-airframe dataset for MCQ generation.
The MCQ generator has a max_length filter of 5000 characters, so we need
to split the text into smaller chunks.
"""
from datasets import load_dataset, Dataset
import json
import os

# Get token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Load the original dataset
print("Loading original dataset...")
ds = load_dataset("Remixonwin/amt-airframe-flat")
original = ds["train"][0]

text = original["text"]
print(f"Original text length: {len(text)} characters")

# Split text into chunks of ~3000 characters (leaving room for prompt overhead)
CHUNK_SIZE = 3000
chunks = []
current_chunk = ""

# Split by sentences to maintain coherence
import re

# Split on sentence boundaries
sentences = re.split(r'(?<=[.!?])\s+', text)

for sentence in sentences:
    if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
        current_chunk += " " + sentence if current_chunk else sentence
    else:
        if current_chunk:
            chunks.append(current_chunk.strip())
        # Start new chunk, but if single sentence is too long, truncate it
        if len(sentence) > CHUNK_SIZE:
            # Split long sentence
            current_chunk = sentence[:CHUNK_SIZE]
        else:
            current_chunk = sentence

# Don't forget the last chunk
if current_chunk:
    chunks.append(current_chunk.strip())

print(f"Created {len(chunks)} chunks")
print(f"First chunk length: {len(chunks[0])} characters")
print(f"First chunk preview: {chunks[0][:200]}...")

# Create new dataset
chunked_data = {
    "doc_id": [],
    "title": [],
    "text": [],
    "chunk_index": [],
    "word_count": [],
}

for i, chunk in enumerate(chunks):
    chunked_data["doc_id"].append(original["doc_id"])
    chunked_data["title"].append(original["title"])
    chunked_data["text"].append(chunk)
    chunked_data["chunk_index"].append(i)
    chunked_data["word_count"].append(len(chunk.split()))

# Create dataset
chunked_ds = Dataset.from_dict(chunked_data)
print(f"Created dataset with {len(chunked_ds)} rows")

# Push to HuggingFace
print("Pushing to HuggingFace...")
chunked_ds.push_to_hub("Remixonwin/amt-airframe-chunks", token=HF_TOKEN)
print("Done! Dataset pushed to Remixonwin/amt-airframe-chunks")
