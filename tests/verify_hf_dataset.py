import os
import sys
import logging
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

REPO_ID = os.getenv("REPO_ID", "Remixonwin/prepware_study_guide-dataset")


def _first_image_from_field(images):
    """Return the first image-like dict from a dataset field that may be a list or dict."""
    if not images:
        return None
    # If it's a list, return first element
    if isinstance(images, (list, tuple)):
        return images[0] if len(images) > 0 else None
    # If it's a dict, try numeric keys then fallback to first value
    if isinstance(images, dict):
        try:
            keys = sorted(images.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
            return images[keys[0]]
        except Exception:
            # Fallback: return first value
            return next(iter(images.values()), None)
    # Unknown type
    return None


def verify_dataset():
    console.print(f"\n[bold blue]🔍 Verifying Dataset: {REPO_ID}[/bold blue]")

    try:
        # Load the dataset from the Hub
        logger.info(f"Downloading dataset from {REPO_ID}...")
        dataset = load_dataset(REPO_ID)

        console.print(f"\n[green]✅ Dataset loaded successfully![/green]")
        console.print(f"Splits found: {list(dataset.keys())}")

        # Check features
        table = Table(title="Dataset Features Audit")
        table.add_column("Split", style="cyan")
        table.add_column("Rows", style="magenta")
        table.add_column("Features", style="green")
        table.add_column("Image Sample", style="yellow")

        for split_name, split_data in dataset.items():
            features = list(split_data.features.keys())
            num_rows = len(split_data)

            image_sample = "N/A"
            if "images" in split_data.features:
                sample = split_data[0].get("images")
                img_obj = _first_image_from_field(sample)
                if img_obj:
                    # If images field contains multiple images, report the count
                    if isinstance(sample, (list, tuple)):
                        image_sample = f"{len(sample)} images in 1st row"
                    elif isinstance(sample, dict):
                        image_sample = f"{len(sample.keys())} images in 1st row"

            table.add_row(
                split_name,
                str(num_rows),
                ", ".join(features[:5]) + ("..." if len(features) > 5 else ""),
                image_sample
            )

        console.print(table)

        # Verify specific content
        if "train" in dataset:
            first_row = dataset["train"][0]
            console.print(f"\n[bold]Sample Content Audit (Train Split, Row 0):[/bold]")
            console.print(f"- Doc ID: {first_row.get('doc_id')}")
            console.print(f"- Title: {first_row.get('title')}")
            images = first_row.get('images', [])
            img_obj = _first_image_from_field(images)
            img_count = 0
            if isinstance(images, (list, tuple)):
                img_count = len(images)
            elif isinstance(images, dict):
                img_count = len(images.keys())
            console.print(f"- Image count: {img_count}")

            # Check for AI captions
            if img_obj:
                caption = img_obj.get("caption", "No caption found")
                console.print(f"- Sample Caption: [italic]{caption}[/italic]")
            else:
                console.print("- Sample Caption: [italic]No image/caption available[/italic]")

        console.print("\n[bold green]🌟 HF Compliance Audit Passed![/bold green]")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]❌ Verification failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_dataset()
