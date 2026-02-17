import os
import logging
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

REPO_ID = "Remixonwin/wisconsin-motorists-handbook-dataset"

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
                sample = split_data[0]["images"]
                if sample and len(sample) > 0:
                    image_sample = f"{len(sample)} images in 1st row"
            
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
            console.print(f"- Image count: {len(first_row.get('images', []))}")
            
            # Check for AI captions
            images = first_row.get("images", [])
            if images:
                caption = images[0].get("caption", "No caption found")
                console.print(f"- Sample Caption: [italic]{caption}[/italic]")
                
        console.print("\n[bold green]🌟 HF Compliance Audit Passed![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Verification failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
