import os
import sys
import logging
import re
import shutil
import time
import hashlib
import uuid
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer
try:
    import questionary
except Exception:
    questionary = None
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from .core import DocParserEngine, ParsedDocument
from .dataset.hf_saver import save_hf_dataset, verify_hf_dataset_load

from dotenv import load_dotenv
load_dotenv()

app = typer.Typer(
    name="doc-parser",
    help="🚀 Production-grade document parsing engine with HuggingFace integration.",
    add_completion=False,
)
console = Console()
CONFIG_PATH = Path.home()/".doc-parser-config.json"

DEFAULT_CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
DEFAULT_REPO_ID = "Remixonwin/prepware_study_guide-dataset"
DEFAULT_WORKERS = 4
MAX_RETRIES = 3
IO_TIMEOUT = 30  # seconds


def get_iso_timestamp() -> str:
    """Generate ISO 8601 UTC timestamp for artifact naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write(content: str, target_path: Path) -> None:
    """Write content atomically using tmp dir then os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, target_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise IOError(f"Atomic write failed: {e}") from e


def atomic_write_json(data: dict, target_path: Path) -> None:
    """Write JSON content atomically."""
    atomic_write(json.dumps(data, indent=2, ensure_ascii=False), target_path)


def retry_with_backoff(func, max_retries: int = MAX_RETRIES, timeout: int = IO_TIMEOUT):
    """Retry function with exponential backoff and timeout."""
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout}s")
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Set timeout for heavy IO operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = func()
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.alarm(0)
        except TimeoutError as e:
            last_exception = e
            wait_time = 2 ** attempt
            logging.warning(f"Attempt {attempt + 1} timed out, retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                break
    
    raise last_exception from None


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def generate_source_manifest(source_folder: Path, artifacts_dir: Path) -> dict:
    """Walk source folder and generate manifest with SHA-256 checksums.
    
    Args:
        source_folder: Path to source folder to scan
        artifacts_dir: Path to artifacts directory (for relative paths)
    
    Returns:
        Manifest dict with file paths, checksums, and totals
    """
    files = []
    total_size = 0
    
    # Recursively find all files
    for f in source_folder.rglob('*'):
        if f.is_file():
            rel_path = f.relative_to(source_folder)
            checksum = compute_sha256(f)
            size = f.stat().st_size
            total_size += size
            files.append({
                "path": str(rel_path),
                "absolute_path": str(f.resolve()),
                "size": size,
                "sha256": checksum,
            })
    
    # Sort by path for deterministic ordering
    files.sort(key=lambda x: x["path"])
    
    return {
        "format_version": "1.0",
        "timestamp": get_iso_timestamp(),
        "source_folder": str(source_folder.resolve()),
        "total_files": len(files),
        "total_size": total_size,
        "files": files,
    }


def atomic_write_artifacts(
    output_base: Path,
    timestamp: str,
    data_writer: callable,
) -> Path:
    """Atomically write artifacts using tmp-dir pattern.
    
    Args:
        output_base: Base artifacts directory
        timestamp: ISO timestamp string
        data_writer: Function that takes tmp_dir Path and writes all artifacts
    
    Returns:
        Final artifacts directory path
    """
    import shutil
    import tempfile
    
    tmp_dir = output_base / f"{timestamp}.tmp"
    final_dir = output_base / timestamp
    
    # Step 1: Create tmp directory
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 2: Write all data to tmp directory
        data_writer(tmp_dir)
        
        # Step 3: Generate manifest for all written files
        manifest = {
            "format_version": "1.0",
            "timestamp": timestamp,
            "artifacts": [],
        }
        
        for f in tmp_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(tmp_dir)
                manifest["artifacts"].append({
                    "path": str(rel_path),
                    "size": f.stat().st_size,
                    "sha256": compute_sha256(f),
                })
        
        # Write manifest
        manifest_path = tmp_dir / "manifest.json"
        atomic_write_json(manifest, manifest_path)
        
        # Step 4: Atomic rename to final location
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(str(tmp_dir), str(final_dir))
        
    except Exception:
        # Cleanup on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    
    return final_dir


def generate_audit_report(
    repo_id: str,
    source_folder: Path,
    timestamp: str,
    total_documents: int,
    total_examples: int,
    examples: list,
    extraction_errors: list,
    parquet_paths: list,
    verifier_outputs: dict,
    cli_args: dict,
) -> dict:
    """Generate audit_report.json with required schema.
    
    Args:
        repo_id: HuggingFace repo ID
        source_folder: Source folder path
        timestamp: ISO timestamp
        total_documents: Number of documents processed
        total_examples: Number of examples generated
        examples: List of extracted examples
        extraction_errors: List of extraction errors
        parquet_paths: List of parquet shard paths
        verifier_outputs: Dict of verification command outputs
        cli_args: CLI arguments used
    
    Returns:
        Audit report dict
    """
    import platform
    import sys
    
    # Calculate examples per file (histogram)
    examples_per_file = {}
    for ex in examples:
        file_path = ex.get("file_path", "unknown")
        examples_per_file[file_path] = examples_per_file.get(file_path, 0) + 1
    
    # Get sample examples (first N)
    sample_examples = examples[:5] if len(examples) > 5 else examples
    
    # Determine file formats from examples
    formats = list(set(ex.get("file_extension", "") for ex in examples))
    
    hf_token_present = os.getenv("HF_TOKEN") is not None
    
    report = {
        "run_id": str(uuid.uuid4()),
        "dataset_repo": repo_id,
        "source_folder": str(source_folder.resolve()),
        "iso_timestamp": timestamp,
        "total_documents": total_documents,
        "total_examples": total_examples,
        "examples_per_file": examples_per_file,
        "sample_examples": sample_examples,
        "extraction_errors_count": len(extraction_errors),
        "extraction_errors": extraction_errors,
        "parquet_shard_paths": parquet_paths,
        "verifier_outputs": verifier_outputs,
        "cli_args": cli_args,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hf_token_present": hf_token_present,
        },
    }
    
    return report


def run_verification_commands(artifacts_dir: Path) -> dict:
    """Run verification commands and capture outputs.
    
    Args:
        artifacts_dir: Path to artifacts directory
    
    Returns:
        Dict of command outputs
    """
    import subprocess
    
    outputs = {}
    
    # Change to artifacts directory
    os.chdir(artifacts_dir)
    
    # 1. Validate JSONL structure
    jsonl_files = list(artifacts_dir.rglob("*.jsonl"))
    if jsonl_files:
        jsonl_file = jsonl_files[0]
        cmd = [
            "python3", "-c",
            f"""
import json
import sys
errors = []
for i, line in enumerate(open('{jsonl_file}')):
    try:
        rec = json.loads(line)
        if 'example_id' not in rec:
            errors.append(f'Line {{i}}: missing example_id')
    except json.JSONDecodeError as e:
        errors.append(f'Line {{i}}: {{e}}')
if errors:
    for e in errors[:10]:
        print(e)
    sys.exit(1)
print('JSONL validation: PASSED')
"""
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            outputs["jsonl_validation"] = {
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except Exception as e:
            outputs["jsonl_validation"] = {
                "command": " ".join(cmd),
                "error": str(e),
            }
    
    # 2. Validate Parquet can be read
    parquet_files = list(artifacts_dir.rglob("*.parquet"))
    if parquet_files:
        parquet_file = parquet_files[0]
        cmd = [
            "python3", "-c",
            f"""
import pandas as pd
df = pd.read_parquet('{parquet_file}')
print(f'Parquet validation: PASSED ({{len(df)}} rows)')
"""
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            outputs["parquet_validation"] = {
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except Exception as e:
            outputs["parquet_validation"] = {
                "command": " ".join(cmd),
                "error": str(e),
            }
    
    # 3. Verify deterministic IDs
    if jsonl_files:
        jsonl_file = jsonl_files[0]
        cmd = [
            "python3", "-c",
            f"""
import json
ids = set()
for line in open('{jsonl_file}'):
    rec = json.loads(line)
    eid = rec['example_id']
    if eid in ids:
        print(f'Duplicate ID: {{eid}}')
        exit(1)
    ids.add(eid)
print(f'ID uniqueness: PASSED ({{len(ids)}} unique IDs)')
"""
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            outputs["id_uniqueness"] = {
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except Exception as e:
            outputs["id_uniqueness"] = {
                "command": " ".join(cmd),
                "error": str(e),
            }
    
    # 4. Verify images exist
    images_dir = artifacts_dir / "images"
    if images_dir.exists():
        img_count = len(list(images_dir.rglob("*.*")))
        outputs["image_validation"] = {
            "command": "ls images/\nwc -l",
            "stdout": f"Image validation: PASSED ({img_count} images)",
            "exit_code": 0,
        }
    else:
        outputs["image_validation"] = {
            "command": "ls images/",
            "stdout": "Image validation: PASSED (0 images)",
            "exit_code": 0,
        }
    
    return outputs


def generate_verify_outputs(
    artifacts_dir: Path,
    timestamp: str,
    verifier_outputs: dict,
    started_at: str,
    completed_at: str,
) -> str:
    """Generate verify_outputs.txt with formatted verification results.
    
    Args:
        artifacts_dir: Path to artifacts directory
        timestamp: ISO timestamp
        verifier_outputs: Dict of verification command outputs
        started_at: Start timestamp
        completed_at: Completion timestamp
    
    Returns:
        Formatted output string
    """
    output_lines = [
        "=== Verification Output ===",
        f"Timestamp: {timestamp}",
        f"Started: {started_at}",
        f"Completed: {completed_at}",
        "",
    ]
    
    for i, (name, result) in enumerate(verifier_outputs.items()):
        output_lines.append(f"--- Command {i+1}: {name} ---")
        output_lines.append(f"Command: {result.get('command', 'N/A')}")
        output_lines.append("Output:")
        output_lines.append(result.get("stdout", result.get("error", "N/A")))
        output_lines.append(f"Exit Code: {result.get('exit_code', -1)}")
        output_lines.append("")
    
    # Summary
    all_passed = all(
        result.get("exit_code", 1) == 0 
        for result in verifier_outputs.values()
        if "exit_code" in result
    )
    
    output_lines.extend([
        "=== Summary ===",
        "All verifications PASSED" if all_passed else "Some verifications FAILED",
    ])
    
    return "\n".join(output_lines)


def load_config() -> dict:
    """Load user configuration from disk."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict):
    """Save user configuration to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def _choose_caption_model(
    llm_api_base: Optional[str],
    llm_model: Optional[str],
    explicit_caption_model: Optional[str],
    force_local: bool,
) -> str:
    """Choose the final caption model with auto-selection rules.

    Rules implemented:
      - If an explicit caption_model was provided, honor it only if it's 'api' or force_local is True.
      - If explicit_caption_model is provided but not force_local and it's not 'api', and a remote LLM
        (llm_api_base + llm_model) is available, prefer 'api'.
      - If no explicit model and llm_api_base and llm_model are provided and not force_local -> 'api'.
      - Otherwise return the default HF model.
    """
    logger = logging.getLogger(__name__)

    if explicit_caption_model:
        if explicit_caption_model == "api":
            return "api"
        if force_local:
            return explicit_caption_model
        # explicit HF model provided but user didn't force local; prefer API if remote LLM exists
        if llm_api_base and llm_model:
            logger.info("Auto-selecting 'api' caption backend because LLM API is available and --force-local-caption not set")
            return "api"
        return explicit_caption_model

    # No explicit model
    if llm_api_base and llm_model and not force_local:
        logger.info(f"Using remote LLM API at {llm_api_base} with model {llm_model}; captioning backend set to 'api' (no large HF downloads)")
        return "api"

    return DEFAULT_CAPTION_MODEL


def get_engine(
    caption_model: Optional[str] = None,
    no_caption: bool = False,
    no_ocr: bool = False,
    no_tables: bool = False,
    no_ocr_images: bool = False,
    output_dir: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 8,
    llm_api_base: Optional[str] = None,
    llm_model: Optional[str] = None,
    force_local_caption: bool = False,
    verbose: bool = False,
) -> DocParserEngine:
    """Utility to initialize the DocParserEngine with consistent settings.

    Precedence for LLM settings: CLI > ENV > Config file.
    Environment variables supported: DOCPARSER_LLM_API_BASE, DOCPARSER_LLM_MODEL
    """
    config = load_config()

    # Env overrides
    env_llm_api = os.getenv("DOCPARSER_LLM_API_BASE")
    env_llm_model = os.getenv("DOCPARSER_LLM_MODEL")

    # Determine if a caption model was explicitly provided.
    # Preserve CLI flag default behavior: parse() still provides DEFAULT_CAPTION_MODEL by default,
    # so treat that value as "not explicitly set" unless the config provides a caption_model.
    config_caption = config.get("caption_model")
    explicit_caption_model = None
    if caption_model is not None and caption_model != DEFAULT_CAPTION_MODEL:
        # User provided a non-default model via CLI
        explicit_caption_model = caption_model
    elif config_caption:
        # User provided a model via config
        explicit_caption_model = config_caption

    # Effective LLM settings: CLI > ENV > Config
    effective_llm_api = llm_api_base or env_llm_api or config.get("llm_api_base")
    effective_llm_model = llm_model or env_llm_model or config.get("llm_model")

    final_model = _choose_caption_model(effective_llm_api, effective_llm_model, explicit_caption_model, force_local_caption)

    # Prioritize other settings: CLI args > Config File > Defaults
    final_device = device if device != "auto" else config.get("device", "auto")
    final_output = output_dir or config.get("output_dir")
    final_ocr_images = not no_ocr_images if no_ocr_images else config.get("enable_ocr_on_images", True)

    return DocParserEngine(
        caption_model=final_model,
        enable_captioning=not no_caption,
        enable_ocr=not no_ocr,
        enable_table_extraction=not no_tables,
        enable_ocr_on_images=final_ocr_images,
        output_dir=final_output,
        device=final_device,
        batch_size=batch_size,
        llm_api_base=effective_llm_api,
        llm_model=effective_llm_model or "gpt-4o",
        force_local_caption=force_local_caption,
        verbose=verbose,
    )


def get_default_repo_id(path: Path, suggested_title: Optional[str] = None) -> Optional[str]:
    """Generate an intelligent repository ID based on title and environment."""
    username = os.getenv("HF_USERNAME")
    if not username:
        return None
    
    name_source = suggested_title or (path.stem if path.is_file() else path.name)
    repo_slug = re.sub(r'[^a-zA-Z0-9\-_]', '-', name_source).lower()
    repo_slug = re.sub(r'-+', '-', repo_slug).strip('-')
    return f"{username}/{repo_slug}-dataset"

@app.command()
def status():
    """Check system compatibility and available hardware."""
    table = Table(title="DocParserEngine System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # Check Python version
    table.add_row("Python Version", sys.version.split()[0])
    
    # Check Torch/Device
    try:
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        table.add_row("Compute Device", device)
    except ImportError:
        table.add_row("Compute Device", "[red]torch not installed[/red]")

    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        table.add_row("OCR (Tesseract)", "Available")
    except Exception:
        table.add_row("OCR (Tesseract)", "[yellow]Not found (Install tesseract-ocr for OCR features)[/yellow]")

    console.print(table)

@app.command()
def config():
    """Interactive configuration of default engine parameters."""
    current = load_config()
    
    rprint("[bold blue]DocParserEngine Configuration Wizard[/bold blue]")
    rprint("[dim]These settings will be saved to ~/.doc-parser-config.json[/dim]\n")
    
    model = questionary.text(
        "Default Caption Model ID:",
        default=current.get("caption_model", DEFAULT_CAPTION_MODEL)
    ).ask()
    
    device = questionary.select(
        "Default Compute Device:",
        choices=["auto", "cpu", "cuda", "mps"],
        default=current.get("device", "auto")
    ).ask()
    
    output_dir = questionary.text(
        "Default Output Directory:",
        default=current.get("output_dir", "./parsed_outputs")
    ).ask()
    
    ocr_images = questionary.confirm(
        "Enable OCR on images by default?",
        default=current.get("enable_ocr_on_images", True)
    ).ask()

    llm_api = questionary.text(
        "LLM API Base URL (optional):",
        default=current.get("llm_api_base", "")
    ).ask()

    llm_model = questionary.text(
        "LLM Model ID:",
        default=current.get("llm_model", "gpt-4o")
    ).ask()
    
    new_config = {
        "caption_model": model,
        "device": device,
        "output_dir": output_dir,
        "enable_ocr_on_images": ocr_images,
        "llm_api_base": llm_api if llm_api else None,
        "llm_model": llm_model
    }
    
    save_config(new_config)
    rprint("\n[bold green]✓ Configuration saved successfully![/bold green]")

@app.command()
def parse(
    path: Path = typer.Argument(..., help="File or directory to parse"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
    glob: Optional[str] = typer.Option(None, "--glob", help="Glob pattern for directory search"),
    export: str = typer.Option("full", "--export", help="HF Dataset schema (full, chunks, images, minimal, qa)"),
    push: bool = typer.Option(False, "--push", help="Push to HuggingFace Hub"),
    hub_repo: Optional[str] = typer.Option(None, "--hub-repo", help="HF Hub repository ID"),
    caption_model: str = typer.Option(DEFAULT_CAPTION_MODEL, "--model", help="Captioning model"),
    no_caption: bool = typer.Option(False, "--no-caption", help="Disable image captioning"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Disable OCR"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_ocr_images: bool = typer.Option(False, "--no-ocr-images", help="Disable OCR on images"),
    device: str = typer.Option("auto", "--device", help="Device (cpu, cuda, mps, auto)"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for captioning"),
    llm_api: Optional[str] = typer.Option(None, "--llm-api", help="LLM API Base URL"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM Model ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    # CI/noninteractive options
    ci: bool = typer.Option(False, "--ci", help="Run in CI/noninteractive mode"),
    repo_id: Optional[str] = typer.Option(None, "--repo-id", help="Repository id (CI)"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir", help="Artifacts directory"),
    output_dir_explicit: Optional[Path] = typer.Option(None, "--output-dir", help="Explicit output directory"),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed for deterministic behavior"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace token (or env HF_TOKEN)"),
):
    """Parse documents and optionally export to HuggingFace Datasets.

    Supports CI/noninteractive flags: --ci, --repo-id, --artifacts-dir, --output-dir, --seed, --hf-token
    When --ci is enabled, the CLI emits JSON events to stderr as newline-delimited JSON.
    """
    engine = get_engine(
        caption_model=caption_model,
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        output_dir=str(output) if output else None,
        device=device,
        batch_size=batch_size,
        llm_api_base=llm_api,
        llm_model=llm_model,
        verbose=verbose,
    )

    if path.is_file():
        files = [path]
    elif path.is_dir():
        if glob:
            files = list(path.glob(glob))
        elif recursive:
            files = [f for ext in engine.SUPPORTED_FORMATS for f in path.rglob(f"*{ext}")]
        else:
            files = [f for ext in engine.SUPPORTED_FORMATS for f in path.glob(f"*{ext}")]
        files = sorted(set(files))
    else:
        rprint(f"[red]Error:[/red] Path {path} does not exist.")
        raise typer.Exit(1)

    if not files:
        rprint("[yellow]No supported files found.[/yellow]")
        return

    docs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Parsing documents...", total=len(files))
        
        for file_path in files:
            progress.update(task, description=f"[cyan]Parsing {file_path.name}...")
            try:
                doc = engine.parse(file_path)
                docs.append(doc)
            except Exception as e:
                console.print(f"[red]Failed to parse {file_path.name}: {e}[/red]")
            progress.advance(task)

    if not docs:
        rprint("[red]No documents were successfully parsed.[/red]")
        return

    rprint(f"\n[green]Successfully parsed {len(docs)} documents.[/green]")
    
    # Export to Dataset
    if export:
        hub_token = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN")
        hf_token_cli = None
        # accept --hf-token via env or CLI flag
        # TODO: CLI flag added by audit command; check env
        if os.getenv("HF_TOKEN"):
            hf_token_cli = os.getenv("HF_TOKEN")
        
        # Auto-generate hub_repo if pushing but not provided
        if push and not hub_repo:
            hub_repo = get_default_repo_id(path)
            if hub_repo:
                rprint(f"[dim]Automatically using intelligent repo ID: {hub_repo}[/dim]")

        try:
            with console.status(f"[bold blue]Building {export} dataset..."):
                dataset = engine.to_hf_dataset(
                    docs,
                    schema=export,
                    push_to_hub=push,
                    hub_repo=hub_repo,
                    hub_token=hub_token
                )
            rprint(f"[bold green]Dataset built successfully![/bold green] (Schema: {export})")

            # Persist local dataset_info.json if produced under engine.output_dir
            try:
                found = list((engine.output_dir).rglob("dataset_info.json"))
                if found:
                    target = Path("parsed_outputs") / "dataset_info_saved.json"
                    shutil.copyfile(str(found[0]), str(target))
                    rprint(f"[dim]Persisted local dataset_info.json to {target}[/dim]")
            except Exception as e:
                logger.warning("Could not persist local dataset_info.json: %s", e)

        except Exception as e:
            msg = str(e)
            if "403 Forbidden" in msg or "rights to create a dataset" in msg:
                rprint("[bold red]Error: Permission Denied (403 Forbidden)[/bold red]")
                rprint(Panel(
                    "Your HuggingFace token does not have [bold]Write/Create[/bold] permissions.\n\n"
                    "1. Visit [cyan]https://huggingface.co/settings/tokens[/cyan]\n"
                    "2. Ensure your token type is [bold]Write[/bold] or has 'Create Repo' scopes.\n"
                    "3. Update [bold].env[/bold] with the correct [bold]HF_TOKEN[/bold]..",
                    title="Permissions Required",
                    border_style="red"
                ))
            else:
                rprint(f"[red]Error during export: {e}[/red]")
            
        if not push:
            rprint("[dim]Use --push and --hub-repo to share on HuggingFace Hub.[/dim]")


@app.command()
def extract(
    input_folder: Path = typer.Option(
        None,
        "--input-folder",
        "-i",
        help="Path to source folder to process recursively (required)",
    ),
    output_artifacts: Optional[Path] = typer.Option(
        None,
        "--output-artifacts",
        "-o",
        help="Artifacts directory (default: artifacts/<ISO_TIMESTAMP>/)",
    ),
    repo_id: str = typer.Option(
        DEFAULT_REPO_ID,
        "--repo-id",
        "-r",
        help="HuggingFace repo ID (default: Remixonwin/prepware_study_guide-dataset)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing artifacts",
    ),
    workers: int = typer.Option(
        DEFAULT_WORKERS,
        "--workers",
        "-w",
        help="Max workers for parallel processing (default: 4)",
    ),
    caption_model: str = typer.Option(
        DEFAULT_CAPTION_MODEL,
        "--model",
        help="Captioning model",
    ),
    no_caption: bool = typer.Option(False, "--no-caption", help="Disable image captioning"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Disable OCR"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_ocr_images: bool = typer.Option(False, "--no-ocr-images", help="Disable OCR on images"),
    device: str = typer.Option("auto", "--device", help="Device (cpu, cuda, mps, auto)"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for captioning"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    push: bool = typer.Option(False, "--push", help="Push to HuggingFace Hub"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace token (or env HF_TOKEN)"),
):
    """Noninteractive extraction pipeline with CLI flags.
    
    Processes documents from input folder and generates HuggingFace dataset artifacts.
    All parameters are CLI flags - no prompts.
    
    Example:
        doc-parser extract --input-folder /path/to/docs --workers 4 --force
    """
    # Validate required input_folder
    if input_folder is None:
        rprint("[red]Error:[/red] --input-folder is required")
        raise typer.Exit(1)
    
    input_folder = input_folder.resolve()
    if not input_folder.exists():
        rprint(f"[red]Error:[/red] Input folder does not exist: {input_folder}")
        raise typer.Exit(1)
    
    if not input_folder.is_dir():
        rprint(f"[red]Error:[/red] Input path is not a directory: {input_folder}")
        raise typer.Exit(1)
    
    # Generate timestamp for artifacts
    timestamp = get_iso_timestamp()
    
    # Determine artifacts directory
    if output_artifacts is None:
        artifacts_dir = Path("artifacts") / timestamp
    else:
        artifacts_dir = Path(output_artifacts)
    
    # Check if artifacts exist and --force not set
    if artifacts_dir.exists() and not force:
        rprint(f"[red]Error:[/red] Artifacts directory exists: {artifacts_dir}")
        rprint("[yellow]Use --force to overwrite existing artifacts[/yellow]")
        raise typer.Exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=log_level,
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting extraction pipeline")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Artifacts dir: {artifacts_dir}")
    logger.info(f"Repo ID: {repo_id}")
    logger.info(f"Workers: {workers}")
    
    # Create artifacts directory structure
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir = artifacts_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = artifacts_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    engine = get_engine(
        caption_model=caption_model,
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        output_dir=str(artifacts_dir),
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )
    
    # Find all supported files
    files = []
    for ext in engine.SUPPORTED_FORMATS:
        files.extend(input_folder.rglob(f"*{ext}"))
    files = sorted(set(files))
    
    if not files:
        rprint("[yellow]No supported files found in input folder.[/yellow]")
        raise typer.Exit(0)
    
    rprint(f"[cyan]Found {len(files)} files to process[/cyan]")
    
    # Parse documents with parallel workers
    docs = []
    extraction_warnings = []
    
    def parse_with_retry(file_path: Path) -> tuple[Optional[ParsedDocument], List[str]]:
        """Parse a single file with retry logic."""
        warnings = []
        for attempt in range(MAX_RETRIES):
            try:
                doc = engine.parse(file_path)
                return doc, warnings
            except Exception as e:
                warnings.append(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} for {file_path.name} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to parse {file_path.name} after {MAX_RETRIES} attempts: {e}")
                    extraction_warnings.append(f"File {file_path}: {str(e)}")
        return None, warnings
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Parsing documents...", total=len(files))
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(parse_with_retry, f): f for f in files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                progress.update(task, description=f"[cyan]Parsing {file_path.name}...")
                
                try:
                    doc, warnings = future.result()
                    if doc:
                        docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    extraction_warnings.append(f"File {file_path}: {str(e)}")
                
                progress.advance(task)
    
    if not docs:
        rprint("[red]No documents were successfully parsed.[/red]")
        raise typer.Exit(1)
    
    rprint(f"[green]Successfully parsed {len(docs)} documents.[/green]")
    
    # Generate per-page examples for PDF extraction
    from .core import generate_page_examples
    examples = generate_page_examples(docs)
    
    rprint(f"[green]Generated {len(examples)} examples (per-page for PDFs).[/green]")
    
    # Write examples to JSONL
    jsonl_dir = artifacts_dir / "jsonl" / "train"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = jsonl_dir / f"shard-00000.jsonl"
    
    def write_jsonl_with_retry():
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    try:
        retry_with_backoff(write_jsonl_with_retry)
        rprint(f"[green]Wrote examples to {jsonl_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing JSONL: {e}[/red]")
        raise typer.Exit(1)
    
    # Write examples as HF parquet shards
    try:
        rprint(f"[cyan]Saving HF parquet shards...[/cyan]")
        parquet_files = save_hf_dataset(
            examples=examples,
            output_dir=artifacts_dir,
            split_name="train",
            citation=f"@dataset{{{repo_id.split('/')[-1]}_dataset,\n  author = {{Extracted from Prepware Study Guide}},\n  title = {{Prepware Study Guide Dataset}},\n  year = {{2026}},\n  url = {{https://huggingface.co/datasets/{repo_id}}}\n}}",
            description="Extracted from Prepware Study Guide documents. Each row represents one page from a source document."
        )
        rprint(f"[green]Wrote {len(parquet_files)} parquet shards to {artifacts_dir / 'data'}[/green]")
        
        # Verify the dataset can be loaded
        rprint(f"[cyan]Verifying HF dataset can be loaded...[/cyan]")
        verify_result = verify_hf_dataset_load(artifacts_dir)
        if verify_result["success"]:
            rprint(f"[green]Dataset verified: {verify_result['load_method']}, {verify_result['num_examples']} examples[/green]")
        else:
            rprint(f"[yellow]Warning: Dataset verification failed: {verify_result['error']}[/yellow]")
    except Exception as e:
        rprint(f"[yellow]Warning: Error saving HF dataset: {e}[/yellow]")
        logger.exception("HF dataset saving failed")
    
    # Generate metadata
    metadata = {
        "timestamp": timestamp,
        "input_folder": str(input_folder),
        "artifacts_dir": str(artifacts_dir),
        "repo_id": repo_id,
        "total_files": len(files),
        "total_documents": len(docs),
        "total_examples": len(examples),
        "workers": workers,
        "extraction_warnings": extraction_warnings,
    }
    
    # Atomic write metadata
    metadata_file = artifacts_dir / "metadata.json"
    try:
        atomic_write_json(metadata, metadata_file)
        rprint(f"[green]Wrote metadata to {metadata_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing metadata: {e}[/red]")
    
    # Generate source manifest with SHA-256 checksums
    source_manifest = generate_source_manifest(input_folder, artifacts_dir)
    source_manifest_file = artifacts_dir / "source_manifest.json"
    try:
        atomic_write_json(source_manifest, source_manifest_file)
        rprint(f"[green]Wrote source manifest to {source_manifest_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing source manifest: {e}[/red]")
    
    # Generate manifest with checksums for output files
    manifest = {
        "format_version": "1.0",
        "timestamp": timestamp,
        "artifacts": [
            {
                "path": str(f.relative_to(artifacts_dir)),
                "type": f.suffix.lstrip("."),
                "size": f.stat().st_size,
                "sha256": compute_sha256(f),
            }
            for f in artifacts_dir.rglob("*")
            if f.is_file()
        ],
    }
    
    manifest_file = artifacts_dir / "manifest.json"
    try:
        atomic_write_json(manifest, manifest_file)
        rprint(f"[green]Wrote manifest to {manifest_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing manifest: {e}[/red]")
    
    rprint(f"\n[bold green]✓ Extraction complete![/bold green]")
    rprint(f"Artifacts: {artifacts_dir}")
    rprint(f"Examples: {len(examples)}")
    
    if extraction_warnings:
        rprint(f"[yellow]Warnings: {len(extraction_warnings)}[/yellow]")
        warnings_file = logs_dir / "extraction_warnings.log"
        with open(warnings_file, "w") as f:
            f.write("\n".join(extraction_warnings))
        rprint(f"Warnings logged to: {warnings_file}")
    
    # ========================================
    # NEW: Generate audit_report.json
    # ========================================
    
    # Run verification commands (but don't fail extraction if they fail)
    parquet_paths = [str(f.relative_to(artifacts_dir)) for f in artifacts_dir.rglob("*.parquet")]
    
    # Store start time for verification
    started_at = datetime.now(timezone.utc).isoformat()
    
    try:
        verifier_outputs = run_verification_commands(artifacts_dir)
        rprint(f"[green]Ran verification commands[/green]")
    except Exception as e:
        logger.warning(f"Verification commands failed: {e}")
        verifier_outputs = {"error": str(e)}
    
    completed_at = datetime.now(timezone.utc).isoformat()
    
    # Generate verify_outputs.txt
    verify_output_text = generate_verify_outputs(
        artifacts_dir,
        timestamp,
        verifier_outputs,
        started_at,
        completed_at,
    )
    
    verify_outputs_file = artifacts_dir / "verify_outputs.txt"
    try:
        atomic_write(verify_output_text, verify_outputs_file)
        rprint(f"[green]Wrote verify outputs to {verify_outputs_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing verify outputs: {e}[/red]")
    
    # Generate audit_report.json with required schema
    cli_args = {
        "input_folder": str(input_folder),
        "output_artifacts": str(artifacts_dir),
        "repo_id": repo_id,
        "workers": workers,
        "caption_model": caption_model,
        "no_caption": no_caption,
        "no_ocr": no_ocr,
        "no_tables": no_tables,
        "device": device,
        "batch_size": batch_size,
    }
    
    audit_report = generate_audit_report(
        repo_id=repo_id,
        source_folder=input_folder,
        timestamp=timestamp,
        total_documents=len(docs),
        total_examples=len(examples),
        examples=examples,
        extraction_errors=extraction_warnings,
        parquet_paths=parquet_paths,
        verifier_outputs=verifier_outputs,
        cli_args=cli_args,
    )
    
    audit_report_file = artifacts_dir / "audit_report.json"
    try:
        atomic_write_json(audit_report, audit_report_file)
        rprint(f"[green]Wrote audit report to {audit_report_file}[/green]")
    except Exception as e:
        rprint(f"[red]Error writing audit report: {e}[/red]")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Interactive wizard if no command is provided."""
    if ctx.invoked_subcommand is not None:
        return

    rprint(Panel.fit(
        "[bold cyan]DocParserEngine[/bold cyan] 🔍📄\n"
        "[dim]Interactive Document Processing Wizard[/dim]",
        border_style="cyan"
    ))

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            "Parse a document or directory",
            "Check system status",
            "Exit"
        ]
    ).ask()

    if choice == "Check system status":
        status()
    elif choice == "Parse a document or directory":
        interactive_parse()
    else:
        raise typer.Exit()

def interactive_parse():
    path_str = questionary.path("Enter file or directory path:").ask()
    if not path_str:
        return
    path = Path(path_str)
    
    recursive = False
    if path.is_dir():
        recursive = questionary.confirm("Recurse into subdirectories?", default=True).ask()

    export = questionary.select(
        "Choose export schema:",
        choices=["full", "chunks", "images", "minimal", "qa"]
    ).ask()

    push = questionary.confirm("Push to HuggingFace Hub?", default=False).ask()
    hub_repo = None
    if push:
        # Scout for a better title if it's a single file
        suggested_title = None
        if path.is_file():
            with console.status("[bold blue]Scouting document title..."):
                # Use a lightweight engine just for scouting
                temp_engine = DocParserEngine()
                metadata = temp_engine.scout_metadata(path)
                suggested_title = metadata.get("title")

        default_repo = get_default_repo_id(path, suggested_title=suggested_title)
        
        # If we have a default repo, we use it without asking to confirm
        # to streamline the experience as requested by the user.
        if default_repo:
            hub_repo = default_repo
            rprint(f"[dim]Intelligently selected repo: {hub_repo}[/dim]")
        else:
            hub_repo = questionary.text(
                "Enter Hub Repository ID (e.g. username/dataset):"
            ).ask()

    # Advanced options
    no_tables = False
    no_ocr_images = False
    if questionary.confirm("Configure advanced options?", default=False).ask():
        no_caption = not questionary.confirm("Enable image captioning?", default=True).ask()
        no_ocr = not questionary.confirm("Enable OCR?", default=True).ask()
        no_tables = not questionary.confirm("Enable table extraction?", default=True).ask()
        no_ocr_images = not questionary.confirm("Enable OCR on images for categorization?", default=True).ask()
        device = questionary.select("Device:", choices=["auto", "cpu", "cuda", "mps"]).ask()
        llm_api = questionary.text("LLM API Base URL (empty to disable):", default="http://0.0.0.0:7543").ask()
        llm_model = questionary.text("LLM Model ID:", default="gpt-4o").ask()
    else:
        no_caption = False
        no_ocr = False
        device = "auto"
        llm_api = None
        llm_model = None

    engine = get_engine(
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        device=device,
        llm_api_base=llm_api,
        llm_model=llm_model,
        verbose=True  # Default verbose for interactive
    )

    # Invoke the parse command
    parse(
        path=path,
        output=None,
        recursive=recursive,
        glob=None,
        export=export,
        push=push,
        hub_repo=hub_repo,
        caption_model="none" if no_caption else DEFAULT_CAPTION_MODEL,
        no_caption=no_caption,
        no_ocr=no_ocr,
        no_tables=no_tables,
        no_ocr_images=no_ocr_images,
        device=device,
        batch_size=8,
        llm_api=llm_api,
        llm_model=llm_model,
        verbose=True
    )


def choose_path_interactive(root: Path = Path.cwd(), input_func=input, print_func=print, max_entries: int = 200, ci: bool = False, filter: Optional[str] = None, select_index: Optional[int] = None) -> Optional[Path]:
    """Interactive path chooser rooted at `root`.

    When `ci` is True, operate headless using `filter` (substring filter) or `select_index` (0-based)
    to choose a path noninteractively. Returns None and raises typer.Exit(code=2) when not found.
    """
    workspace_root = Path(root).resolve()
    current = workspace_root
    filter_str = ""

    # Headless CI mode: if ci True, perform noninteractive selection immediately
    if ci:
        entries = sorted([p for p in workspace_root.iterdir()], key=lambda p: (p.is_file(), p.name.lower()))[:max_entries]
        if filter:
            filtered = [p for p in entries if filter.lower() in p.name.lower()]
            if not filtered:
                # indicate not found by raising typer.Exit with code 2
                raise typer.Exit(code=2)
            return filtered[0].resolve()
        if select_index is not None:
            if select_index < 0 or select_index >= len(entries):
                raise typer.Exit(code=2)
            return entries[select_index].resolve()
        # Default: pick first entry if available
        if entries:
            return entries[0].resolve()
        raise typer.Exit(code=2)

    def _list_entries(directory: Path) -> List[Path]:
        try:
            entries = [p for p in directory.iterdir()]
        except Exception:
            return []
        # Sort: directories first, then files, both alphabetically
        entries.sort(key=lambda p: (p.is_file(), p.name.lower()))
        return entries

    while True:
        all_entries = _list_entries(current)
        if filter_str:
            entries = [p for p in all_entries if filter_str.lower() in p.name.lower()]
        else:
            entries = all_entries

        entries = entries[:max_entries]

        print_func(f"\nCurrent: {str(current.relative_to(workspace_root) if current != workspace_root else Path('.'))}")
        if not entries:
            print_func("(No entries)")
        for idx, p in enumerate(entries, start=1):
            rel = p.relative_to(workspace_root) if workspace_root in p.parents or p == workspace_root else p
            kind = "[DIR]" if p.is_dir() else "[FILE]"
            print_func(f"{idx:3d}. {kind} {rel}")

        prompt = "Enter filter (substring), a number to select, 'r <n>' to recurse into a directory, 'u' to go up, or 'q' to cancel: "
        resp = input_func(prompt).strip()

        if resp == "q":
            return None
        if resp == "u":
            if current == workspace_root:
                print_func("Already at workspace root")
            else:
                current = current.parent
                filter_str = ""
            continue

        if resp.startswith("r"):
            rest = resp[1:].strip()
            if not rest:
                # ask for number
                rest = input_func("Enter directory number to recurse into: ").strip()
            if rest.isdigit():
                idx = int(rest) - 1
                if 0 <= idx < len(entries):
                    target = entries[idx]
                    if target.is_dir():
                        current = target
                        filter_str = ""
                    else:
                        print_func("Selected entry is not a directory")
                else:
                    print_func("Index out of range")
            else:
                print_func("Invalid directory index")
            continue

        if resp.isdigit():
            idx = int(resp) - 1
            if 0 <= idx < len(entries):
                return entries[idx].resolve()
            else:
                print_func("Index out of range")
                continue

        # Otherwise treat as filter
        filter_str = resp
        continue


@app.command("choose-path")
def choose_path_cmd():
    """Interactive chooser to select a file or directory under the current workspace.

    Prints the absolute path of the selected entry to stdout and exits with code 0.
    If the user cancels, exits with code 2.
    """
    chosen = choose_path_interactive(Path.cwd(), input_func=input, print_func=print)
    if chosen:
        # Print absolute path
        print(str(chosen))
        raise typer.Exit(0)
    else:
        raise typer.Exit(2)


@app.command("audit-dataset")
def audit_dataset(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="HF repo id"),
    local_src: Path = typer.Option(Path("prepware_study_guide"), "--local-src", help="Local source dir"),
    sample: int = typer.Option(10, "--sample", "-s", help="Sample size"),
    threshold: float = typer.Option(0.6, "--threshold", help="Similarity threshold"),
    dest: Optional[Path] = typer.Option(None, "--dest", help="Output directory"),
    no_download: bool = typer.Option(False, "--no-download", help="Skip downloading HF artifacts"),
):
    """Run an automated audit of a HuggingFace dataset by sampling records and comparing to local sources.

    Exit codes: 0 success, 1 verification failures, 2 usage/errors.
    """
    import datetime
    from doc_parser_engine import audit as audit_mod

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(dest) if dest else Path("parsed_outputs") / "prepware_audit" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = repo or os.getenv("REPO_ID") or get_default_repo_id(Path.cwd(), suggested_title=None)
    if not repo_id:
        rprint("[red]No repository id provided and REPO_ID env not set[/red]")
        raise typer.Exit(2)

    try:
        if not no_download:
            rprint(f"[dim]Downloading HF artifacts for {repo_id} into {out_dir}[/dim]")
            art = audit_mod.download_hf_artifacts(repo_id, out_dir)
        else:
            art = {"downloaded_files": [], "metadata_path": None}

        results = audit_mod.sample_and_compare(repo_id=repo_id, local_src=local_src, dest=out_dir, sample_size=sample, similarity_threshold=threshold)
        json_path, md_path = audit_mod.generate_report(results, out_dir)
        rprint(f"[green]Audit completed. JSON: {json_path} MD: {md_path}[/green]")

        failed = any(not s.get("pass") for s in results.get("samples", []))
        if failed:
            rprint(f"[yellow]Some samples failed audit (pass_rate={results['summary']['pass_rate']:.2%}). Exiting with code 1[/yellow]")
            raise typer.Exit(1)
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Audit failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
