from __future__ import annotations

# Usage: PYTHONPATH=. python3 -m src.doc_parser_engine.audit --dataset REPO_ID --output artifacts/<ISO_TIMESTAMP> --sample N --no-download --seed 42

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import load_dataset, DownloadConfig
from huggingface_hub import HfApi, hf_hub_download
from difflib import SequenceMatcher
import os
import re
import tempfile
import shutil
import hashlib
import platform
import sys
from datetime import datetime

# Lazy import engine factory to avoid circular imports during test collection
def _get_engine():
    # Importing the CLI module here is intentional and lazy to prevent import-time side effects
    from doc_parser_engine.cli import get_engine as _g

    return _g()


# Basic logger for module-level messages; per-run handlers are configured in run_audit
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _retry(fn, retries=3, base_delay=1.0, exceptions=(Exception,), *args, **kwargs):
    """Retry helper with exponential backoff. Raises a RuntimeError on final failure."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            last_exc = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed: %s; retrying in %.1fs", attempt, e, delay)
            time.sleep(delay)
    logger.error("All %d retries failed: %s", retries, last_exc)
    raise RuntimeError(f"Operation failed after {retries} retries: {last_exc}")


def download_hf_artifacts(repo_id: str, dest: Path, hf_token: Optional[str] = None, no_download: bool = False) -> Dict:
    """Download README, dataset metadata, dataset_infos.json and parquet shards into dest.

    If no_download is True, network operations are skipped and a minimal metadata file is written.
    Returns dict with keys: downloaded_files (list), metadata_path (str)
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if no_download:
        logger.info("Skipping downloads for %s (no_download=True)", repo_id)
        meta_path = dest / "metadata.json"
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            json.dump({"repo_id": repo_id, "note": "download_skipped"}, tmp, indent=2)
            tmp.close()
            os.replace(tmp.name, str(meta_path))
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        return {"downloaded_files": [], "metadata_path": str(meta_path)}

    api = HfApi(token=hf_token) if hf_token else HfApi()

    def _list_files():
        return api.list_repo_files(repo_id)

    try:
        files = _retry(_list_files, retries=3, base_delay=1.0)
    except Exception as e:
        logger.error("Failed to list files for %s: %s", repo_id, e)
        raise RuntimeError(f"Could not list repository files for {repo_id}: {e}")

    downloaded = []

    # Prefer README files
    for candidate in ("README.md", "readme.md", "README", "README.MD"):
        if candidate in files:
            try:
                local = _retry(lambda: hf_hub_download(repo_id=repo_id, filename=candidate, token=hf_token), retries=3)
                dest_path = dest / Path(local).name
                # atomic move
                os.replace(local, str(dest_path))
                downloaded.append(str(dest_path))
                logger.info("Downloaded README -> %s", dest_path)
            except Exception as e:
                logger.warning("Failed to download README %s: %s", candidate, e)
            break

    # Save dataset metadata via HfApi.dataset_info if available
    meta_path = dest / "metadata.json"
    try:
        info = _retry(lambda: api.dataset_info(repo_id), retries=3)
        try:
            payload = info.to_dict() if hasattr(info, "to_dict") else dict(info)
        except Exception:
            payload = {"repo_id": repo_id, "note": "dataset_info_unserializable"}
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            json.dump(payload, tmp, indent=2)
            tmp.close()
            os.replace(tmp.name, str(meta_path))
            logger.info("Saved metadata.json to %s", meta_path)
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
    except Exception as e:
        logger.warning("Could not fetch dataset_info from HF: %s", e)
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            json.dump({"repo_id": repo_id, "note": "dataset_info not available"}, tmp, indent=2)
            tmp.close()
            os.replace(tmp.name, str(meta_path))
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    # Also attempt to download dataset_infos.json if listed
    if "dataset_infos.json" in files:
        try:
            local = _retry(lambda: hf_hub_download(repo_id=repo_id, filename="dataset_infos.json", token=hf_token), retries=3)
            target = dest / "dataset_infos.json"
            os.replace(local, str(target))
            downloaded.append(str(target))
            logger.info("Downloaded dataset_infos.json -> %s", target)
        except Exception as e:
            logger.warning("Failed to download dataset_infos.json: %s", e)

    # Download parquet shards (files ending with .parquet)
    parquet_files = [f for f in files if f.lower().endswith(".parquet") or ".parquet" in f.lower()]
    if not parquet_files:
        logger.info("No parquet files listed for repo %s", repo_id)
    for p in parquet_files:
        try:
            local = _retry(lambda: hf_hub_download(repo_id=repo_id, filename=p, token=hf_token), retries=3)
            target = data_dir / Path(p).name
            os.replace(local, str(target))
            downloaded.append(str(target))
            logger.info("Downloaded parquet -> %s", target)
        except Exception as e:
            logger.warning("Failed to download %s: %s", p, e)

    return {"downloaded_files": downloaded, "metadata_path": str(meta_path)}


def _jaccard(a: str, b: str) -> float:
    aw = set([w.lower() for w in str(a).split() if w.strip()])
    bw = set([w.lower() for w in str(b).split() if w.strip()])
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return float(len(aw & bw)) / float(len(aw | bw))


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a), str(b)).ratio()


def sample_and_compare(
    repo_id: str,
    local_src: Path,
    dest: Path,
    sample_size: int = 10,
    similarity_threshold: float = 0.6,
    seed: int = 42,
    hf_token: Optional[str] = None,
) -> Dict:
    """Sample records from HF dataset, attempt to map to local PDFs and compare parsed text.

    Returns dict with keys: samples (list), summary (dict)
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Load dataset (tests may monkeypatch load_dataset). Keep simple to stay compatible.
    ds = _retry(lambda: load_dataset(repo_id, **({"use_auth_token": hf_token} if hf_token else {})), retries=3)
    # ds is a DatasetDict
    split_names = list(ds.keys())
    sizes = {s: len(ds[s]) for s in split_names}
    total = sum(sizes.values())
    if total == 0:
        raise ValueError("Dataset has no records")

    # compute per-split sample counts proportional
    import random

    rnd = random.Random(seed)
    sample_counts = {s: max(1, int(sample_size * (sizes[s] / total))) for s in split_names}
    # adjust to exact sample_size
    count_sum = sum(sample_counts.values())
    while count_sum > sample_size:
        # remove 1 from largest
        largest = max(sample_counts, key=lambda k: sample_counts[k])
        if sample_counts[largest] > 1:
            sample_counts[largest] -= 1
            count_sum -= 1
        else:
            break
    while count_sum < sample_size:
        smallest = min(sample_counts, key=lambda k: sample_counts[k])
        sample_counts[smallest] += 1
        count_sum += 1

    samples_results = []

    local_src = Path(local_src)
    # gather local pdf files for improved matching (sorted for determinism)
    target_files = sorted(list(local_src.rglob("*.pdf"))) if local_src.exists() else []

    def _clean(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

    unmatched = 0

    for split, cnt in sample_counts.items():
        if cnt <= 0:
            continue
        ds_split = ds[split]
        indices = rnd.sample(range(len(ds_split)), min(cnt, len(ds_split)))
        for idx in indices:
            record = ds_split[int(idx)]
            rec = {"split": split, "index": int(idx), "record": record}

            # try mapping
            source_path = None
            doc_id = None
            title = None
            if isinstance(record, dict):
                # handle list-valued fields (datasets often return lists)
                def _scalar(v):
                    if isinstance(v, list):
                        return v[0] if v else None
                    return v

                source_path = _scalar(record.get("source_path") or record.get("source") or record.get("file"))
                doc_id = _scalar(record.get("doc_id") or record.get("id"))
                title = _scalar(record.get("title"))

            mapped = None
            # 1) explicit source_path
            if source_path:
                p = Path(source_path)
                if not p.is_absolute():
                    candidate = local_src / p
                else:
                    candidate = p
                if candidate.exists():
                    mapped = candidate

            # 2) prefer exact stem match against doc_id or title
            if mapped is None and (doc_id or title) and target_files:
                key = _clean(str(doc_id or title))
                # exact stem match
                for tf in target_files:
                    if _clean(tf.stem) == key:
                        mapped = tf
                        break
                # substring match on stem or filename
                if mapped is None:
                    for tf in target_files:
                        if key in _clean(tf.stem) or key in _clean(tf.name):
                            mapped = tf
                            break

            if mapped is None:
                unmatched += 1
                rec.update({"mapped_path": None, "error": "no local mapping"})
                rec.update({"jaccard": 0.0, "seq_ratio": 0.0, "word_count_ratio": 0.0, "word_count_diff": 1.0, "pass": False})
                samples_results.append(rec)
                continue

            # parse local pdf
            try:
                engine = _get_engine()
                parsed = engine.parse(str(mapped))
                parsed_text = getattr(parsed, "text", "")
            except Exception as e:
                logger.exception("Parsing failed for %s: %s", mapped, e)
                rec.update({"mapped_path": str(mapped), "error": f"parse_error: {e}"})
                rec.update({"jaccard": 0.0, "seq_ratio": 0.0, "word_count_ratio": 0.0, "word_count_diff": 1.0, "pass": False})
                samples_results.append(rec)
                continue

            # compare to dataset text field
            ds_text = None
            for key in ("text", "content", "extracted_text", "body"):
                if key in record:
                    ds_text = record[key]
                    break
            # coerce list-valued dataset fields to scalar strings for comparison
            if isinstance(ds_text, list):
                ds_text = ds_text[0] if ds_text else ""
            if ds_text is None:
                # try concatenating available string fields
                ds_text = " ".join([str(v) for v in record.values() if isinstance(v, str)])

            j = _jaccard(parsed_text, ds_text)
            r = _seq_ratio(parsed_text, ds_text)
            wc_parsed = len(str(parsed_text).split())
            wc_ds = len(str(ds_text).split())
            wc_ratio = min(wc_parsed, wc_ds) / max(1, max(wc_parsed, wc_ds))
            # relative word-count difference (0.0 = identical, larger = worse)
            word_count_diff = abs(wc_parsed - wc_ds) / max(1, wc_ds)
            avg_sim = (j + r) / 2.0
            passed = avg_sim >= similarity_threshold

            rec.update(
                {
                    "mapped_path": str(mapped),
                    "jaccard": j,
                    "seq_ratio": r,
                    "word_count_ratio": wc_ratio,
                    "word_count_diff": word_count_diff,
                    "avg_similarity": avg_sim,
                    "pass": passed,
                }
            )
            samples_results.append(rec)

    # summary
    passed = sum(1 for s in samples_results if s.get("pass"))
    total = len(samples_results)
    summary = {"passed": passed, "total": total, "pass_rate": passed / max(1, total)}

    # save a copy of results (atomic)
    out = {"samples": samples_results, "summary": summary}
    out_path = dest / "audit_results.json"
    tmpf = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        json.dump(out, tmpf, indent=2)
        tmpf.close()
        os.replace(tmpf.name, str(out_path))
    finally:
        if os.path.exists(tmpf.name):
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass

    logger.info("Audit completed: %d/%d passed (%.2f%%)", passed, total, summary["pass_rate"] * 100.0)
    if unmatched:
        logger.warning("Unmatched records during mapping: %d", unmatched)

    return out


def generate_report(results: Dict, dest: Path, pipeline_version: str = "1.0.0", run_id: Optional[str] = None, dataset_id: Optional[str] = None, sample_params: Optional[Dict] = None) -> Tuple[str, str]:
    """Produce final audit artifacts in `dest` and return paths (json, md).

    The audit_report.json follows the agreed schema:
      - run_id, repo_id, timestamp (ISO UTC), cli_args, environment,
        dataset_summary, quality_checks, artifacts, warnings, errors, exit_code

    This function also writes manifest.json and samples/*.ndjson atomically.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = str(uuid.uuid4())

    summary = results.get("summary", {})
    passed = summary.get("passed", 0)
    total = summary.get("total", 0)
    pass_rate = summary.get("pass_rate", 0.0)

    # Environment info
    env = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
    }

    # Dataset summary: best-effort from available artifacts
    data_dir = dest / "data"
    num_files = 0
    formats = set()
    for p in (data_dir.rglob("*") if data_dir.exists() else []):
        if p.is_file():
            num_files += 1
            formats.add(p.suffix.lower().lstrip("."))

    dataset_summary = {
        "num_rows": results.get("dataset_total_rows") if results.get("dataset_total_rows") is not None else None,
        "num_files": num_files,
        "formats": sorted(list(formats)),
        "sample_records_count": len(results.get("samples", [])),
    }

    # Quality checks
    threshold = (sample_params or {}).get("similarity_threshold", 0.6)
    qc_pass = pass_rate >= threshold
    quality_checks = [
        {
            "name": "pass_rate_vs_threshold",
            "status": "pass" if qc_pass else "fail",
            "message": f"Pass rate {pass_rate:.3f} {'>=' if qc_pass else '<'} threshold {threshold}",
            "metrics": {"pass_rate": pass_rate, "threshold": threshold},
        }
    ]

    # Persist samples as ndjson and copy any referenced images
    samples_dir = dest / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_ndjson = samples_dir / "samples.ndjson"
    tmp_samples = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        for s in results.get("samples", []):
            # Write the dataset record (if present) as one JSON line
            rec = s.get("record") if isinstance(s.get("record"), dict) else {"record": s.get("record")}
            json.dump({"split": s.get("split"), "index": s.get("index"), "mapped_path": s.get("mapped_path"), "metrics": {k: v for k, v in s.items() if k in ("jaccard", "seq_ratio", "avg_similarity", "pass")}, "record": rec}, tmp_samples)
            tmp_samples.write("\n")
            # copy image fields if present and path-like
            try:
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        if k.lower().startswith("image") and isinstance(v, str) and os.path.exists(v):
                            images_out = samples_dir / "images"
                            images_out.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(v, images_out / Path(v).name)
            except Exception:
                pass
        tmp_samples.close()
        os.replace(tmp_samples.name, str(samples_ndjson))
    finally:
        if os.path.exists(tmp_samples.name):
            try:
                os.unlink(tmp_samples.name)
            except Exception:
                pass

    # Build artifact manifest and artifacts list with sha256
    manifest = []
    artifacts_list = []
    for p in sorted([p for p in dest.rglob("*") if p.is_file()]):
        rel = str(p.relative_to(dest))
        try:
            size = p.stat().st_size
        except Exception:
            size = None
        sha = None
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            sha = h.hexdigest()
        except Exception:
            sha = None
        manifest.append({"path": rel, "size": size, "sha256": sha})
        artifacts_list.append({"path": rel, "size": size, "sha256": sha})

    # Atomic write manifest.json
    manifest_path = dest / "manifest.json"
    tmp_m = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        json.dump(manifest, tmp_m, indent=2)
        tmp_m.close()
        os.replace(tmp_m.name, str(manifest_path))
    finally:
        if os.path.exists(tmp_m.name):
            try:
                os.unlink(tmp_m.name)
            except Exception:
                pass

    report = {
        "run_id": run_id,
        "repo_id": dataset_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cli_args": sample_params or {},
        "environment": env,
        "dataset_summary": dataset_summary,
        "quality_checks": quality_checks,
        "artifacts": artifacts_list,
        "warnings": results.get("warnings", []),
        "errors": results.get("errors", []),
        "exit_code": 0 if qc_pass else 1,
    }

    # Atomic write audit_report.json
    json_path = dest / "audit_report.json"
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        json.dump(report, tmp, indent=2)
        tmp.close()
        os.replace(tmp.name, str(json_path))
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    # create markdown summary atomically (keep prior behavior)
    md_path = dest / "audit_summary.md"
    samples_list = results.get("samples", [])
    discrepancies = sorted(samples_list, key=lambda s: s.get("avg_similarity", 1.0))[:5]

    lines = [f"# Dataset Audit Report\n", f"- Total samples: {total}\n", f"- Passed: {passed}\n", f"- Pass rate: {pass_rate:.2%}\n", "\n", "## Top discrepancies\n"]

    for d in discrepancies:
        lines.append(f"### {d.get('mapped_path') or d.get('record', {}).get('id') or 'unknown'}\n")
        lines.append(f"- split: {d.get('split')} index: {d.get('index')}\n")
        lines.append(f"- avg_similarity: {d.get('avg_similarity', 0):.3f}\n")
        excerpt = None
        try:
            rec = d.get('record', {})
            excerpt = (rec.get('text') or list(rec.values())[0]) if isinstance(rec, dict) else None
        except Exception:
            excerpt = None
        if excerpt:
            excerpt_short = str(excerpt)[:400].replace('\n', ' ')
            lines.append(f"- excerpt: ```{excerpt_short}```\n")
        lines.append("\n")

    md_text = "\n".join(lines)
    tmp_md = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        tmp_md.write(md_text)
        tmp_md.close()
        os.replace(tmp_md.name, str(md_path))
    finally:
        if os.path.exists(tmp_md.name):
            try:
                os.unlink(tmp_md.name)
            except Exception:
                pass

    return str(json_path), str(md_path)


def _setup_jsonl_logger(log_path: Path, level=logging.INFO):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "event": record.getMessage(),
                "details": {"logger": record.name},
            }
            if record.exc_info:
                payload["details"]["exc"] = self.formatException(record.exc_info)
            # Ensure each log record is a single JSON line
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # remove other handlers to avoid duplicate logs in CI
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    return handler


def run_audit(
    dataset: str,
    output: Optional[str] = None,
    sample: int = 10,
    seed: int = 42,
    similarity_threshold: float = 0.6,
    hf_token: Optional[str] = None,
    no_download: bool = False,
) -> int:
    """Run the full audit pipeline and return exit code (0 success, 1 fail threshold, 2 usage/cancel/not found).

    Example one-liner (module entrypoint):
    # PYTHONPATH=. python3 -m src.doc_parser_engine.audit --dataset REPO_ID --output artifacts/<ISO_TIMESTAMP>

    """
    # deterministic timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(output) if output else Path("artifacts") / timestamp
    # if output provided but is a directory for multiple runs, append timestamp
    if output and Path(output).exists() and Path(output).is_dir():
        out_root = Path(output) / timestamp
    else:
        out_root = Path(output) if output else Path("artifacts") / timestamp

    run_id = str(uuid.uuid4())
    logs_dir = out_root / "logs"
    log_file = logs_dir / "audit.log"
    _setup_jsonl_logger(log_file)

    logger.info("Starting audit run %s for dataset %s", run_id, dataset)

    try:
        downloads = download_hf_artifacts(dataset, dest=out_root, hf_token=hf_token, no_download=no_download)
    except Exception as e:
        logger.exception("Failed to download HF artifacts: %s", e)
        return 2

    try:
        # Determine local source directory for mapping. Prefer local directory adjacent to the
        # provided output (useful for tests), then explicit output/local, then project parsed_outputs,
        # and finally the current working directory.
        candidates = []
        if output:
            # parent/local (e.g. /tmp/xxx/local)
            candidates.append(Path(output).parent / "local")
            # output/local if output is a parent dir
            candidates.append(Path(output) / "local")
        # Project-level parsed_outputs
        candidates.append(Path("parsed_outputs"))
        # Fallback to CWD
        candidates.append(Path.cwd())

        local_src_chosen = None
        for c in candidates:
            if c.exists():
                local_src_chosen = c
                break
        if local_src_chosen is None:
            local_src_chosen = Path.cwd()

        # If tests provide a local directory adjacent to the output (common in tests),
        # copy PDFs into the run's output local dir so sample_and_compare can operate
        # deterministically inside the artifacts tree.
        if output:
            parent_local = Path(output).parent / "local"
            if parent_local.exists() and any(parent_local.rglob("*.pdf")):
                out_local = out_root / "local"
                out_local.mkdir(parents=True, exist_ok=True)
                for p in parent_local.rglob("*.pdf"):
                    try:
                        shutil.copy2(p, out_local / p.name)
                    except Exception:
                        pass
                local_src_chosen = out_local

        logger.info("Using local_src for mapping: %s", str(local_src_chosen))
        results = sample_and_compare(dataset, local_src=local_src_chosen, dest=out_root, sample_size=sample, similarity_threshold=similarity_threshold, seed=seed, hf_token=hf_token)
    except Exception as e:
        logger.exception("Sampling and comparison failed: %s", e)
        return 2

    # generate report
    sample_params = {"sample_size": sample, "seed": seed, "similarity_threshold": similarity_threshold}
    try:
        report_path, md_path = generate_report(results, dest=out_root, pipeline_version="1.0.0", run_id=run_id, dataset_id=dataset, sample_params=sample_params)
        logger.info("Wrote report %s and summary %s", report_path, md_path)
    except Exception as e:
        logger.exception("Failed to generate report: %s", e)
        return 2

    pass_rate = results.get("summary", {}).get("pass_rate", 0.0)
    # CI-friendly exit codes
    if pass_rate >= similarity_threshold:
        logger.info("Audit passed: pass_rate=%.3f threshold=%.3f", pass_rate, similarity_threshold)
        return 0
    else:
        logger.warning("Audit failed: pass_rate=%.3f threshold=%.3f", pass_rate, similarity_threshold)
        return 1


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description="Run dataset audit")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--output", required=False, help="Output artifacts directory or parent dir")
    parser.add_argument("--sample", type=int, default=10, help="Number of samples to check")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic sampling")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="Threshold for pass/fail")
    parser.add_argument("--hf-token", required=False, help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--no-download", action="store_true", help="Skip HF downloads and use local artifacts only")

    args = parser.parse_args()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    rc = run_audit(dataset=args.dataset, output=args.output, sample=args.sample, seed=args.seed, similarity_threshold=args.similarity_threshold, hf_token=hf_token, no_download=args.no_download)
    exit(rc)


# Allow running as a module: PYTHONPATH=. python3 -m src.doc_parser_engine.audit --dataset REPO_ID --output artifacts/<timestamp>
if __name__ == "__main__":
    # Ensure 'src' is importable when running as a module from project root
    try:
        src_dir = str(Path(__file__).resolve().parents[1])
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    except Exception:
        pass
    _parse_args_and_run()
