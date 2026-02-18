from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from difflib import SequenceMatcher
import os
import re
import tempfile
import shutil
from datetime import datetime

# Lazy import engine factory to avoid circular imports during test collection
def _get_engine():
    # Importing the CLI module here is intentional and lazy to prevent import-time side effects
    from doc_parser_engine.cli import get_engine as _g
    return _g()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _retry(fn, retries=3, base_delay=1.0, exceptions=(Exception,), *args, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            last_exc = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed: %s; retrying in %.1fs", attempt, e, delay)
            time.sleep(delay)
    logger.error("All %d retries failed", retries)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Operation failed after retries")


def download_hf_artifacts(repo_id: str, dest: Path) -> Dict:
    """Download README and parquet shards + save metadata.json into dest.

    Returns a dict with keys: downloaded_files (list), metadata_path (str)
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    def _list_files():
        return api.list_repo_files(repo_id)

    files = _retry(_list_files, retries=3, base_delay=1.0)

    downloaded = []

    # Prefer README files
    for candidate in ("README.md", "readme.md", "README", "README.MD"):
        if candidate in files:
            path = _retry(lambda: hf_hub_download(repo_id=repo_id, filename=candidate), retries=3)
            dest_path = dest / Path(path).name
            Path(path).rename(dest_path)
            downloaded.append(str(dest_path))
            break

    # Save dataset metadata via HfApi.dataset_info if available
    try:
        info = _retry(lambda: api.dataset_info(repo_id), retries=3)
        meta_path = dest / "metadata.json"
        try:
            payload = info.to_dict() if hasattr(info, "to_dict") else dict(info)
        except Exception:
            payload = {"repo_id": repo_id, "note": "dataset_info_unserializable"}
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as e:
        logger.warning("Could not fetch dataset_info from HF: %s", e)
        meta_path = dest / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump({"repo_id": repo_id, "note": "dataset_info not available"}, fh)

    # Download parquet shards (files ending with .parquet)
    parquet_files = [f for f in files if f.lower().endswith(".parquet") or ".parquet" in f.lower()]
    if not parquet_files:
        logger.info("No parquet files listed for repo %s", repo_id)
    for p in parquet_files:
        try:
            local = _retry(lambda: hf_hub_download(repo_id=repo_id, filename=p), retries=3)
            target = data_dir / Path(p).name
            Path(local).rename(target)
            downloaded.append(str(target))
        except Exception as e:
            logger.warning("Failed to download %s: %s", p, e)

    return {"downloaded_files": downloaded, "metadata_path": str(meta_path)}


def _jaccard(a: str, b: str) -> float:
    aw = set([w.lower() for w in a.split() if w.strip()])
    bw = set([w.lower() for w in b.split() if w.strip()])
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return float(len(aw & bw)) / float(len(aw | bw))


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def sample_and_compare(
    repo_id: str,
    local_src: Path,
    dest: Path,
    sample_size: int = 10,
    similarity_threshold: float = 0.6,
    seed: int = 42,
) -> Dict:
    """Sample records from HF dataset, attempt to map to local PDFs and compare parsed text.

    Returns dict with keys: samples (list), summary (dict)
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id)
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
            if isinstance(record, dict):
                source_path = record.get("source_path") or record.get("source") or record.get("file")
                doc_id = record.get("doc_id") or record.get("id") or record.get("title")
            else:
                doc_id = None

            mapped = None
            if source_path:
                p = Path(source_path)
                if not p.is_absolute():
                    candidate = local_src / p
                else:
                    candidate = p
                if candidate.exists():
                    mapped = candidate
            # fallback: search by doc_id within filenames
            if mapped is None and doc_id:
                target_files = list(local_src.rglob("*.pdf"))
                doc_id_str = str(doc_id).lower()
                best = None
                for tf in target_files:
                    name = tf.name.lower()
                    if doc_id_str in name:
                        best = tf
                        break
                mapped = best

            if mapped is None:
                rec.update({"mapped_path": None, "error": "no local mapping"})
                rec.update({"jaccard": 0.0, "seq_ratio": 0.0, "word_count_ratio": 0.0, "pass": False})
                samples_results.append(rec)
                continue

            # parse local pdf
            try:
                engine = _get_engine()
                parsed = engine.parse(str(mapped))
                parsed_text = getattr(parsed, "text", str(parsed))
            except Exception as e:
                logger.exception("Parsing failed for %s: %s", mapped, e)
                rec.update({"mapped_path": str(mapped), "error": f"parse_error: {e}"})
                rec.update({"jaccard": 0.0, "seq_ratio": 0.0, "word_count_ratio": 0.0, "pass": False})
                samples_results.append(rec)
                continue

            # compare to dataset text field
            ds_text = None
            for key in ("text", "content", "extracted_text", "body"):
                if key in record:
                    ds_text = record[key]
                    break
            if ds_text is None:
                # try concatenating available string fields
                ds_text = " ".join([str(v) for v in record.values() if isinstance(v, str)])

            j = _jaccard(parsed_text, ds_text)
            r = _seq_ratio(parsed_text, ds_text)
            wc_parsed = len(parsed_text.split())
            wc_ds = len(str(ds_text).split())
            wc_ratio = min(wc_parsed, wc_ds) / max(1, max(wc_parsed, wc_ds))
            avg_sim = (j + r) / 2.0
            passed = avg_sim >= similarity_threshold

            rec.update(
                {
                    "mapped_path": str(mapped),
                    "jaccard": j,
                    "seq_ratio": r,
                    "word_count_ratio": wc_ratio,
                    "avg_similarity": avg_sim,
                    "pass": passed,
                }
            )
            samples_results.append(rec)

    # summary
    passed = sum(1 for s in samples_results if s.get("pass"))
    total = len(samples_results)
    summary = {"passed": passed, "total": total, "pass_rate": passed / max(1, total)}

    # save a copy of results
    out = {"samples": samples_results, "summary": summary}
    out_path = dest / "audit_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    return out


def generate_report(results: Dict, dest: Path) -> Tuple[str, str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    json_path = dest / "audit_report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # create markdown summary
    md_path = dest / "audit_summary.md"
    total = results.get("summary", {}).get("total", 0)
    passed = results.get("summary", {}).get("passed", 0)
    pass_rate = results.get("summary", {}).get("pass_rate", 0.0)

    samples = results.get("samples", [])
    # top discrepancies = lowest avg_similarity
    discrepancies = sorted(samples, key=lambda s: s.get("avg_similarity", 1.0))[:5]

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
    md_path.write_text(md_text, encoding="utf-8")

    return str(json_path), str(md_path)
