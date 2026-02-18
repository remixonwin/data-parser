import os
import json
from pathlib import Path
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from doc_parser_engine import audit


def test_download_hf_artifacts_no_download(tmp_path):
    dest = tmp_path / "out"
    res = audit.download_hf_artifacts("some/repo", dest=dest, hf_token=None, no_download=True)
    assert isinstance(res, dict)
    assert res.get("downloaded_files") == []
    assert Path(res.get("metadata_path")).exists()


def test_sample_and_compare_mapping(monkeypatch, tmp_path):
    # Create a fake pdf file
    local_src = tmp_path / "local"
    local_src.mkdir()
    pdf = local_src / "sample_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake pdf")

    # Build dataset with a record that references the pdf by doc_id and title
    record = {"id": ["sample_doc"], "doc_id": ["sample_doc"], "title": ["Sample Doc"], "text": ["This is expected text from pdf"]}
    ds = DatasetDict({"train": Dataset.from_dict(record)})

    # monkeypatch datasets.load_dataset to return our dataset (accept kwargs)
    monkeypatch.setattr(audit, "load_dataset", lambda repo_id, **kwargs: ds)

    class DummyParsed:
        def __init__(self, text):
            self.text = text

    class DummyEngine:
        def parse(self, path):
            return DummyParsed("This is expected text from pdf")

    monkeypatch.setattr(audit, "_get_engine", lambda: DummyEngine())

    dest = tmp_path / "out"
    res = audit.sample_and_compare("some/repo", local_src=local_src, dest=dest, sample_size=1, similarity_threshold=0.1)
    assert isinstance(res, dict)
    assert "samples" in res and "summary" in res
    assert len(res["samples"]) == 1
    s = res["samples"][0]
    assert "jaccard" in s and "seq_ratio" in s and "avg_similarity" in s
    assert s.get("mapped_path") is not None


def test_deterministic_sampling(monkeypatch, tmp_path):
    # Create synthetic dataset with 10 records
    records = {"id": [str(i) for i in range(10)], "text": [f"text {i}" for i in range(10)]}
    ds = DatasetDict({"train": Dataset.from_dict(records)})
    monkeypatch.setattr(audit, "load_dataset", lambda repo_id, **kwargs: ds)

    # Dummy engine that returns empty parsed text
    class DummyParsed:
        def __init__(self, text):
            self.text = text

    class DummyEngine:
        def parse(self, path):
            return DummyParsed("")

    monkeypatch.setattr(audit, "_get_engine", lambda: DummyEngine())

    dest1 = tmp_path / "out1"
    dest2 = tmp_path / "out2"

    r1 = audit.sample_and_compare("some/repo", local_src=tmp_path, dest=dest1, sample_size=5, seed=12345)
    r2 = audit.sample_and_compare("some/repo", local_src=tmp_path, dest=dest2, sample_size=5, seed=12345)

    # Samples should be identical between runs with same seed
    assert r1["samples"] == r2["samples"]


def test_run_audit_creates_artifacts(monkeypatch, tmp_path):
    # Prepare fake dataset (single row) and local pdf
    local_src = tmp_path / "local"
    local_src.mkdir()
    pdf = local_src / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake pdf")

    record = {"id": ["doc"], "doc_id": ["doc"], "title": ["Doc"], "text": ["sample"]}
    ds = DatasetDict({"train": Dataset.from_dict(record)})

    monkeypatch.setattr(audit, "load_dataset", lambda repo_id, **kwargs: ds)

    class DummyParsed:
        def __init__(self, text):
            self.text = text

    class DummyEngine:
        def parse(self, path):
            return DummyParsed("sample")

    monkeypatch.setattr(audit, "_get_engine", lambda: DummyEngine())

    out_dir = tmp_path / "artifacts_test"

    # Run audit (no network downloads)
    rc = audit.run_audit(dataset="some/repo", output=str(out_dir), sample=1, seed=1, hf_token=None, no_download=True)
    assert rc == 0

    # Check artifact files (report written under out_dir directly)
    assert out_dir.exists()
    # audit_results.json should exist
    assert (out_dir / "audit_results.json").exists()
    # audit_report.json should exist
    assert (out_dir / "audit_report.json").exists()
    assert (out_dir / "logs" / "audit.log").exists()

    # Validate report schema
    report = json.loads((out_dir / "audit_report.json").read_text(encoding="utf-8"))
    # Top-level keys
    assert "run_id" in report
    assert "repo_id" in report
    assert "timestamp" in report
    assert isinstance(report.get("environment"), dict)
    assert isinstance(report.get("dataset_summary"), dict)
    assert isinstance(report.get("quality_checks"), list)
    assert isinstance(report.get("artifacts"), list)
    assert "exit_code" in report
