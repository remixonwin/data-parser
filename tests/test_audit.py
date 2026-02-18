import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from doc_parser_engine import audit


def test_sample_and_compare_monkeypatched(monkeypatch, tmp_path):
    # create a tiny dataset with one record that points to an existing local PDF in the repo
    # find any pdf in prepware_study_guide
    repo_root = Path(__file__).resolve().parents[1]
    local_src = repo_root / "prepware_study_guide"
    pdfs = list(local_src.rglob("*.pdf"))
    assert pdfs, "No local PDFs found for test"
    pdf = pdfs[0]

    # dataset library expects dict of lists
    record = {"id": ["testdoc1"], "source_path": [str(pdf)], "text": ["This is some expected text from pdf"]}
    ds = DatasetDict({"train": Dataset.from_dict(record)})

    # monkeypatch datasets.load_dataset to return our tiny dataset
    monkeypatch.setattr(audit, "load_dataset", lambda repo_id: ds)

    # monkeypatch get_engine to return a dummy engine whose parse returns an object with .text
    class DummyParsed:
        def __init__(self, text):
            self.text = text

    class DummyEngine:
        def parse(self, path):
            return DummyParsed("This is some expected text from pdf")

    # audit module uses a lazy _get_engine factory; patch that
    monkeypatch.setattr(audit, "_get_engine", lambda: DummyEngine())

    dest = tmp_path / "out"
    res = audit.sample_and_compare("some/repo", local_src=local_src, dest=dest, sample_size=1, similarity_threshold=0.1)
    assert isinstance(res, dict)
    assert "samples" in res and "summary" in res
    assert len(res["samples"]) == 1
    s = res["samples"][0]
    assert "jaccard" in s and "seq_ratio" in s and "avg_similarity" in s
