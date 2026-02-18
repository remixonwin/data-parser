import os
from doc_parser_engine.cli import get_engine


def test_cli_overrides_env(tmp_path, monkeypatch):
    # Ensure env vars are ignored if CLI provided
    monkeypatch.setenv('DOCPARSER_LLM_API_BASE', 'http://env:8000')
    monkeypatch.setenv('DOCPARSER_LLM_MODEL', 'env-model')

    engine = get_engine(llm_api_base='http://cli:9000', llm_model='cli-model', force_local_caption=False)
    assert engine.llm_api_base == 'http://cli:9000'
    assert engine.llm_model == 'cli-model'


def test_env_used_when_no_cli(monkeypatch):
    monkeypatch.setenv('DOCPARSER_LLM_API_BASE', 'http://env:8000')
    monkeypatch.setenv('DOCPARSER_LLM_MODEL', 'env-model')
    engine = get_engine()
    assert engine.llm_api_base == 'http://env:8000'
    assert engine.llm_model == 'env-model'


def test_force_local_caption_flag(monkeypatch):
    # If force_local_caption True, engine.force_local_caption should be True
    engine = get_engine(force_local_caption=True, llm_api_base='http://0.0.0.0:7543', llm_model='gpt-4o')
    assert engine.force_local_caption is True
