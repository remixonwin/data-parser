import os
from doc_parser_engine.cli import get_engine
from unittest import mock


def test_integration_no_hf_download(monkeypatch, tmp_path):
    # Simulate env LLM API base and model
    monkeypatch.setenv('DOCPARSER_LLM_API_BASE', 'http://0.0.0.0:7543')
    monkeypatch.setenv('DOCPARSER_LLM_MODEL', 'gpt-4o')

    # Prevent transformers from being called anywhere
    def fail(*a, **k):
        raise AssertionError('transformers called')
    monkeypatch.setitem(__builtins__, 'open', open)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', fail, raising=False)
    monkeypatch.setattr('transformers.from_pretrained', fail, raising=False)

    # Create engine pointed at prepware_study_guide dir (exists in repo)
    engine = get_engine()
    # Ensure caption model selected is api
    assert engine._caption_model_id == 'api'

    # Also ensure CaptionEngine won't try to call transformers when captioning via API
    # We'll create a dummy image entry and ensure caption_batch uses llm_client if available
    class FakeLLM:
        def generate_caption(self, b):
            return 'ok'
        def categorize_image(self, b, ocr_text=''):
            return 'photo'
    engine._llm_client = FakeLLM()
    engine._caption_model_id = 'api'
    # Create fake image
    imgs = [{'image_bytes': b'img', 'width': 1000, 'height': 800}]
    res = engine.caption_engine.caption_batch(imgs)
    assert res[0]['caption'] == 'ok'
