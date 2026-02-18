import pytest
from doc_parser_engine.captioning.caption_engine import CaptionEngine

class DummyLLM:
    def generate_caption(self, b):
        return "remote caption"
    def categorize_image(self, b, ocr_text=''):
        return 'photo'


def test_caption_engine_uses_api_when_llm_client(monkeypatch):
    llm = DummyLLM()
    engine = CaptionEngine(model_id='api', llm_client=llm, force_local=False)
    imgs = [{'image_bytes': b'img'}]
    # monkeypatch transformers.from_pretrained to raise if called
    import transformers
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *a, **k: (_ for _ in ()).throw(AssertionError('HF called')))
    res = engine.caption_batch(imgs)
    assert res[0]['caption'] == 'remote caption'


def test_caption_engine_force_local_loads(monkeypatch):
    # when force_local True and model is not 'api', _load_model should attempt HF load
    # We'll monkeypatch transformers to simulate a successful minimal load
    class FakeModel:
        def to(self, device):
            return self
        def eval(self):
            pass
    class FakeProcessor:
        def __call__(self, images, return_tensors=None):
            class O:
                def to(self, d):
                    return self
            return O()
    fake = pytest.MonkeyPatch()
    fake.setattr('doc_parser_engine.captioning.caption_engine.BlipProcessor', FakeProcessor, raising=False)
    fake.setattr('doc_parser_engine.captioning.caption_engine.BlipForConditionalGeneration', FakeModel, raising=False)
    try:
        engine = CaptionEngine(model_id='Salesforce/blip-image-captioning-large', llm_client=None, force_local=True)
        # _load_model should not raise
        engine._load_model()
        assert engine._model is not None
    finally:
        fake.undo()
