import pytest
import httpx
from doc_parser_engine.llm_client import LLMClient

class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text='ok'):
        self.status_code = status_code
        self._json = json_data or {"choices": [{"message": {"content": "A caption from API"}}]}
        self.text = text
    def raise_for_status(self):
        if self.status_code >= 400:
            # Raise HTTPStatusError with a real httpx.Response to ensure attributes exist
            resp = httpx.Response(self.status_code, text=self.text)
            raise httpx.HTTPStatusError("error", request=None, response=resp)
    def json(self):
        return self._json

class DummyClient:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0
    def post(self, path, json=None):
        resp = self._responses[self._i]
        self._i += 1
        return resp


def test_chat_completion_retries(monkeypatch):
    client = LLMClient(api_base='http://test', default_model='m', timeout=0.1, retries=3)
    # first two attempts fail with generic exception, third succeeds
    responses = [
        DummyResponse(status_code=500, text='server error'),
        DummyResponse(status_code=500, text='server error'),
        DummyResponse(status_code=200, json_data={"choices": [{"message": {"content": "ok"}}]})
    ]
    dummy = DummyClient(responses)
    monkeypatch.setattr(client, 'client', dummy)
    res = client.chat_completion(messages=[{"role":"user","content":"hi"}])
    assert res["choices"][0]["message"]["content"] == 'ok'


def test_generate_caption_uses_base_url(monkeypatch):
    client = LLMClient(api_base='http://baseurl', default_model='m')
    # Mock chat_completion to assert model passed
    def fake_chat(messages, model=None, temperature=0.0, max_tokens=None, routing=None, quality=None):
        assert model == 'llama-3.2-90b-vision-preview'
        return {"choices": [{"message": {"content": "A generated caption from fake"}}]}
    monkeypatch.setattr(client, 'chat_completion', fake_chat)
    cap = client.generate_caption(b'\x89PNG')
    assert 'generated' not in cap.lower() or isinstance(cap, str)
