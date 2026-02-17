from doc_parser_engine.llm_client import LLMClient
import unittest
from unittest.mock import MagicMock, patch

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(api_base="http://mock-api:7543")

    @patch("httpx.Client.post")
    def test_chat_completion(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.chat_completion(messages)
        
        self.assertEqual(result["choices"][0]["message"]["content"], "Test response")
        mock_post.assert_called_once()

    @patch("doc_parser_engine.llm_client.LLMClient.chat_completion")
    def test_generate_caption(self, mock_chat):
        mock_chat.return_value = {
            "choices": [{"message": {"content": "A beautiful landscape."}}]
        }
        
        caption = self.client.generate_caption(b"fake-image-bytes")
        self.assertEqual(caption, "A beautiful landscape.")

if __name__ == "__main__":
    unittest.main()
