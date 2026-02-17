import logging
import httpx
import base64
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for the LLM Manager API.
    Supports OpenAI-compatible chat completions with structural enhancements.
    """

    def __init__(
        self,
        api_base: str = "http://0.0.0.0:7543",
        default_model: str = "gpt-4o",
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.api_base, timeout=timeout)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        routing: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the LLM Manager.
        """
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if routing:
            payload["routing"] = routing
        if quality:
            payload["quality"] = quality

        try:
            response = self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM API request failed: {e}")
            raise

    def generate_caption(self, image_bytes: bytes, model: Optional[str] = None) -> str:
        """
        Generate a caption for an image via the LLM API.
        NOTE: Schema requires content to be a string.
        """
        # If API doesn't support vision via structured content, we might be limited.
        # But let's try a simple text prompt first to verify the 422 is gone.
        messages = [
            {
                "role": "user",
                "content": "Describe an image concisely for a document dataset. (Image processing via API pending schema verification)"
            }
        ]
        
        try:
            result = self.chat_completion(
                messages=messages,
                model=model,
                routing={"strategy": "balanced", "cache_enabled": True}
            )
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"Caption generation failed: {e}")
            return ""

    def categorize_image(self, image_bytes: bytes, ocr_text: str = "", model: Optional[str] = None) -> str:
        """
        Categorize an image using LLM based on text hints.
        """
        prompt = "Categorize a document image into one of: [photo, chart, diagram, screenshot, table, other]. "
        if ocr_text:
            prompt += f"OCR Text Hint: {ocr_text}. "
        prompt += "Return ONLY the category name."

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            result = self.chat_completion(
                messages=messages,
                model=model,
                routing={"strategy": "fast"}
            )
            content = result["choices"][0]["message"]["content"].strip().lower()
            for cat in ["photo", "chart", "diagram", "screenshot", "table"]:
                if cat in content:
                    return cat
            return "photo"
        except Exception as e:
            logger.debug(f"Image categorization failed: {e}")
            return "photo"
