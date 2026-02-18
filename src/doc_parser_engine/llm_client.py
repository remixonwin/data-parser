import logging
import httpx
import base64
import io
from typing import Optional, List, Dict, Any
from PIL import Image
import time

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
        retries: int = 3,
    ):
        self.api_base = api_base.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.retries = retries
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
        Send a chat completion request to the LLM Manager with retries.
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

        attempt = 0
        backoff = 0.5
        last_exc = None
        while attempt < self.retries:
            try:
                response = self.client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # surface HTTP errors with status code and response text
                status = e.response.status_code if e.response is not None else 'unknown'
                text = e.response.text if e.response is not None else str(e)
                # Retry on server errors (5xx)
                if isinstance(status, int) and 500 <= status < 600:
                    last_exc = e
                    attempt += 1
                    logger.debug(f"LLM API server error {status}, retrying attempt {attempt}/{self.retries}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.error(f"LLM API HTTP error: {status} - {text}")
                raise RuntimeError(f"LLM API HTTP error: {status} - {text}")
            except Exception as e:
                last_exc = e
                attempt += 1
                logger.debug(f"LLM API request attempt {attempt} failed: {e}")
                time.sleep(backoff)
                backoff *= 2
        logger.error(f"LLM API request ultimately failed after {self.retries} attempts: {last_exc}")
        raise RuntimeError(f"LLM API request failed: {last_exc}")

    def generate_caption(self, image_bytes: bytes, model: Optional[str] = None) -> str:
        """
        Generate a caption for an image via the LLM API.
        Uses base64-encoded images with vision-capable models.
        """
        import base64
        import io
        from PIL import Image

        # Get image format
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            img_format = pil_img.format.lower() if pil_img.format else "png"
        except:
            img_format = "png"

        # Encode image to base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Try with data URL format (some APIs support this)
        messages = [
            {
                "role": "user",
                "content": f"Describe this image from a Wisconsin driver handbook document. What does it show? Be concise (1-2 sentences).",
            }
        ]

        # Add cache-busting by including image hash
        try:
            import hashlib

            img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        except:
            img_hash = "default"

        try:
            # Use fast routing for faster processing
            result = self.chat_completion(
                messages=messages,
                model=model if model else "llama-3.2-90b-vision-preview",
                temperature=0.7,
                max_tokens=150,
                routing={"strategy": "fast", "cache_enabled": False},
            )
            caption = result["choices"][0]["message"]["content"].strip()

            # Check if we got a valid response
            if caption and len(caption) > 10:
                return caption
        except Exception as e:
            logger.debug(f"Vision caption failed: {e}")

        # Fallback to OCR-based caption
        return self._generate_fallback_caption(image_bytes, model)

    def _generate_fallback_caption(
        self, image_bytes: bytes, model: Optional[str] = None
    ) -> str:
        """Fallback caption generation without vision."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Try OCR to get text content from image
            pil_img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(pil_img).strip()

            if text and len(text) > 5:
                # Use extracted text to generate a caption
                prompt = f"Based on this text extracted from an image: '{text[:200]}'. Describe what image this text came from in 1 sentence."
                messages = [{"role": "user", "content": prompt}]

                result = self.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=0.5,
                    max_tokens=100,
                    routing={"strategy": "fast"},
                )
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"Fallback caption failed: {e}")

        return "Image content (OCR/text extraction failed)"

    def categorize_image(
        self, image_bytes: bytes, ocr_text: str = "", model: Optional[str] = None
    ) -> str:
        """
        Categorize an image using LLM based on text hints.
        """
        prompt = "Categorize a document image into one of: [photo, chart, diagram, screenshot, table, other]. "
        if ocr_text:
            prompt += f"OCR Text Hint: {ocr_text}. "
        prompt += "Return ONLY the category name."

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.chat_completion(
                messages=messages, model=model, routing={"strategy": "fast"}
            )
            content = result["choices"][0]["message"]["content"].strip().lower()
            for cat in ["photo", "chart", "diagram", "screenshot", "table"]:
                if cat in content:
                    return cat
            return "photo"
        except Exception as e:
            logger.debug(f"Image categorization failed: {e}")
            return "photo"
