"""
CaptionEngine - Intelligent image captioning system.

Supports multiple backends:
1. BLIP-2 (Salesforce/blip2-opt-2.7b) - Best quality
2. BLIP (Salesforce/blip-image-captioning-large) - Fast
3. ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning) - Lightweight
4. Alt-text fallback - No GPU needed

Also performs OCR on images that appear to contain text (charts, diagrams).
"""

import logging
import io
from typing import Optional, Any
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


class CaptionEngine:
    """
    Multi-backend intelligent image captioning.

    Automatically selects best available model based on hardware
    and generates descriptive captions + OCR text for all images.
    """

    # Model configurations ordered by quality
    MODEL_CONFIGS = {
        "Salesforce/blip2-opt-2.7b": {
            "type": "blip2",
            "min_vram_gb": 6,
            "quality": "excellent",
        },
        "Salesforce/blip2-flan-t5-xl": {
            "type": "blip2",
            "min_vram_gb": 8,
            "quality": "excellent",
        },
        "Salesforce/blip-image-captioning-large": {
            "type": "blip",
            "min_vram_gb": 2,
            "quality": "good",
        },
        "Salesforce/blip-image-captioning-base": {
            "type": "blip",
            "min_vram_gb": 1,
            "quality": "fair",
        },
        "nlpconnect/vit-gpt2-image-captioning": {
            "type": "vit-gpt2",
            "min_vram_gb": 0.5,
            "quality": "basic",
        },
        "api": {
            "type": "api",
            "min_vram_gb": 0,
            "quality": "gpt-4o",
        }
    }

    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-large",
        device: str = "cpu",
        batch_size: int = 4,
        max_new_tokens: int = 100,
        enable_ocr_on_images: bool = True,
        ocr_text_threshold: float = 0.3,  # If >30% pixels are text-like, run OCR
        llm_client: Optional[LLMClient] = None,
        force_local: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_ocr_on_images = enable_ocr_on_images
        self.ocr_text_threshold = ocr_text_threshold
        self.llm_client = llm_client
        self.force_local = force_local

        self._model = None
        self._processor = None
        self._model_type = None

        logger.info(f"CaptionEngine: model={model_id} device={device} force_local={force_local}")

    def _load_model(self):
        """Lazy-load the captioning model."""
        if self._model is not None:
            return

        config = self.MODEL_CONFIGS.get(self.model_id, {})
        model_type = config.get("type", "blip")

        # If model_type is api or we have an llm_client and not forcing local, prefer API path
        if model_type == "api" or (self.llm_client and not self.force_local):
            if not self.llm_client:
                logger.warning("API model selected but no LLM client provided. Falling back to alt-text.")
                self._model = "fallback"
            else:
                self._model = "api"
            self._model_type = "api"
            return

        # At this point, local HF model loading will only happen if force_local is True or no llm_client available
        if not self.force_local:
            logger.info("Local HF model loading suppressed (force_local=False). To load local models set force_local=True")
            self._model = "fallback"
            self._model_type = "fallback"
            return

        try:
            # Warn user about large downloads when attempting HF model loading
            logger.warning("Loading a local HF model via transformers.from_pretrained. This may download large artifacts. Use --force-local-caption to confirm this behavior.")
            from transformers import (
                BlipProcessor, BlipForConditionalGeneration,
                Blip2Processor, Blip2ForConditionalGeneration,
                VisionEncoderDecoderModel, ViTImageProcessor,
                AutoTokenizer,
            )

            if model_type == "blip2":
                self._processor = Blip2Processor.from_pretrained(self.model_id)
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype="auto" if self.device != "cpu" else None,
                )
            elif model_type == "blip":
                self._processor = BlipProcessor.from_pretrained(self.model_id)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    self.model_id,
                )
            elif model_type == "vit-gpt2":
                self._processor = ViTImageProcessor.from_pretrained(self.model_id)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

            self._model = self._model.to(self.device)
            self._model.eval()
            self._model_type = model_type
            logger.info(f"Loaded caption model: {self.model_id}")

        except ImportError:
            raise RuntimeError(
                "transformers not installed. pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load {self.model_id}: {e}")
            logger.warning("Falling back to alt-text only captioning")
            self._model = "fallback"

    def caption_batch(self, images: list[dict]) -> list[dict]:
        """
        Generate captions for a batch of images.

        Args:
            images: List of image dicts from ImageExtractor

        Returns:
            Images with 'caption', 'caption_model', 'ocr_text' filled
        """
        self._load_model()

        if self._model == "fallback":
            images = self._apply_fallback_captions(images)
        elif self._model == "api":
            # Process via API
            for img_dict in images:
                if img_dict.get("image_bytes"):
                    img_dict["caption"] = self.llm_client.generate_caption(img_dict["image_bytes"])
                    img_dict["caption_model"] = "api"
                    img_dict["caption_confidence"] = 1.0
        else:
            # Process in batches
            for i in range(0, len(images), self.batch_size):
                batch = images[i: i + self.batch_size]
                self._caption_batch_chunk(batch)

        # OCR pass on text-heavy images
        if self.enable_ocr_on_images:
            for img_dict in images:
                if not img_dict.get("ocr_text") and self._should_ocr(img_dict):
                    img_dict["ocr_text"] = self._run_ocr(img_dict.get("image_bytes", b""))

        # Categorization pass
        for img_dict in images:
            img_dict["category"] = self._categorize_image(img_dict)

        return images

    def _categorize_image(self, img_dict: dict) -> str:
        """Categorize image based on properties and OCR text."""
        ocr_text = img_dict.get("ocr_text", "")
        width = img_dict.get("width", 0)
        height = img_dict.get("height", 0)
        aspect = width / height if height > 0 else 1

        if self.llm_client and img_dict.get("image_bytes"):
            return self.llm_client.categorize_image(img_dict["image_bytes"], ocr_text=ocr_text)

        if len(ocr_text) > 50:
            return "chart" if any(c.isdigit() for c in ocr_text) else "diagram"
        if aspect > 2.5 or aspect < 0.4:
            return "diagram"
            
        return "photo"

    def _caption_batch_chunk(self, batch: list[dict]):
        """Caption a single chunk of images."""
        from PIL import Image as PILImage
        import torch

        pil_images = []
        valid_indices = []

        for i, img_dict in enumerate(batch):
            try:
                data = img_dict.get("image_bytes", b"")
                if data:
                    pil_img = PILImage.open(io.BytesIO(data)).convert("RGB")
                    pil_images.append(pil_img)
                    valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Image open failed: {e}")

        if not pil_images:
            return

        try:
            if self._model_type in ("blip", "blip2"):
                inputs = self._processor(
                    images=pil_images,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=4,
                        early_stopping=True,
                    )

                captions = self._processor.batch_decode(out, skip_special_tokens=True)

            elif self._model_type == "vit-gpt2":
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                pixel_values = self._processor(
                    images=pil_images, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    out = self._model.generate(
                        pixel_values,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=4,
                    )
                captions = tokenizer.batch_decode(out, skip_special_tokens=True)

            else:
                captions = ["" for _ in pil_images]

            # Assign captions back
            for list_idx, caption in zip(valid_indices, captions):
                caption_clean = self._clean_caption(caption)
                batch[list_idx]["caption"] = caption_clean
                batch[list_idx]["caption_model"] = self.model_id
                batch[list_idx]["caption_confidence"] = 1.0

        except Exception as e:
            logger.warning(f"Batch captioning failed: {e}")
            for i in valid_indices:
                batch[i]["caption"] = batch[i].get("alt_text", "")
                batch[i]["caption_model"] = "fallback"

    def _apply_fallback_captions(self, images: list[dict]) -> list[dict]:
        """Use alt text or generate basic descriptive captions."""
        for img_dict in images:
            alt = img_dict.get("alt_text", "")
            if alt:
                img_dict["caption"] = alt
                img_dict["caption_model"] = "alt_text"
            else:
                w = img_dict.get("width", 0)
                h = img_dict.get("height", 0)
                fmt = img_dict.get("format", "image")
                page = img_dict.get("page", 0)
                img_dict["caption"] = (
                    f"A {w}x{h} {fmt} image"
                    f"{f' from page {page}' if page else ''}."
                )
                img_dict["caption_model"] = "rule_based"
            img_dict["caption_confidence"] = 0.5
        return images

    def _should_ocr(self, img_dict: dict) -> bool:
        """Heuristic: should we OCR this image?"""
        # Screenshots, charts, diagrams often contain text
        w = img_dict.get("width", 0)
        h = img_dict.get("height", 0)
        if w == 0 or h == 0:
            return False
        # Wide/squat images are often charts/screenshots
        aspect = w / h if h > 0 else 1
        return 0.5 < aspect < 5.0 and w > 200

    def _run_ocr(self, image_bytes: bytes) -> str:
        """Run pytesseract OCR on image bytes."""
        try:
            import pytesseract
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            cleaned = " ".join(text.split())
            return cleaned if len(cleaned) > 10 else ""
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""

    def _clean_caption(self, caption: str) -> str:
        """Clean and normalize generated captions."""
        caption = caption.strip()
        # Remove common model artifacts
        for prefix in ("a photo of", "an image of", "a picture of"):
            if caption.lower().startswith(prefix):
                caption = caption[len(prefix):].strip()
                break
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        # Ensure ends with period
        if caption and not caption[-1] in ".!?":
            caption += "."
        return caption
