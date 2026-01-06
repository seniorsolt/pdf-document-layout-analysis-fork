import base64
import io
import threading
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from PIL import Image

from configuration import (
    REMOTE_OCR_API_KEY,
    REMOTE_OCR_BASE_URL,
    REMOTE_OCR_MODEL,
    REMOTE_OCR_TEMPERATURE,
    REMOTE_OCR_TIMEOUT_SEC,
)


def _pil_image_to_data_url(image: Image.Image, mime: str = "image/png") -> str:
    """
    Convert PIL image into a data URL for OpenAI-compatible 'image_url' inputs.
    """
    if mime not in {"image/png", "image/jpeg"}:
        mime = "image/png"

    buf = io.BytesIO()
    if mime == "image/jpeg":
        image.convert("RGB").save(buf, format="JPEG", quality=95)
    else:
        image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _get_client() -> OpenAI:
    # OpenAI-compatible endpoints (e.g. vLLM or OpenRouter). Some servers are API-key agnostic,
    # but the SDK still requires a value.
    return OpenAI(api_key=REMOTE_OCR_API_KEY, base_url=REMOTE_OCR_BASE_URL, timeout=REMOTE_OCR_TIMEOUT_SEC)


_thread_local = threading.local()


def _get_async_client() -> AsyncOpenAI:
    # OpenAI-compatible endpoints (e.g. vLLM or OpenRouter). Some servers are API-key agnostic,
    # but the SDK still requires a value.
    client = getattr(_thread_local, "async_client", None)
    if client is None:
        client = AsyncOpenAI(api_key=REMOTE_OCR_API_KEY, base_url=REMOTE_OCR_BASE_URL, timeout=REMOTE_OCR_TIMEOUT_SEC)
        _thread_local.async_client = client
    return client


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        # remove first fence line
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else ""
        # remove trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def ocr_table_html(image: Image.Image, model: Optional[str] = None) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model or REMOTE_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_image_to_data_url(image)}},
                    {
                        "type": "text",
                        "text": (
                            "Extract the table from the image.\n"
                            "Return ONLY valid HTML for the table (prefer <table>...</table>).\n"
                            "Do NOT include markdown fences or any extra text."
                        ),
                    },
                ],
            }
        ],
        temperature=REMOTE_OCR_TEMPERATURE,
    )
    return _strip_code_fences(response.choices[0].message.content or "")


async def ocr_table_html_async(image: Image.Image, model: Optional[str] = None) -> str:
    client = _get_async_client()
    response = await client.chat.completions.create(
        model=model or REMOTE_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_image_to_data_url(image)}},
                    {
                        "type": "text",
                        "text": (
                            "Extract the table from the image.\n"
                            "Return ONLY valid HTML for the table (prefer <table>...</table>).\n"
                            "Do NOT include markdown fences or any extra text."
                        ),
                    },
                ],
            }
        ],
        temperature=REMOTE_OCR_TEMPERATURE,
    )
    return _strip_code_fences(response.choices[0].message.content or "")


def ocr_formula_latex(image: Image.Image, model: Optional[str] = None) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model or REMOTE_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_image_to_data_url(image)}},
                    {
                        "type": "text",
                        "text": (
                            "Extract the equation from the image.\n"
                            "Return ONLY the LaTeX expression.\n"
                            "Do NOT wrap it in $$ or \\( \\). Do NOT include any extra text."
                        ),
                    },
                ],
            }
        ],
        temperature=REMOTE_OCR_TEMPERATURE,
    )
    return _strip_code_fences(response.choices[0].message.content or "")


async def ocr_formula_latex_async(image: Image.Image, model: Optional[str] = None) -> str:
    client = _get_async_client()
    response = await client.chat.completions.create(
        model=model or REMOTE_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_image_to_data_url(image)}},
                    {
                        "type": "text",
                        "text": (
                            "Extract the equation from the image.\n"
                            "Return ONLY the LaTeX expression.\n"
                            "Do NOT wrap it in $$ or \\( \\). Do NOT include any extra text."
                        ),
                    },
                ],
            }
        ],
        temperature=REMOTE_OCR_TEMPERATURE,
    )
    return _strip_code_fences(response.choices[0].message.content or "")

