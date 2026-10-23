"""The only hosted model calls: Gemini, tried model by model.

A 429 (quota spent) or 503 (overloaded) moves on to the next configured
model, since free-tier quota is counted per model. Anything else is a real
error and surfaces at once.
"""
from __future__ import annotations

from functools import lru_cache

from google import genai
from google.genai import types

from . import config


class LLMError(RuntimeError):
    pass


def _transient(error: Exception) -> bool:
    text = str(error).lower()
    return any(m in text for m in ("429", "quota", "rate limit", "503",
                                   "unavailable", "overloaded", "high demand"))


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set (copy .env.example to .env).")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate(contents, system: str = "", json: bool = False,
             temperature: float = 0.0) -> str:
    """`contents` is a prompt string, or a list mixing strings and image
    parts. Thinking is off: answering from five passages needs no reasoning
    budget and runs ~5x faster without one."""
    last = None
    for model in config.GEMINI_MODELS:
        options = dict(
            system_instruction=system or None, temperature=temperature,
            response_mime_type="application/json" if json else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            try:
                response = _client().models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        **options))
            except Exception as error:
                # Some models reject a thinking budget outright; ask again
                # without one rather than give up on the model.
                if "invalid_argument" not in str(error).lower():
                    raise
                response = _client().models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(**options))
            return (response.text or "").strip()
        except Exception as error:
            if not _transient(error):
                raise LLMError(str(error).strip().splitlines()[0]) from error
            last = error
    raise LLMError(f"Every configured Gemini model is out of quota or busy: {last}")


def image_part(png: bytes) -> types.Part:
    return types.Part.from_bytes(data=png, mime_type="image/png")
