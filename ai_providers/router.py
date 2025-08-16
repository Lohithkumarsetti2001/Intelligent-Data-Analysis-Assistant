
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

from .ollama_provider import ollama_generate
from .hf_provider import hf_generate

@dataclass
class AIResponse:
    text: str
    provider: str
    meta: dict

def multi_ai_generate(prompt: str, system: Optional[str] = None, **kwargs) -> AIResponse:
    """Try local Ollama first, then fall back to Hugging Face (if token provided)."""
    # 1) Ollama
    try:
        text, meta = ollama_generate(prompt=prompt, system=system, **kwargs)
        if text:
            return AIResponse(text=text, provider="ollama", meta=meta)
    except Exception as e:
        last_error = f"Ollama error: {e}"
    # 2) Hugging Face Inference
    if os.getenv("HF_API_TOKEN"):
        try:
            text, meta = hf_generate(prompt=prompt, system=system, **kwargs)
            if text:
                return AIResponse(text=text, provider="huggingface", meta=meta)
        except Exception as e:
            last_error = f"Hugging Face error: {e}"

    return AIResponse(text=f"(fallback) Could not reach any LLM. Last error: {locals().get('last_error','none')}", provider="none", meta={})
