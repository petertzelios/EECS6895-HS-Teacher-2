from __future__ import annotations

import os
from typing import Dict, Optional

from openai import OpenAI


def create_llm_client(
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> OpenAI:
    if timeout is None:
        timeout_value = os.getenv("LLM_TIMEOUT_SECONDS")
        timeout = float(timeout_value) if timeout_value else None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def build_llm_client() -> Optional[OpenAI]:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("OPENROUTER_API_KEY"):
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not base_url and os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    default_headers = {}
    if base_url and "openrouter.ai" in base_url:
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_APP_TITLE", "HS Teacher Retrieval Demo")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if app_title:
            default_headers["X-Title"] = app_title

    timeout_value = os.getenv("LLM_TIMEOUT_SECONDS")
    timeout = float(timeout_value) if timeout_value else None

    return create_llm_client(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
        timeout=timeout,
    )


def _extract_chat_text(resp) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def complete_text(
    client: Optional[OpenAI],
    prompt: str,
    model: str,
    reasoning_effort: str = "low",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if client is None:
        raise RuntimeError("No LLM client available. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    try:
        kwargs = {
            "model": model,
            "input": prompt,
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.create(**kwargs)
        return (resp.output_text or "").strip()
    except Exception:
        chat_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            chat_kwargs["temperature"] = temperature
        if max_tokens is not None:
            chat_kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**chat_kwargs)
        return _extract_chat_text(resp)
