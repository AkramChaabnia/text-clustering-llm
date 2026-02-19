"""Drop-in adapter: loads .env and returns an openai.OpenAI client pre-configured
for OpenRouter (or plain OpenAI if LLM_PROVIDER=openai).

Usage (inside any script):

    from openrouter_adapter import make_client, MODEL

    client = make_client()

The original scripts use OpenAI() with no arguments — the openai SDK reads
OPENAI_API_KEY and OPENAI_BASE_URL from the environment automatically.
This file just ensures those env vars are set correctly before any script runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env from project root (works regardless of cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env", override=False)  # don't override already-set vars


def make_client() -> OpenAI:
    """Build and return an OpenAI-compatible client.

    - If OPENAI_BASE_URL is set (e.g. OpenRouter), it is used automatically
      by the openai SDK via the environment.
    - Extra OpenRouter headers (HTTP-Referer, X-Title) are injected via
      default_headers so every request is properly attributed.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    extra_headers: dict[str, str] = {}
    if provider == "openrouter":
        site_url = os.getenv("OR_SITE_URL", "")
        app_name = os.getenv("OR_APP_NAME", "text-clustering-llm")
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),  # None -> uses OpenAI default
        default_headers=extra_headers or None,
    )
    return client


# ---------------------------------------------------------------------------
# Resolved model name -- scripts can import this instead of hardcoding
# ---------------------------------------------------------------------------
MODEL: str = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))


# ---------------------------------------------------------------------------
# Quick smoke-test: python openrouter_adapter.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Provider : {os.getenv('LLM_PROVIDER', 'openrouter')}")
    print(f"Base URL : {os.getenv('OPENAI_BASE_URL', '(openai default)')}")
    print(f"Model    : {MODEL}")
    print(f"Temp     : {TEMPERATURE}  |  Max tokens: {MAX_TOKENS}")

    client = make_client()
    print("\nSending smoke-test message to API...")
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(f"Response : {resp.choices[0].message.content}")
    print("Smoke test passed.")
