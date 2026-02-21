"""
client.py — OpenAI-compatible client factory for OpenRouter (and any compatible provider).

Thin wrapper that loads .env and returns a configured openai.OpenAI client.

The original pipeline scripts call OpenAI() with no arguments and rely on
OPENAI_API_KEY / OPENAI_BASE_URL being set in the environment. This module
sets those variables from .env before any script runs, so the rest of the
code works without modification whether you are using OpenRouter, Groq, or
a direct OpenAI key.

Usage
-----
    from text_clustering.client import make_client

    client = make_client()
    response = client.chat.completions.create(model=..., ...)

Run as a script to smoke-test the configuration:

    python -m text_clustering.client
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# __file__ is text_clustering/client.py — go one level up to reach the project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)


def make_client() -> OpenAI:
    """Build and return an OpenAI-compatible client.

    If OPENAI_BASE_URL is set to an OpenRouter endpoint, the optional
    HTTP-Referer and X-Title headers are injected as required by OpenRouter's
    attribution guidelines.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    extra_headers: dict[str, str] = {}
    if provider == "openrouter":
        if site_url := os.getenv("OR_SITE_URL", ""):
            extra_headers["HTTP-Referer"] = site_url
        if app_name := os.getenv("OR_APP_NAME", "text-clustering-llm"):
            extra_headers["X-Title"] = app_name

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
        default_headers=extra_headers or None,
    )


# Resolved at import time so other modules can do:
#   from text_clustering.client import MODEL
# instead of reading the env var themselves.
MODEL: str = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))


if __name__ == "__main__":
    print(f"Provider  : {os.getenv('LLM_PROVIDER', 'openrouter')}")
    print(f"Base URL  : {os.getenv('OPENAI_BASE_URL', '(openai default)')}")
    print(f"Model     : {MODEL}")
    print(f"Temp      : {TEMPERATURE}  |  Max tokens: {MAX_TOKENS}")
    print("\nSending smoke-test request...")
    client = make_client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(f"Response  : {resp.choices[0].message.content}")
    print("Smoke test passed.")
