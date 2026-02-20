"""
llm.py — LLM client initialisation and chat helper shared by all pipeline steps.

Functions
---------
ini_client()
    Return a configured OpenAI-compatible client via openrouter_adapter.

_strip_fenced_json(text)
    Strip markdown code fences that some models wrap their JSON output in.

chat(prompt, client)
    Send a single-turn chat prompt and return the raw text response.
    Retries up to 5 times on HTTP 429 with linear back-off.
"""

import logging
import re
import time

from text_clustering.config import FORCE_JSON_MODE, MAX_TOKENS, MODEL, REQUEST_DELAY, TEMPERATURE

logger = logging.getLogger(__name__)


def ini_client():
    from text_clustering.client import make_client

    return make_client()


def _strip_fenced_json(text: str) -> str:
    """Remove markdown code fences if the model wrapped its JSON output in them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def chat(prompt, client):
    kwargs = dict(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    if FORCE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
            return _strip_fenced_json(raw)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 20 * (attempt + 1)
                logger.warning("rate limit — attempt %d/5, retrying in %ds...", attempt + 1, wait)
                time.sleep(wait)
            else:
                logger.error("api error: %s", e)
                return None
    return None
