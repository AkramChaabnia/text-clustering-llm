"""
config.py — Environment-variable configuration shared by all pipeline steps.

All LLM-related settings are read once at import time from the environment
(populated by python-dotenv / .env).  Any pipeline module that needs these
values imports them from here instead of re-reading os.environ itself.
"""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
FORCE_JSON_MODE: bool = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"
REQUEST_DELAY: float = float(os.getenv("LLM_REQUEST_DELAY", "0"))
