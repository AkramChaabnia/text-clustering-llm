"""
logging_config.py — logging setup for the text_clustering package.

Each pipeline step calls setup_logging(log_path) once at startup.
After that, every module uses logging.getLogger(__name__) normally.

Log format
----------
  2026-02-20 14:32:01 | INFO     | label_generation | Run dir: ./runs/...

Both a StreamHandler (stdout, INFO) and a FileHandler (DEBUG) are attached
to the root logger so every log call is captured in the step's own log file.

Log files per step
------------------
  runs/seed_labels.log                      ← Step 0
  runs/<run_dir>/step1_label_gen.log        ← Step 1
  runs/<run_dir>/step2_classification.log   ← Step 2
  runs/<run_dir>/step3_evaluation.log       ← Step 3
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_path: str | Path) -> None:
    """
    Configure the root logger to write to stdout and to *log_path*.

    Call this once at the start of each pipeline step's main(), passing
    the path to the log file (usually ``<run_dir>/run.log``).
    Calling it again (e.g. in tests) replaces the existing handlers.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any handlers added by a previous call or by basicConfig
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
