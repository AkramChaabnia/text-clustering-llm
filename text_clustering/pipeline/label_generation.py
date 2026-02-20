"""
label_generation.py — Step 1 of the clustering pipeline.

For each dataset, the script:
  1. Loads the dataset and shuffles it.
  2. Reads the seed labels produced by seed_labels.py (chosen_labels.json).
  3. Iterates over the texts in chunks of --chunk_size (default 15).
     For each chunk, it asks the LLM to propose new label names for texts
     that don't fit any existing label.
  4. Merges the resulting label set with a second LLM call that deduplicates
     near-synonymous labels.
  5. Writes all outputs to a timestamped run directory under --runs_dir:
       runs/<data>_<size>_<timestamp>/
         labels_true.json           ground-truth label list
         labels_proposed.json       all labels before merge
         labels_merged.json         merged/deduplicated labels  ← used by Step 2

Run directory path is printed on start so Step 2 can be pointed at it via --run_dir.

The LLM is configured entirely through environment variables (see .env.example).
Model selection, temperature, and token limits are read at startup — no code
change is needed to switch models.

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - ini_client() wired to text_clustering.client.make_client()
  - response_format disabled by default (LLM_FORCE_JSON_MODE=false)
  - _strip_fenced_json() strips markdown fences before eval()
  - chat() retries up to 5 times with backoff on 429 errors
  - model read from LLM_MODEL env var
  - outputs written to timestamped run directories under ./runs/
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

from text_clustering.data import get_label_list, load_dataset
from text_clustering.llm import chat, ini_client
from text_clustering.logging_config import setup_logging
from text_clustering.prompts import prompt_construct_generate_label, prompt_construct_merge_label

logger = logging.getLogger(__name__)


def make_run_dir(runs_dir: str, data: str, size: str) -> str:
    """Create and return a timestamped run directory path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{data}_{size}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %s", path)


def get_sentences(sentence_list):
    return [item["input"] for item in sentence_list]


def label_generation(args, client, data_list, chunk_size):
    with open(args.given_label_path, "r") as f:
        given_labels = json.load(f)

    all_labels = list(given_labels.get(args.data, []))
    count = 0

    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i : i + chunk_size]
        sentences = get_sentences(chunk)
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        raw = chat(prompt, client)
        if raw is None:
            continue
        count += 1
        try:
            response = eval(raw)  # noqa: S307
        except Exception:
            continue

        first_key = list(response.keys())[0]
        if isinstance(response[first_key], list):
            for label in response[first_key]:
                if "unknown_topic" in label or "new_label" in label:
                    continue
                if label not in all_labels:
                    all_labels.append(label)
        else:
            all_labels.append(response[first_key])

        if args.print_details:
            print(f"Prompt:\n{prompt}")
            print(f"Raw response: {raw}")
            print(f"Parsed: {response}")
            print(f"Label count so far: {len(all_labels)}")
            if count >= args.test_num:
                break
        elif count % 10 == 0:
            logger.debug("chunk %d — label count so far: %d", count, len(all_labels))

    return all_labels


def merge_labels(args, all_labels, client, target_k: int | None = None):
    prompt = prompt_construct_merge_label(all_labels, target_k=target_k)
    response = chat(prompt, client, max_tokens=4096)
    try:
        response = eval(response)  # noqa: S307
        merged = []
        for sub_list in response.values():
            merged.extend(sub_list)
        return merged
    except Exception:
        return all_labels


def main(args):
    size = "large" if args.use_large else "small"
    run_dir = make_run_dir(args.runs_dir, args.data, size)
    setup_logging(os.path.join(run_dir, "run.log"))

    logger.info("=== Step 1 — Label Generation ===")
    logger.info("Dataset : %s  |  split: %s", args.data, size)
    logger.info("Run dir : %s", run_dir)
    logger.info("  (pass --run_dir %s to Step 2)", run_dir)
    start = time.time()

    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    random.shuffle(data_list)

    true_labels = get_label_list(data_list)
    logger.info("True cluster count: %d", len(true_labels))
    write_json(os.path.join(run_dir, "labels_true.json"), true_labels)

    all_labels = label_generation(args, client, data_list, args.chunk_size)
    logger.info("Labels proposed (before merge): %d", len(all_labels))
    write_json(os.path.join(run_dir, "labels_proposed.json"), all_labels)

    final_labels = merge_labels(args, all_labels, client, target_k=len(true_labels))
    write_json(os.path.join(run_dir, "labels_merged.json"), final_labels)
    logger.info("Labels after merge: %d", len(final_labels))

    ratio = len(final_labels) / len(true_labels)
    if ratio > 2:
        logger.warning(
            "Merged label count (%d) is %.1fx the true class count (%d). "
            "Classification results will not be comparable to the paper baseline. "
            "Consider re-running Step 1 before proceeding to Step 2.",
            len(final_labels), ratio, len(true_labels),
        )

    logger.info("Done in %.1fs", time.time() - start)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 1: LLM label generation and merge.")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--runs_dir", type=str, default="./runs",
                        help="Root directory where timestamped run folders are created")
    parser.add_argument("--given_label_path", type=str, default="./runs/chosen_labels.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=15)
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())


def main_cli():
    """Console-script entry point (registered in pyproject.toml)."""
    main(build_parser().parse_args())
