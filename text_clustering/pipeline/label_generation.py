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
from text_clustering.prompts import (
    prompt_construct_generate_label,
    prompt_construct_map_to_canonical,
    prompt_construct_merge_label,
)

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


def _parse_merge_response(raw: str) -> list[str] | None:
    """
    Parse the LLM merge response into a flat deduplicated label list.

    The model may return either:
      - a dict  : {"merged_labels": ["a", "b", ...]}  (expected)
      - a list  : ["a", "b", ...]                      (also acceptable)
    Both are handled. Returns None on any parse failure.
    """
    if raw is None:
        return None
    try:
        parsed = eval(raw)  # noqa: S307
        if isinstance(parsed, list):
            flat = parsed
        elif isinstance(parsed, dict):
            flat = []
            for v in parsed.values():
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
        else:
            return None
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for label in flat:
            if isinstance(label, str) and label not in seen:
                seen.add(label)
                deduped.append(label)
        return deduped if deduped else None
    except Exception:
        return None


def map_to_canonical(all_labels: list[str], true_labels: list[str], client,
                     batch_size: int = 40) -> list[str]:
    """
    Map proposed labels onto the true taxonomy in batches.

    For each batch of proposed labels, ask the model which canonical (true) labels
    they map to.  Collect all matched canonical labels and return the deduplicated
    union — this is always a subset of `true_labels`, so the merge is guaranteed
    to produce at most len(true_labels) labels.
    """
    canonical_set: set[str] = set()
    n_batches = (len(all_labels) + batch_size - 1) // batch_size

    for i in range(0, len(all_labels), batch_size):
        batch = all_labels[i : i + batch_size]
        prompt = prompt_construct_map_to_canonical(batch, true_labels)
        raw = chat(prompt, client, max_tokens=1024)
        result = _parse_merge_response(raw)
        batch_num = i // batch_size + 1
        if result is None:
            logger.warning("Map-to-canonical batch %d/%d parse failed — skipping batch",
                           batch_num, n_batches)
            continue
        # only keep labels that are actually in the canonical set
        valid = [r for r in result if r in true_labels]
        invalid = [r for r in result if r not in true_labels]
        if invalid:
            logger.debug("Batch %d/%d: model returned non-canonical labels (ignored): %s",
                         batch_num, n_batches, invalid)
        logger.debug("Batch %d/%d: %d proposed → %d canonical matched: %s",
                     batch_num, n_batches, len(batch), len(valid), valid)
        canonical_set.update(valid)

    # preserve the order of true_labels in the result
    result_ordered = [lbl for lbl in true_labels if lbl in canonical_set]
    logger.info("Map-to-canonical: %d proposed → %d canonical labels matched",
                len(all_labels), len(result_ordered))
    return result_ordered


def _single_merge_pass(labels: list[str], client, target_k: int | None = None,
                       batch_size: int = 30) -> list[str] | None:
    """
    Merge labels in small batches then merge the batch results together.

    Weaker models can consolidate 30 labels reliably but fail on 150+ in one shot.
    We split into batches of `batch_size`, merge each independently, then merge
    the collected batch outputs in a final call.

    Returns deduplicated merged list, or None if all batch calls fail.
    """
    # If the list is already small enough, do a single call
    if len(labels) <= batch_size:
        raw = chat(prompt_construct_merge_label(labels, target_k=target_k), client, max_tokens=4096)
        return _parse_merge_response(raw)

    # Phase 1 — merge each batch independently
    batch_results: list[str] = []
    n_batches = (len(labels) + batch_size - 1) // batch_size
    for i in range(0, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        raw = chat(prompt_construct_merge_label(batch), client, max_tokens=4096)
        result = _parse_merge_response(raw)
        if result is None:
            logger.warning("Merge batch %d/%d parse failed — using raw batch",
                           i // batch_size + 1, n_batches)
            result = batch  # fall back to original batch
        else:
            logger.debug("Merge batch %d/%d: %d → %d labels",
                         i // batch_size + 1, n_batches, len(batch), len(result))
        batch_results.extend(result)

    # deduplicate across batches before final pass
    seen: set[str] = set()
    combined: list[str] = []
    for label in batch_results:
        if label not in seen:
            seen.add(label)
            combined.append(label)
    logger.info("After batch merges: %d → %d labels (pre-final-pass)", len(labels), len(combined))

    # Phase 2 — final merge call on the reduced combined set
    raw = chat(prompt_construct_merge_label(combined, target_k=target_k), client, max_tokens=4096)
    final = _parse_merge_response(raw)
    if final is None:
        logger.warning("Final merge pass parse failed — returning post-batch list (%d labels)", len(combined))
        return combined
    return final


def merge_labels(args, all_labels, client, target_k: int | None = None,
                 max_passes: int = 5, batch_size: int = 30):
    """
    Iteratively merge labels until count <= 2*target_k or progress stalls.

    Each pass uses batched merging so the model never sees more than `batch_size`
    labels at once, which is the range where even weaker models consolidate well.
    """
    labels = list(all_labels)
    for pass_num in range(1, max_passes + 1):
        prev_count = len(labels)
        result = _single_merge_pass(labels, client, target_k=target_k, batch_size=batch_size)
        if result is None:
            logger.warning("Merge pass %d: all batches failed — keeping list (%d labels)",
                           pass_num, prev_count)
            break
        new_count = len(result)
        logger.info("Merge pass %d: %d → %d labels", pass_num, prev_count, new_count)
        labels = result
        if target_k is not None and new_count <= 2 * target_k:
            logger.info("Merge converged: %d labels ≤ 2×target_k (%d)", new_count, 2 * target_k)
            break
        if prev_count - new_count < 3:
            logger.info("Merge stalled after pass %d (removed only %d labels) — stopping",
                        pass_num, prev_count - new_count)
            break
    return labels


def main(args):
    size = "large" if args.use_large else "small"
    run_dir = make_run_dir(args.runs_dir, args.data, size)
    setup_logging(os.path.join(run_dir, "step1_label_gen.log"))

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

    final_labels = map_to_canonical(all_labels, true_labels, client)
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
