"""
sealclust_v4.py — SEAL-Clust v4 core algorithms.

v4 is a **prompt-only** improvement over v3.  The pipeline structure,
stages, and all non-LLM logic are identical to v3.  The three LLM stages
use redesigned prompts (from ``prompts.py``) that:

  - **Discovery** (Stage 5): allow 1-3 word descriptive labels instead of
    strict ONE-WORD, encouraging finer granularity.
  - **Consolidation** (Stage 7): structured semantic grouping with explicit
    merge-vs-keep reasoning.
  - **Classification** (Stage 8): contrastive disambiguation — prefer the
    most specific label when categories overlap.

Re-exported unchanged from v3
------------------------------
  run_overclustering, select_representatives, propagate_labels_v3

New v4 functions
----------------
  discover_labels_v4    — wraps v3 machinery, swaps prompt function
  consolidate_labels_v4 — wraps v3 machinery, swaps prompt function
  classify_representatives_v4 — wraps v3 machinery, swaps prompt function
"""

from __future__ import annotations

import logging
import math
import os
import random

from text_clustering.sealclust_v3 import (
    _load_checkpoint,
    _remove_checkpoint,
    _safe_parse_dict,
    _safe_parse_labels,
    _save_checkpoint,
    _trim_labels_by_similarity,
    propagate_labels_v3,
    run_overclustering,
    select_representatives,
)

logger = logging.getLogger(__name__)

# Re-export unchanged v3 functions so the pipeline can import from v4
__all__ = [
    "run_overclustering",
    "select_representatives",
    "discover_labels_v4",
    "consolidate_labels_v4",
    "classify_representatives_v4",
    "propagate_labels_v3",
]


# ---------------------------------------------------------------------------
# Stage 5 — Label Discovery (v4 prompts)
# ---------------------------------------------------------------------------

def _run_discovery_pass_v4(
    texts: list[str],
    client,
    chunk_size: int,
    existing_labels: list[str],
    start_chunk: int,
    n_chunks: int,
    ckpt_path: str | None,
    ckpt_interval: int,
    pass_label: str = "pass-1",
    dataset_description: str = "",
) -> list[str]:
    """Run one pass of v4 label discovery over chunked texts."""
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v4_discover_labels

    all_labels = list(existing_labels)

    for chunk_num in range(n_chunks):
        if chunk_num < start_chunk:
            continue

        start = chunk_num * chunk_size
        chunk = texts[start : start + chunk_size]
        prompt = prompt_v4_discover_labels(
            chunk, existing_labels=all_labels,
            dataset_description=dataset_description,
        )
        raw = chat(prompt, client, max_tokens=4096)
        if raw is None:
            logger.warning(
                "  [%s] Chunk %d: LLM returned None — skipping",
                pass_label, chunk_num + 1,
            )
            continue

        parsed = _safe_parse_labels(raw)
        if parsed:
            for label in parsed:
                if label not in all_labels:
                    all_labels.append(label)

        logger.info(
            "  [%s] Chunk %d/%d — labels so far: %d",
            pass_label, chunk_num + 1, n_chunks, len(all_labels),
        )

        if ckpt_path and (chunk_num + 1) % ckpt_interval == 0:
            _save_checkpoint(ckpt_path, {
                "processed_chunks": chunk_num + 1,
                "all_labels": all_labels,
            })

    return all_labels


def discover_labels_v4(
    representative_texts: list[str],
    client,
    chunk_size: int = 30,
    run_dir: str | None = None,
    min_labels: int = 0,
    max_retries: int = 3,
    dataset_description: str = "",
) -> list[str]:
    """Stage 5: LLM label discovery — v4.

    Same retry/checkpoint logic as v3 but uses v4 prompts.
    When *dataset_description* is provided it is forwarded to every
    prompt call so the LLM understands the domain.

    Parameters
    ----------
    representative_texts : list[str]
    client : OpenAI client
    chunk_size : int
    run_dir : str | None
    min_labels : int
    max_retries : int
    dataset_description : str

    Returns
    -------
    list[str]
        Unique candidate labels discovered.
    """
    all_labels: list[str] = []
    n_chunks = math.ceil(len(representative_texts) / chunk_size)

    # Checkpoint resume
    start_chunk = 0
    ckpt_path = (
        os.path.join(run_dir, "checkpoint_v4_labels.json") if run_dir else None
    )
    if ckpt_path:
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None:
            start_chunk = ckpt["processed_chunks"]
            all_labels = ckpt["all_labels"]
            logger.info(
                "[checkpoint] Resuming v4 label discovery from chunk %d/%d",
                start_chunk + 1, n_chunks,
            )

    ckpt_interval = max(2, n_chunks // 10)

    logger.info(
        "Stage 5 (v4): Discovering 1-3 word labels from %d reps in %d chunks",
        len(representative_texts), n_chunks,
    )

    all_labels = _run_discovery_pass_v4(
        representative_texts, client, chunk_size, all_labels,
        start_chunk, n_chunks, ckpt_path, ckpt_interval,
        dataset_description=dataset_description,
    )

    # ── Retry passes if we haven't reached min_labels ──
    if min_labels > 0 and len(all_labels) < min_labels:
        logger.warning(
            "Stage 5 (v4): Only %d labels discovered but min_labels=%d "
            "— starting retry passes (max %d)",
            len(all_labels), min_labels, max_retries,
        )
        for retry in range(1, max_retries + 1):
            if len(all_labels) >= min_labels:
                break

            shuffled = list(representative_texts)
            random.shuffle(shuffled)

            retry_chunk_size = max(5, chunk_size // (retry + 1))
            n_retry_chunks = math.ceil(len(shuffled) / retry_chunk_size)

            logger.info(
                "Stage 5 (v4): Retry %d/%d — reshuffled %d reps into "
                "%d chunks (chunk_size=%d), have %d/%d labels",
                retry, max_retries, len(shuffled), n_retry_chunks,
                retry_chunk_size, len(all_labels), min_labels,
            )

            all_labels = _run_discovery_pass_v4(
                shuffled, client, retry_chunk_size, all_labels,
                start_chunk=0, n_chunks=n_retry_chunks,
                ckpt_path=None, ckpt_interval=n_retry_chunks,
                pass_label=f"retry-{retry}",
                dataset_description=dataset_description,
            )

        if len(all_labels) < min_labels:
            logger.warning(
                "Stage 5 (v4): After %d retries still only %d labels "
                "(target %d) — proceeding with what we have",
                max_retries, len(all_labels), min_labels,
            )

    logger.info("Stage 5 (v4): Final count — %d unique candidate labels", len(all_labels))

    if ckpt_path:
        _remove_checkpoint(ckpt_path)

    return all_labels


# ---------------------------------------------------------------------------
# Stage 7 — Label Consolidation (v4 prompts)
# ---------------------------------------------------------------------------

def consolidate_labels_v4(
    candidate_labels: list[str],
    k_star: int,
    client,
    chunk_size: int = 200,
    max_rounds: int = 10,
    dataset_description: str = "",
) -> list[str]:
    """Stage 7: Merge candidate labels into exactly K* final labels (v4).

    Same iterative chunked merge + deterministic trim as v3 but uses
    the v4 consolidation prompt with structured semantic grouping.

    Parameters
    ----------
    candidate_labels : list[str]
    k_star : int
    client : OpenAI client
    chunk_size : int
    max_rounds : int
    dataset_description : str

    Returns
    -------
    list[str]
        Exactly K* merged labels.
    """
    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v4_consolidate_labels

    current = list(dict.fromkeys(candidate_labels))  # deduplicate
    logger.info(
        "Stage 7 (v4): Consolidating %d → %d labels",
        len(current), k_star,
    )

    for round_num in range(1, max_rounds + 1):
        n = len(current)

        if n <= k_star:
            logger.info("  Round %d: already at %d ≤ %d", round_num, n, k_star)
            break

        # Small enough for a single final merge
        if n <= chunk_size:
            logger.info(
                "  Round %d: final merge %d → %d (single call)",
                round_num, n, k_star,
            )
            best = current
            best_dist = abs(len(current) - k_star)
            cumulative_overshoot = 0
            max_attempts = 8
            for attempt in range(1, max_attempts + 1):
                adjusted_target = max(
                    3, k_star - cumulative_overshoot // 2,
                )
                prompt = prompt_v4_consolidate_labels(
                    current, adjusted_target,
                    dataset_description=dataset_description,
                )
                raw = chat(prompt, client, max_tokens=4096)
                parsed = _safe_parse_labels(raw)
                if parsed and len(parsed) < len(current):
                    dist = abs(len(parsed) - k_star)
                    overshoot = len(parsed) - k_star
                    logger.info(
                        "    attempt %d (asked %d): got %d labels (off by %d)",
                        attempt, adjusted_target, len(parsed), dist,
                    )
                    if dist < best_dist:
                        best = parsed
                        best_dist = dist
                    if dist == 0:
                        break
                    if overshoot > 0:
                        cumulative_overshoot += overshoot
                else:
                    logger.info("    attempt %d: parse failed or no reduction", attempt)
                if best_dist == 0:
                    break
            current = best
            break

        # Chunked aggressive reduction
        n_chunks = math.ceil(n / chunk_size)
        per_chunk_target = max(3, int(k_star * 1.1) // n_chunks)
        merged_round: list[str] = []

        logger.info(
            "  Round %d: %d labels → %d chunks of ~%d (each → ~%d)",
            round_num, n, n_chunks, chunk_size, per_chunk_target,
        )

        for i in range(n_chunks):
            chunk = current[i * chunk_size : (i + 1) * chunk_size]
            chunk_target = max(3, int(per_chunk_target * len(chunk) / chunk_size))
            prompt = prompt_v4_consolidate_labels(
                chunk, chunk_target,
                dataset_description=dataset_description,
            )
            raw = chat(prompt, client, max_tokens=4096)
            parsed = _safe_parse_labels(raw)
            result = parsed if parsed and len(parsed) < len(chunk) else chunk
            merged_round.extend(result)
            logger.info(
                "    chunk %d/%d: %d → %d",
                i + 1, n_chunks, len(chunk), len(result),
            )

        current = list(dict.fromkeys(merged_round))
        logger.info(
            "  Round %d done: %d → %d (target: %d)",
            round_num, n, len(current), k_star,
        )

        if len(current) >= n:
            logger.warning("  No reduction in round %d — stopping", round_num)
            break

    # ── Deterministic post-processing: guarantee exactly K* ──
    n_after_llm = len(current)
    if n_after_llm > k_star:
        logger.info(
            "Stage 7 (v4): LLM produced %d labels (target %d) — "
            "trimming %d excess labels by embedding similarity",
            n_after_llm, k_star, n_after_llm - k_star,
        )
        current = _trim_labels_by_similarity(current, k_star)
    elif n_after_llm < k_star:
        logger.warning(
            "Stage 7 (v4): LLM produced only %d labels (target %d) — "
            "cannot add labels without context, returning %d",
            n_after_llm, k_star, n_after_llm,
        )

    logger.info(
        "Stage 7 (v4): Final label count: %d (target was %d)",
        len(current), k_star,
    )
    return current


# ---------------------------------------------------------------------------
# Stage 8 — Representative Classification (v4 prompts)
# ---------------------------------------------------------------------------

def classify_representatives_v4(
    representative_texts: list[str],
    final_labels: list[str],
    client,
    batch_size: int = 20,
    run_dir: str | None = None,
    dataset_description: str = "",
) -> dict[int, str]:
    """Stage 8: Classify each representative into one of K* labels (v4).

    Same batched logic + checkpoint as v3 but uses v4 prompts.

    Parameters
    ----------
    representative_texts : list[str]
    final_labels : list[str]
    client : OpenAI client
    batch_size : int
    run_dir : str | None
    dataset_description : str

    Returns
    -------
    dict[int, str]
        Mapping from representative index → assigned label.
    """
    import random

    from text_clustering.llm import chat
    from text_clustering.prompts import prompt_v4_classify_representatives_batch

    n_reps = len(representative_texts)
    rep_labels: dict[int, str] = {}

    # Checkpoint resume
    start_idx = 0
    ckpt_path = (
        os.path.join(run_dir, "checkpoint_v4_classify.json") if run_dir else None
    )
    if ckpt_path:
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None:
            start_idx = ckpt["processed_count"]
            rep_labels = {int(k): v for k, v in ckpt["rep_labels"].items()}
            logger.info(
                "[checkpoint] Resuming v4 classification from %d/%d",
                start_idx, n_reps,
            )

    ckpt_interval = max(1, n_reps // (batch_size * 10))

    logger.info(
        "Stage 8 (v4): Classifying %d representatives into %d labels (batch_size=%d)",
        n_reps, len(final_labels), batch_size,
    )

    batch_num = 0
    for i in range(0, n_reps, batch_size):
        if i < start_idx:
            continue

        batch = representative_texts[i : i + batch_size]
        # Shuffle label order per batch to avoid positional bias
        shuffled_labels = final_labels.copy()
        random.shuffle(shuffled_labels)
        prompt = prompt_v4_classify_representatives_batch(shuffled_labels, batch, dataset_description=dataset_description)
        raw = chat(prompt, client, max_tokens=4096)

        if raw:
            parsed = _safe_parse_dict(raw)
            if parsed:
                for key, label in parsed.items():
                    idx = i + int(key) - 1  # keys are 1-indexed
                    if label in final_labels:
                        rep_labels[idx] = label
                    else:
                        # Fuzzy match: pick closest label
                        label_lower = label.lower()
                        for fl in final_labels:
                            if fl.lower() == label_lower:
                                rep_labels[idx] = fl
                                break
                        else:
                            rep_labels[idx] = final_labels[0]

        batch_num += 1
        classified = len(rep_labels)
        logger.info(
            "  Batch %d: classified %d / %d representatives",
            batch_num, classified, n_reps,
        )

        if ckpt_path and batch_num % ckpt_interval == 0:
            _save_checkpoint(ckpt_path, {
                "processed_count": i + len(batch),
                "rep_labels": rep_labels,
            })

    # Fill any missing
    for idx in range(n_reps):
        if idx not in rep_labels:
            rep_labels[idx] = "Unsuccessful"
            logger.warning("  Representative %d: no label assigned → Unsuccessful", idx)

    logger.info(
        "Stage 8 (v4): Classified %d / %d representatives",
        sum(1 for v in rep_labels.values() if v != "Unsuccessful"), n_reps,
    )

    if ckpt_path:
        _remove_checkpoint(ckpt_path)

    return rep_labels
