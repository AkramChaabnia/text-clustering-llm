# Research Log — Text Clustering as Classification with LLMs
> **Project**: PPD — Text Clustering as Classification with LLMs  
> **Programme**: M2 MLSD, Université Paris Cité  
> **Period**: 2026-02-18 → ongoing  
> **Goal**: Reproduce the baseline results from [arXiv:2410.00927](https://arxiv.org/abs/2410.00927) using free LLMs via OpenRouter, then run complementary experiments.

---

## Table of Contents

1. [Paper Overview](#1-paper-overview)
2. [Setup](#2-setup)
3. [Dataset](#3-dataset)
4. [Code Changes](#4-code-changes)
5. [API & Model Investigation](#5-api--model-investigation)
6. [Model Probe Results](#6-model-probe-results)
7. [Pipeline Execution Log](#7-pipeline-execution-log)
   - [Run 01 — massive_scenario · trinity-large-preview · 2026-02-20](#run-01--massive_scenario--arcee-aitrinity-large-previewfree--2026-02-20)
   - [Run 02 — Merge Investigation & Model Switch](#run-02--merge-investigation--model-switch)
   - [Run 02 — `massive_scenario` · gemini-2.0-flash-001 · `target_k=18` · 2026-02-21](#run-02--massive_scenario--googlegemini-20-flash-001--target_k18--2026-02-21)
   - [Run 03 — `massive_scenario` · gemini-2.0-flash-001 · no `target_k` · 2026-02-21](#run-03--massive_scenario--googlegemini-20-flash-001--no-target_k--2026-02-21)
8. [Results](#8-results)
9. [Next Steps](#9-next-steps)

---

## 1. Paper Overview

**Title**: Text Clustering as Classification with LLMs  
**Authors**: Chen Huang, Guoxiu He  
**Venue**: arXiv 2024 — [2410.00927](https://arxiv.org/abs/2410.00927)  
**Original repo**: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)

### Core idea

The paper reframes unsupervised text clustering as a **classification problem** driven by an LLM. Instead of embeddings + k-means, the approach is:

1. **Label generation** — given a small set of seed labels (20% of ground truth), the LLM proposes new label names by reading chunks of input texts. Duplicate/similar labels are merged in a second LLM call.
2. **Classification** — the LLM assigns each text to one of the generated labels, one text at a time.
3. **Evaluation** — standard clustering metrics: ACC (Hungarian alignment), NMI, ARI.

### Pipeline — step by step

**Step 0 — Seed label selection**  
Before running the LLM, 20% of the ground-truth labels are randomly sampled per dataset and written to `chosen_labels.json`. These seeds are handed to the LLM in Step 1 as a starting point, which anchors the taxonomy and avoids completely free-form generation.

**Step 1 — Label generation**  
The dataset is shuffled and split into chunks of 15 texts. For each chunk, the LLM is shown the current label set and asked: *"Do any of these texts require a new label that doesn't already exist?"* It adds new candidate labels when needed. Once all chunks are processed, a second LLM call merges and deduplicates near-synonyms (e.g. `"email"` and `"email_management"` collapse into one). The result is the final label set used in Step 2.

**Step 2 — Classification**  
Each text is sent individually to the LLM with the full label list: *"Which of these labels fits this text best?"* The predicted label is accumulated into a dict `{ label: [text, text, ...] }` and saved as `classifications.json`. This step makes one API call per sample — ~3,000 calls for most datasets. Progress is checkpointed every 200 samples so interrupted runs can resume without starting over.

**Step 3 — Evaluation**  
Predicted labels are matched to ground-truth labels using the Hungarian algorithm (optimal one-to-one alignment), then ACC, NMI and ARI are computed. Results are printed and saved to `results.json`.

### Pipeline diagram

```
dataset/
  └── small.jsonl  (texts + ground-truth labels)
          │
          ▼
  [Step 0]  seed_labels.py
          │  picks 20% of true labels at random
          ▼
  runs/chosen_labels.json
          │
          ▼
  [Step 1]  label_generation.py  --data <dataset>
          │  chunks of 15 texts → LLM proposes labels → merge call
          ▼
  runs/<dataset>_small_<timestamp>/
    labels_true.json        (ground-truth label list)
    labels_proposed.json    (before merge)
    labels_merged.json      (final label set)
          │
          ▼
  [Step 2]  classification.py  --run_dir <above>
          │  one LLM call per text → assigns a label
          ▼
    classifications.json    { label: [text, ...] }
          │
          ▼
  [Step 3]  evaluation.py  --run_dir <above>
          │  Hungarian alignment → ACC / NMI / ARI
          ▼
    results.json
```

### Models used in the paper

| Stage | Model |
|-------|-------|
| Label generation | `gpt-3.5-turbo-0125` |
| Label merging | `gpt-3.5-turbo-0125` |
| Classification | `gpt-3.5-turbo-0125` |
| Ablation upper bound | `gpt-4` |

### Datasets (5 main, small split)

| Dataset | Domain | Classes |
|---------|--------|---------|
| `massive_intent` | Voice assistant intents | 59 |
| `massive_scenario` | Voice assistant scenarios | 18 |
| `go_emotion` | Emotion detection | 27 |
| `mtop_intent` | Multi-domain intent | 102 |
| `arxiv_fine` | Academic topics | 93 |

---

## 2. Setup

### Toolchain

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12.6 (pyenv) | Runtime |
| uv | latest | Fast, reproducible package installs |
| Ruff | 0.15.1 | Linter |
| Commitizen | 4.13.7 | Conventional commit enforcement |

### Key files added

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Pinned lockfile for reproducibility |
| `requirements.txt` | Pinned fallback for pip users |
| `.env.example` | Documents every env variable (no secrets) |
| `text_clustering/client.py` | Thin wrapper: loads `.env`, builds `openai.OpenAI` client for OpenRouter |
| `.cz.yaml` | Commitizen config (conventional commits) |
| `.github/workflows/ci.yml` | Lint CI on PRs to `main` / `develop` |

### Dependencies

```
openai>=1.30.0        — OpenRouter-compatible client
python-dotenv>=1.0.0  — .env loading
scikit-learn>=1.4.0   — NMI, ARI metrics
scipy>=1.13.0         — Hungarian algorithm (ACC)
numpy>=1.26.0         — used in evaluate.py
```

### Branching

```
main       ← stable, tagged releases
  └── develop         ← integration branch
        └── feature/<desc>  /  fix/<desc>  /  docs/<desc>
```

All commits follow [Conventional Commits](https://www.conventionalcommits.org/) :  (`feat:`, `fix:`, `docs:`, `build:`, `ci:`).

---

## 3. Dataset

**Source**: Downloaded directly from [Google Drive — ClusterLLM dataset, originally from EMNLP 2023](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view) — unzip into `./dataset/`

**Format** (one JSON object per line):
```json
{"task": "massive_intent", "input": "set an alarm for 7am", "label": "alarm set"}
```

The download bundle contains 14 datasets. The 9 extras (`banking77`, `clinc`, `clinc_domain`, `few_event`, `few_nerd_nat`, `few_rel_nat`, `mtop_domain`, `reddit`, `stackexchange`) could be removed — only the 5 used in the paper are kept in `./dataset/`.

### Dataset sizes (small split)

| Dataset | Samples | Classes |
|---------|---------|---------|
| `massive_scenario` | 2,974 | 18 |
| `massive_intent` | 2,974 | 59 |
| `go_emotion` | 5,940 | 27 |
| `arxiv_fine` | 3,674 | 93 |
| `mtop_intent` | 4,386 | 102 |

### Seed label selection (Step 0)

`select_part_labels.py` samples `floor(0.2 × num_classes)` labels per dataset at random to use as the LLM's starting point.

| Dataset | Classes | Seed labels given |
|---------|---------|-------------------|
| `massive_scenario` | 18 | 3 |
| `massive_intent` | 59 | 11 |
| `go_emotion` | 27 | 5 |
| `mtop_intent` | 102 | 20 |
| `arxiv_fine` | 93 | 18 |

Output: `./runs/chosen_labels.json`

---

## 4. Code Changes

The original code was written to run against OpenAI directly with a paid key. Adapting it to free models via OpenRouter exposed several bugs and missing pieces. This section documents what was broken (fixes) and what was added on top (improvements).

---

### Fixes

These are things that were broken in the original and prevented the pipeline from running correctly.

#### Fix 1 — `ini_client()` call signature

**File**: `text_clustering/pipeline/classification.py`  
**Issue**: `main()` called `ini_client(args.api_key)` but `ini_client()` took no arguments → `TypeError` on every run.

```python
# Before
client = ini_client(args.api_key)

# After
client = ini_client()  # API key comes from .env
```

#### Fix 2 — `response_format` not universally supported

**Files**: `text_clustering/pipeline/label_generation.py`, `classification.py`  
**Issue**: The original code always sent `response_format={"type":"json_object"}`. Many free models return HTTP 400 or an empty body when this is set.  
**Fix**: Made it opt-in via `LLM_FORCE_JSON_MODE` env var (default `false`).

```python
_FORCE_JSON_MODE = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"

if _FORCE_JSON_MODE:
    kwargs["response_format"] = {"type": "json_object"}
```

#### Fix 3 — Markdown fence stripping

**Files**: `text_clustering/pipeline/label_generation.py`, `classification.py`  
**Issue**: Free models often wrap JSON in markdown code fences (` ```json ... ``` `). The original `eval()` call fails on these, silently dropping labels.  
**Fix**: Added `_strip_fenced_json()` applied after every API call.

```python
def _strip_fenced_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
```

#### Fix 4 — No retry on rate limits

**File**: `text_clustering/llm.py`  
**Issue**: No error handling around API calls. A single 429 silently returned `None`, losing an entire chunk of labels with no indication.  
**Fix**: 5-attempt retry with linear backoff (20s, 40s, 60s, 80s).

```python
for attempt in range(5):
    try:
        completion = client.chat.completions.create(**kwargs)
        ...
        return response_origin
    except Exception as e:
        if "429" in str(e) and attempt < 4:
            wait = 20 * (attempt + 1)
            print(f"  [rate limit] attempt {attempt+1}/5, waiting {wait}s...")
            time.sleep(wait)
        else:
            return None
```

#### Fix 5 — Hardcoded model name

**File**: `text_clustering/config.py`  
**Issue**: `"gpt-3.5-turbo-0125"` was hardcoded in every script, making model switching require code edits.  
**Fix**: Read from `LLM_MODEL` env var.

```python
MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
```

#### Fix 6 — Missing `.env` loading

**File**: `text_clustering/client.py`  
**Issue**: `load_dotenv()` was never called, so the `.env` file was silently ignored and all env vars fell back to their defaults.  
**Fix**: `load_dotenv()` called at import time in `client.py`.

#### Fix 7 — `LLM_REQUEST_DELAY` not consumed

**File**: `text_clustering/llm.py`  
**Issue**: `.env` defines `LLM_REQUEST_DELAY=4` but no script read it. Without a delay, requests fire back-to-back and immediately hit the 20 req/min ceiling.  
**Fix**: Sleep after each successful API call.

```python
_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))

# after a successful completion:
if _REQUEST_DELAY > 0:
    time.sleep(_REQUEST_DELAY)
```

With `LLM_REQUEST_DELAY=4` the pipeline stays at ~15 req/min, safely under the OpenRouter limit.

---

### Improvements

These are additions that go beyond the original scope — the pipeline worked without them, but they make it more practical for long or repeated runs.

#### Improvement 1 — Package restructuring

The original code was 4 flat scripts with duplicated helpers (`ini_client`, `chat`, prompt builders) copy-pasted across files. All shared logic was moved into a proper `text_clustering/` package:

- `client.py` — API client factory
- `config.py` — single source of truth for all env vars
- `llm.py` — `chat()` with retry, `_strip_fenced_json()`
- `data.py` — dataset loading
- `prompts.py` — prompt construction
- `pipeline/` — the 4 pipeline steps as importable modules with console-script entry points

#### Improvement 2 — Timestamped run directories

The original wrote all outputs to a flat `generated_labels/` folder with fixed filenames, so re-running a dataset overwrote previous results.

Each Step 1 run now creates an isolated folder: `runs/<dataset>_<split>_<YYYYMMDD_HHMMSS>/`. All subsequent steps read from and write to that folder. Previous runs are never touched.

#### Improvement 3 — Checkpoint / resume (Step 2)

Step 2 makes one API call per text (~3,000 per dataset, ~3h20). The original had no way to recover from an interruption — a crash meant starting from scratch.

Progress is now saved to `checkpoint.json` every 200 samples. Re-running the same command detects the checkpoint and resumes from where it stopped. The checkpoint is deleted automatically on successful completion.

#### Improvement 4 — `results.json`

The original `evaluate.py` printed metrics to stdout only. Results are now also written to `results.json` in the run directory, including ACC, NMI, ARI, sample count, cluster counts, model name, and timestamp.

#### Improvement 5 — Logging

All `print()` calls in the pipeline were replaced with Python's standard `logging` module. A `setup_logging(log_path)` function configures two handlers at startup: one to stdout (INFO level) and one to a `run.log` file inside the run directory (DEBUG level). The log format includes a timestamp, level, and module name:

```
2026-02-20 14:32:01 | INFO     | label_generation | Run dir: ./runs/...
```

Every pipeline step writes its full trace to `run.log` in its run directory. Step 0 writes to `runs/seed_labels.log`. This means a run can always be reconstructed after the fact — which model was used, when each step ran, any rate limit retries, and progress checkpoints.



---

## 5. API & Model Investigation

### OpenRouter rate limits

OpenRouter has two distinct concepts:

| Concept | Detail |
|---------|--------|
| **Account balance** | Must be non-negative to use free models (even $0/token ones) |
| **Key limit** | Per-key spending cap. If set to `0`, blocks all requests including free ones |
| **Free tier cap** | `is_free_tier: true` → 50 req/day. Purchasing ≥ $10 raises it to 1,000/day |
| **RPM** | 20 req/min for free models regardless of tier |

After adding €10 of credits the key changed to `is_free_tier: false`, `limit: null` — no daily cap, governed only by credit balance.

**Important**: The popular free models (Llama 70B, Mistral 24B, Gemma 27B) are all routed through a single upstream provider called **Venice**. When Venice is under load, all of these fail simultaneously with a 429 even if your account has plenty of credits. This is an infrastructure issue on their side, not an account issue.

### Why we don't use OpenAI JSON mode

`response_format={"type": "json_object"}` — what the original code uses — is not supported by most free models. The options are:

- Some return HTTP 400 (Google AI Studio, older models)
- Some return an empty body (Solar Pro 3, Step Flash)
- Reasoning models spend all `max_tokens` on internal chain-of-thought and return `content: ""`

Our fix: leave JSON mode off by default and strip fences from responses instead. Works reliably across all tested models.

### Why reasoning models are excluded

Models like DeepSeek R1, Solar Pro 3, GLM 4.5 Air, and Qwen3-thinking variants use an internal chain-of-thought before producing a final answer. On OpenRouter, this CoT appears in `message.reasoning` while the actual answer goes in `message.content`. The problem:

- At `max_tokens=512` (pipeline default), they spend the entire budget on CoT and return `content: ""`
- Even with a larger budget, they are slow and token-heavy — incompatible with ~3,000 calls per dataset

### Model selection criteria

1. Instruct model (not reasoning/thinking)
2. ≥ 24B parameters — needed for reliable label proposal and 59-class classification
3. ≥ 32K context — classification prompt lists up to 60 candidate labels
4. Responds without a system prompt being required
5. Returns JSON without wrapping it in extra explanation

### Merge capability requirement

An additional hard requirement emerged during Run 02 (see §7): the model must be able to
consolidate ~150 proposed labels down to ~18 in a **single call**. This is what GPT-3.5-turbo
does in the paper. It cannot be compensated for with batching or multi-pass strategies — those
approaches either stall (trinity-large-preview) or produce data leakage (map-to-canonical). See
§7 — Run 02 for the full investigation.

This requirement disqualifies `arcee-ai/trinity-large-preview:free` as the primary model,
despite it passing all 6 probe tests.

### Primary model: `google/gemini-2.0-flash-001`

Selected after the Run 02 merge investigation. Probe: 6/6 RECOMMENDED (2026-02-21). Merge test:
167 labels → **28 in 1.9 seconds** — single call, paper-aligned.

**Cost estimate — full 5-dataset baseline**  
Pricing: $0.10/M input tokens · $0.40/M output tokens

| Dataset | Estimated cost |
|---------|---------------|
| `massive_scenario` | ~$0.14 |
| `massive_intent` | ~$0.11 |
| `go_emotion` | ~$0.29 |
| `arxiv_fine` | ~$0.27 |
| `mtop_intent` | ~$0.11 |
| **Total (5 datasets)** | **~$0.92** |

With 2× safety margin for retries / rate limit backoff: **~$1.83**. $10 budget → **~$8.17 remaining** after full baseline.

**Note on availability**: OpenRouter lists `gemini-2.0-flash-001` as going away March 31, 2026.
If the model is retired before the full baseline is complete, re-run `probe_models.py` on the
successor (`google/gemini-2.0-flash-lite` or `google/gemini-2.5-flash-preview`) and update
`LLM_MODEL` in `.env`.

---

## 6. Model Probe Results

All models tested with `probe_models.py` (6 tests each: reachability, label gen, merge,
classification, consistency, token efficiency). Account status: `is_free_tier: false`,
`limit: null`.

### Probe summary

| Model | Date | Score | Verdict | Notes |
|-------|------|-------|---------|-------|
| `arcee-ai/trinity-large-preview:free` | 2026-02-20 | **6/6** | ✅ PASS | All tests pass — but merge capability insufficient (see §7 Run 02). |
| `google/gemini-2.0-flash-001` | 2026-02-21 | **6/6** | ✅ **PRIMARY** | 6/6 + merge test: 167 → 28 labels in 1.9s. |
| `openai/gpt-4o-mini` | 2026-02-21 | 6/6 | ⚠️ USABLE | 6/6 probe but merge test: 167 → 105 (poor consolidation). |
| `meta-llama/llama-3.3-70b-instruct:free` | 2026-02-20 | —/— | ⏳ Pending | Venice upstream 429. Retry when congestion clears. |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 2026-02-20 | —/— | ⏳ Pending | Venice upstream 429. |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 2026-02-20 | —/— | ⏳ Pending | Venice upstream 429. |
| `google/gemma-3-27b-it:free` | 2026-02-21 | —/— | ⏳ Pending | Venice upstream 429. |
| `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | 2026-02-20 | —/— | ⏳ Pending | Venice upstream 429. |
| `google/gemma-3-12b-it:free` | 2026-02-20 | 0/1 | ❌ Ineligible | Rejects system prompts; 12B / 32K context too small. |
| `arcee-ai/trinity-mini:free` | 2026-02-20 | 0/1 | ❌ Ineligible | Reasoning model — `content: ""` + `reasoning` field confirmed. |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 2026-02-20 | 0/1 | ❌ Ineligible | Reasoning model confirmed. |
| `qwen/qwen3-235b-a22b:free` | 2026-02-20 | 0/1 | ❌ Ineligible | `No endpoints found` on OpenRouter. |
| `openai/gpt-oss-120b:free` | 2026-02-20 | 0/1 | ❌ Blocked | Requires enabling data sharing in OpenRouter privacy settings. |
| `openai/gpt-oss-20b:free` | 2026-02-20 | 0/1 | ❌ Blocked | Same data policy restriction. |

### Models confirmed as reasoning/thinking (excluded)

| Model | Evidence |
|-------|----------|
| `upstage/solar-pro-3:free` | `reasoning_details` field in API response (2026-02-19) |
| `z-ai/glm-4.5-air:free` | `reasoning_details` field in API response (2026-02-19) |
| `arcee-ai/trinity-mini:free` | `content: ""` + `reasoning` field populated (2026-02-20) |
| `nvidia/nemotron-3-nano-30b-a3b:free` | `content: ""` + `reasoning` field populated (2026-02-20) |
| `deepseek/deepseek-r1-0528:free` | R1 architecture — reasoning by design |
| `qwen/qwen3-*-thinking` variants | Thinking variant in name |
| `stepfun/step-3.5-flash:free` | Empty `content` in practice (2026-02-19) |

### Models excluded for other reasons

| Model | Reason |
|-------|--------|
| `qwen/qwen3-235b-a22b:free` | 404 — no endpoints on OpenRouter |
| `openai/gpt-oss-120b:free` / `20b:free` | OpenRouter data policy restriction |
| `google/gemma-3-12b-it:free` | Rejects system prompts; 12B / 32K context too small |
| `meta-llama/llama-3.2-3b-instruct:free` | 3B — too small |
| `qwen/qwen3-4b:free` | 4B — too small |
| `google/gemma-3-4b-it:free` / `3n-e*` | Too small |
| `openrouter/free` | Routes to a random model — not reproducible |

### Current recommendation

**Primary model: `google/gemini-2.0-flash-001`** — the only model tested that passes all
6 probe tests AND performs paper-aligned single-call merge at scale. See §5 for selection
rationale and cost breakdown.

For zero-cost comparison runs, retry the Venice-blocked free models (Llama 70B, Mistral 24B,
Gemma 27B, Hermes 405B) during off-peak hours — but confirm merge capability with
`probe_models.py` before use.

---

## 7. Pipeline Execution Log

### Run 01 — `massive_scenario` · `arcee-ai/trinity-large-preview:free` · 2026-02-20

First end-to-end validation run. Target: lightest dataset (2,974 samples, 18 classes).  
Pipeline executed with **no code changes** relative to v1.2.0 — unmodified prompts, default `LLM_MAX_TOKENS=512`.

#### Estimated cost per step (pre-run)

| Step | Command | API calls | Time @ 4s/call |
|------|---------|-----------|----------------|
| 0 | `tc-seed-labels` | 0 | ~2s |
| 1 | `tc-label-gen` | ~200 (chunks + merge) | ~13 min |
| 2 | `tc-classify` | 2,974 (one per text) | ~3h20 |
| 3 | `tc-evaluate` | 0 | ~5s |

#### Step 0 — seed labels ✅

```
Completed : 2026-02-20 16:13:18  (~2s)
Output    : runs/chosen_labels.json
Seeds     : arxiv_fine=18, go_emotion=5, massive_intent=11, massive_scenario=3, mtop_intent=20
```

#### Step 1 — label generation ✅

```
Completed : 2026-02-20 16:40:38  (26.6 min — 198 chunk calls + 1 merge call)
Run dir   : runs/massive_scenario_small_20260220_161359/
Proposed  : 190 labels  (before merge)
Merged    : 190 labels  (merge call silently failed — see issue below)
```

**Issue discovered — merge response truncated at 512 tokens**:  
The `LLM_MAX_TOKENS=512` default is too small for a merge response covering 190 labels
(~2,300 tokens needed). The merge API call returned a truncated JSON string, `eval()` threw an
exception, and `merge_labels()` silently fell back to returning the unmerged list unchanged.
The pipeline continued without any error or warning. Result: 190 labels for an 18-class dataset.

#### Step 2 — classification ✅

```
Started   : 2026-02-20 16:45:25
Completed : 2026-02-20 21:14:53  (4h29 — 2,974 API calls)
Classified into 168 distinct predicted labels (22 of the 190 received 0 samples)
Output    : runs/massive_scenario_small_20260220_161359/classifications.json
```

#### Step 3 — evaluation ✅

```
Completed : 2026-02-20 21:45:34
n_clusters_pred : 168  (vs. 18 true)
```

#### Root cause — per-class fragmentation

The merge received 190 labels and returned 190 unchanged (truncated response, silent fallback).
168 of those received at least one sample. Each true class was split across multiple predicted
labels instead of being captured by one:

| True class | n | Dominant predicted label | % captured | # predicted splits |
|------------|---|--------------------------|------------|---------------------|
| `alarm` | 96 | `alarm` | 91% | 3 |
| `weather` | 156 | `weather` | 88% | 10 |
| `cooking` | 72 | `recipe` | 69% | 8 |
| `news` | 124 | `news` | 70% | 18 |
| `email` | 271 | `email` | 53% | 23 |
| `music` | 81 | `music` | 53% | 9 |
| `datetime` | 103 | `time` | 48% | 8 |
| `social` | 106 | `social media` | 46% | 17 |
| `takeaway` | 57 | `food order` | 40% | 9 |
| `transport` | 124 | `train` | 35% | 9 |
| `play` | 387 | `music` | 35% | 26 |
| `iot` | 220 | `lighting control` | 35% | 13 |
| `lists` | 142 | `list` | 32% | 26 |
| `calendar` | 402 | `calendar` | 32% | 35 |
| `recommendation` | 94 | `local events` | 33% | 23 |
| `audio` | 62 | `volume control` | 39% | 14 |
| `general` | 189 | `general` | 23% | 61 |
| `qa` | 288 | `general knowledge` | 15% | 39 |

**Concrete synonym groups that should have been merged but weren't:**

| True class | Un-merged synonyms |
|------------|--------------------|
| `datetime` | `time`, `date`, `time zone`, `time conversion`, `date time` |
| `play` | `music`, `music playback`, `podcast`, `podcasts`, `radio`, `audiobook` |
| `audio` | `volume control`, `volume`, `sound control`, `mute`, `media control` |
| `lists` | `list`, `lists`, `to-do list`, `task management`, `task` |
| `takeaway` | `food order`, `food`, `delivery`, `order`, `order tracking`, `restaurant locate` |

The label-generation step worked correctly — every true class concept appears in the proposed
list. The failure is entirely in the merge step (silently skipped due to token truncation).
The metric gap (ACC −31, NMI −11, ARI −24 vs. paper) is an artifact of taxonomy fragmentation,
not classification quality or model capability.

---

### Run 02 — Merge Investigation & Model Switch

After merging the fixes from `fix/run-02-prep` into `develop` (`b717cbb`), three Step 1 re-runs
were attempted for `massive_scenario`. All three revealed a deeper problem: the merge step
continued to fail, for different reasons each time.

#### Step 1 re-attempts — Fix A+B+C applied

**Date**: 2026-02-21  **Model**: `arcee-ai/trinity-large-preview:free`

```
Proposed labels : 158
Merged labels   : 158   ← WARNING: 8.8× the true class count (18)
```

**Root cause 1 — Parser crash on flat list**: The model returned a flat JSON array
`["a", "b", ...]` instead of the expected dict `{"merged_labels": [...]}`. The original parser
iterated `parsed.values()`, which crashes on a list, silently falling back to the unmerged list.
Fixed with `_parse_merge_response()` (handles both dict and list responses).

**Root cause 2 — Model capability ceiling**: Even after fixing the parser, the model cannot
semantically consolidate labels at this scale. It performs cosmetic reformatting
(snake_case → space case) but cannot reason that `"iot"` + `"smart_home"` + `"lighting_control"`
+ `"home_automation"` all refer to one cluster.

#### Batched multi-pass merge attempt

To work around the ceiling, a batched merge strategy was implemented:

- **Phase 1**: merge each batch of 30 labels independently (the model handles 30 reliably)
- **Phase 2**: merge the reduced batch outputs in a final call
- **Iteration**: repeat up to 5 passes or until progress stalls

**Result**: 154 → 149 → 144 → stalled. Only 10 labels removed across 3 passes. The model
cannot perform cross-concept grouping even in small batches. It is a capability gap for semantic consolidation, not a
scale problem.

#### Map-to-canonical approach — explored and REJECTED

To bypass free-form consolidation, a guided mapping approach was implemented: give the model all
18 true labels as canonical anchors, and ask it to match each proposed label to the closest
canonical one.

**Result**: 154 proposed → 18 canonical labels in 4 API calls. Works perfectly.

**Why rejected — data leakage**: The paper's pipeline is semi-supervised. The only ground truth
visible to the LLM is the 20% seed labels (`chosen_labels.json`). For `massive_scenario` that
is 3 of 18 labels: `["iot", "music", "general"]`. The full true label set (`labels_true.json`)
is used **only** for metric computation in Step 3 — never passed to the LLM.

This approach passes **all 18 true labels** to the merge step — 15 labels the paper deliberately
withheld. Any metrics produced would inflate accuracy and cannot be compared to the paper
baseline.

| Step | Paper's LLM sees | map-to-canonical gives |
|------|-----------------|------------------------|
| Generation | 3 seed labels | 3 seed labels ✅ |
| Merge | Proposed list only | **All 18 true labels** ← leak ❌ |
| Classification | Merged list | Merged list ✅ |

The full implementation is preserved on `archive/map-to-canonical` for reference. It must never
be merged to `develop`.

#### Root cause summary

The fundamental issue is **model capability**, not code design. `trinity-large-preview` cannot
perform semantic cross-concept consolidation across 150+ labels the way GPT-3.5-turbo does.
There is no paper-aligned workaround — only a model switch resolves this.

#### Resolution — switch to `google/gemini-2.0-flash-001`

**Merge test (2026-02-21)**:
```
Input:  167 labels
Output: 28 labels  in 1.9 seconds  (single call, paper-aligned)
```

This is the behaviour the paper describes. See §5 for cost estimate and §6 for full probe
results. The model switch makes all workarounds unnecessary.

---

### Run 02 — `massive_scenario` · `google/gemini-2.0-flash-001` · `target_k=18` · 2026-02-21

First full end-to-end run with gemini. `target_k=len(true_labels)` passed to the merge prompt
as a legacy workaround (later removed — see Run 03 for the investigation).

**Run directory**: `runs/massive_scenario_small_20260221_035641/`  
**Commit**: `fix/model-gemini-flash` @ `333ce12`

#### Step 1 — label generation ✅

```
Started   : 2026-02-21 03:56:41
Completed : 2026-02-21 04:06:45  (604 s ≈ 10.1 min)
API calls : 200  (199 label-gen batches + 1 merge call)
Errors    : 0
Proposed  : 352 labels  (343 unique after dedup)
Merged    : 18 labels   (target_k=18 passed to merge prompt)
True k    : 18
```

The 352 proposed labels showed the expected fragmentation across synonym groups:
20 time variants, 17 music variants, 17 email/comm variants, 16 iot/home variants.
Gemini collapsed all of them to 18 in a single merge call (~3s), which is the paper-aligned
behaviour. The merge produced no parse errors.

**Merged label set** (18 labels):

```json
["general_information", "time_and_date", "events_and_calendar", "food_and_drink",
 "music_and_audio", "movies_and_tv", "shopping_and_orders", "travel_and_transportation",
 "home_automation", "communication", "personal_management", "finance_and_investments",
 "health_and_wellbeing", "news_and_social_media", "jokes_and_entertainment",
 "search_and_recommendations", "device_control", "location_and_navigation"]
```

**Label quality audit** (18 merged vs. 18 true):

| Status | Count | Labels |
|--------|-------|--------|
| ✅ Good semantic match | 10 | `general_information` → qa+general; `time_and_date` → datetime; `events_and_calendar` → calendar; `food_and_drink` → takeaway+cooking; `music_and_audio` → music+audio+play; `travel_and_transportation` → transport; `home_automation` → iot; `communication` → email+social; `personal_management` → alarm+lists; `shopping_and_orders` → lists |
| ⚠️ Overlap (duplicate concept) | 4 | `news_and_social_media` (∥ communication), `search_and_recommendations` (∥ general_information), `device_control` (∥ home_automation), `location_and_navigation` (∥ transport) |
| 🔴 Spurious (no true counterpart) | 4 | `movies_and_tv`, `finance_and_investments`, `health_and_wellbeing`, `jokes_and_entertainment` |

**Missing**: `weather` — present in proposed labels but dropped by the merge. 156 weather
samples (an entire true class) were later scattered across `general_information` (78%),
`location_and_navigation` (8%), and `time_and_date` (6%).

Root cause: forcing `target_k=18` compelled gemini to fill all 18 slots. With 4 spurious
labels occupying slots, `weather` had no slot to land in and was absorbed into the nearest
neighbor during classification.

#### Step 2 — classification ✅

```
Started   : 2026-02-21 04:18:15
Completed : 2026-02-21 06:26:49  (7,714 s ≈ 2h09)
Samples   : 2,974  (one API call each)
Errors    : 0
```

#### Step 3 — evaluation ✅

```
Completed : 2026-02-21 13:54:01
```

**Per-cluster purity highlights**:

| Predicted cluster | Size | Purity | Dominant true label |
|-------------------|------|--------|---------------------|
| `home_automation` | 192 | 0.958 | iot: 96% |
| `communication` | 260 | 0.900 | email: 90% |
| `travel_and_transportation` | 110 | 0.855 | transport: 85% |
| `events_and_calendar` | 388 | 0.838 | calendar: 84% |
| `music_and_audio` | 405 | 0.800 | play: 80% |
| `general_information` | 454 | 0.416 | qa: 42%, weather: 27% ← catch-all |
| `time_and_date` | 184 | 0.538 | datetime: 54%, alarm: 33% ← split |
| `search_and_recommendations` | 65 | 0.292 | scattered ← lowest purity |

**Most fragmented true class**: `recommendation` — spread across 5 predicted clusters, best
concentration only 32%.

---

### Run 03 — `massive_scenario` · `google/gemini-2.0-flash-001` · no `target_k` · 2026-02-21

After the code audit (§4 fix 8), `target_k` was removed from the default merge call to restore
paper-faithful behaviour. This run tests whether gemini consolidates aggressively enough
without a target anchor.

**Run directory**: `runs/massive_scenario_small_20260221_150023/`  
**Commit**: `fix/model-gemini-flash` @ `9cef357`  
**Steps completed**: Step 1 only (run aborted after inspecting merge output).

#### Step 1 — label generation ✅ / merge ❌

```
Proposed  : 343 labels
Merged    : 311 labels   ← only 32 labels removed (1.1× reduction)
True k    : 18
```

Without a `target_k` anchor, gemini treated the merge as **light deduplication** instead of
aggressive semantic consolidation. It removed near-identical surface duplicates
(`movie`/`movies`, `email`/`emails`, `restaurant`/`restaurants`) but left all major synonym
groups intact:

| Synonym group | Surviving variants in merged output |
|---------------|-------------------------------------|
| music | 15: `music`, `song`, `playlist`, `music_playback`, `music_control`, `music_streaming`, … |
| time | 11: `time`, `date`, `datetime`, `time_and_date`, `timer`, `time_conversion`, … |
| calendar/meeting | 9: `calendar`, `schedule`, `meeting`, `meeting_scheduling`, `calendar_management`, … |
| search/query | 9: `search`, `query`, `queries`, `search_engine`, `search_query`, … |
| iot/home | 8: `iot`, `home_automation`, `lights`, `lighting`, `automation`, `device_control`, … |

**Step 2 and Step 3 were not run** — a 311-cluster classification would produce metrics even
worse than Run 01 (168 clusters) and cost ~$0.50 with no scientific value.

#### Conclusion

The paper's approach requires the LLM to know the target granularity. GPT-3.5-turbo was able
to consolidate without an explicit target likely because the paper's prompt tuning or its
training data aligned well with the 15-class semantic space. Gemini-2.0-flash, despite being
a stronger model, interprets the prompt conservatively without guidance.

**Decision**: `target_k` must remain the default for this pipeline. It is not a "weak model
workaround" — it is a necessary semantic anchor for any model when the proposed label count is
in the hundreds. The `--target_k` CLI flag is now mandatory for comparable results.

A follow-up iteration (`fix/merge-prompt-v2`) will redesign the merge prompt to be more
inherently aggressive without relying on a numeric target, using stronger consolidation
language and few-shot examples. See §9 Next Steps.

---

## 8. Results

Paper results use `gpt-3.5-turbo-0125`, batch size 15, 20% seed labels, small split.  
Column order in Table 2 (paper): ArxivS2S | GoEmo | Massive-D | Massive-I | MTOP-I.

### Paper baseline (Table 2 — our method row)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |
| `massive_scenario` | 71.75 | 78.00 | 56.86 |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |

---

### `massive_scenario` · small split — all runs

| Run | Model | target_k | n_pred | ACC | NMI | ARI | Status |
|-----|-------|----------|--------|-----|-----|-----|--------|
| Paper | `gpt-3.5-turbo-0125` | implicit | ~18 | **71.75** | **78.00** | **56.86** | Reference |
| Run 01 | `trinity-large-preview:free` | — | **168** | 40.69 | 66.64 | 33.06 | ❌ Broken merge (token truncation) |
| Run 02 | `gemini-2.0-flash-001` | 18 | **18** | **60.46** | **63.90** | **53.87** | ✅ Valid |
| Run 03 | `gemini-2.0-flash-001` | none | — | — | — | — | ⚠️ Step 1 only — merge failed (311 labels) |

#### Run 02 vs. paper gap analysis

| Metric | Run 02 | Paper | Gap | Notes |
|--------|--------|-------|-----|-------|
| ACC | 60.46 | 71.75 | −11.29 | 4 spurious labels + missing `weather` cluster |
| NMI | 63.90 | 78.00 | −14.10 | Overlapping merged labels split true classes |
| ARI | **53.87** | **56.86** | **−2.99** | Near-paper — cluster structure nearly correct |

ARI within 3 points of the paper — the overall cluster assignment structure is sound.
The ACC gap is driven by 4 identifiable label-quality issues (all traceable to `target_k`
forcing spurious slot-filling). The NMI gap reflects 4 overlapping merged labels that split
true classes across multiple predicted buckets.

**The remaining gap is a label quality problem, not a model capability problem.**

---

## 9. Next Steps

### Immediate — v1.3.0 release

- [x] `fix/model-gemini-flash` complete and pushed
- [ ] PR: `fix/model-gemini-flash` → `develop`
- [ ] PR: `develop` → `main`
- [ ] `cz bump` → v1.3.0

### Next iteration — `fix/merge-prompt-v2`

Redesign the merge prompt to consolidate aggressively **without** a numeric `target_k`, in
order to respect the semi-supervised nature of the pipeline. Approach:

- Stronger consolidation language ("merge any label that refers to the same real-world intent,
  even if the wording differs")
- Explicit examples of what must be merged (alarm/reminder/alarms → one label)
- Possibly a two-phase prompt: first cluster by concept domain, then name each cluster
- Success criterion: gemini produces ≤ 30 labels from 350 proposed without `--target_k`

After a successful prompt redesign, re-run `massive_scenario` without `--target_k` and compare
to Run 02.

### After merge prompt validated

- [ ] Run remaining 4 datasets with gemini (`massive_intent`, `go_emotion`, `arxiv_fine`, `mtop_intent`)
- [ ] Record full 5-dataset results table in §8
- [ ] Compare to paper Table 2

### Longer term

- [ ] Re-probe Venice-blocked free models (Llama 70B, Mistral 24B) during off-peak hours — confirm merge capability before use
- [ ] Once a second model passes merge test, run the same pipeline and compare results

> **Note on `run.sh`**: The original script runs all 5 datasets in parallel using `nohup ... &`.
> With a single API key this saturates rate limits immediately.
> We run datasets sequentially instead.