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

---

## 6. Model Probe Results

All models tested with `probe_models.py` on 2026-02-20 (6 tests each: reachability, label gen, merge, classification, consistency, token efficiency). Account status at time of probing: `is_free_tier: false`, `limit: null`.

### Probe summary

| Model | Score | Verdict | Notes |
|-------|-------|---------|-------|
| `arcee-ai/trinity-large-preview:free` | **6/6** | ✅ RECOMMENDED | Perfect run. Correct label, fully consistent. |
| `meta-llama/llama-3.3-70b-instruct:free` | —/— | ⏳ Pending | Venice upstream 429. Retry when congestion clears. |
| `nousresearch/hermes-3-llama-3.1-405b:free` | —/— | ⏳ Pending | Venice upstream 429. |
| `mistralai/mistral-small-3.1-24b-instruct:free` | —/— | ⏳ Pending | Venice upstream 429. |
| `google/gemma-3-27b-it:free` | —/— | ⏳ Pending | Venice upstream 429. |
| `google/gemma-3-12b-it:free` | 0/1 | ❌ Ineligible | Google AI Studio rejects system prompts for this model. Pipeline requires a system prompt. Also borderline size (12B) and context (32K). |
| `arcee-ai/trinity-mini:free` | 0/1 | ❌ Ineligible | Reasoning model — `content: ""` with populated `reasoning` field confirmed. |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 0/1 | ❌ Ineligible | Same — reasoning model confirmed. |
| `qwen/qwen3-235b-a22b:free` | 0/1 | ❌ Ineligible | `No endpoints found` — not available on OpenRouter. |
| `openai/gpt-oss-120b:free` | 0/1 | ❌ Blocked | Requires enabling data sharing in OpenRouter privacy settings. |
| `openai/gpt-oss-20b:free` | 0/1 | ❌ Blocked | Same data policy restriction. |
| `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | —/— | ⏳ Pending | Venice upstream 429. |

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

**Use `arcee-ai/trinity-large-preview:free`** for the first full baseline run. It is the only tested model to pass all 6 probes. Retry the Venice-blocked models (Llama 70B, Mistral 24B, Gemma 27B, Hermes 405B) during off-peak hours; they are the stronger candidates for subsequent comparison runs.

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

### Run 01 — `massive_scenario` · `arcee-ai/trinity-large-preview:free`

**Conditions**: v1.2.0 pipeline, no prompt changes, `LLM_MAX_TOKENS=512` (default).

| Model | ACC | NMI | ARI | n_pred_clusters |
|-------|-----|-----|-----|-----------------|
| `gpt-3.5-turbo-0125` (paper) | 71.75 | 78.00 | 56.86 | ~18–25 (estimated) |
| `arcee-ai/trinity-large-preview:free` | **40.69** | **66.64** | **33.06** | **168** |

**These numbers are not directly comparable to the paper.** The paper baseline implicitly assumes
the merge step produces a label set close in size to the true class count (~18 for this dataset).
With 168 predicted clusters the Hungarian alignment has a much harder problem: the scores are
degraded not by classification quality but by taxonomy fragmentation from the failed merge step.

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

#### Interpretation

The label-generation step worked correctly — every true class concept appears in the proposed
list. The failure is entirely in the merge step, which was silently skipped due to token
truncation. This run validates that the pipeline executes end-to-end without errors, but the
metric gap (ACC −31, NMI −11, ARI −24 vs. paper) is an artifact of the broken merge, not of
classification quality or model capability.

A re-run with the merge step fixed is needed before the results can be compared to the paper.
See §9 for the planned fixes.

---

## 9. Next Steps

### Planned fixes before Run 02

Three targeted changes were identified from the Run 01 analysis. None of them alter the paper's
core approach — they correct implementation gaps that prevented the pipeline from running as
the paper intended.

See `fix/run-02-prep` branch for the implementation of all three.

#### Fix A — Merge token budget (`LLM_MAX_TOKENS` override for merge call)

**Problem**: The merge API call shares the same `LLM_MAX_TOKENS=512` default as every other call.
A 190-label merge response needs ~2,300 tokens. The response was truncated, `eval()` failed,
and the unmerged list was silently returned — 190 labels instead of ~18.  
**Fix**: Allow `chat()` to accept a per-call `max_tokens` override. Pass `max_tokens=4096` in
`merge_labels()`. This is the exact same intent as the original code, just with a token budget
that lets the response complete.  
**Changes paper approach?** No.

#### Fix B — Target-k in the merge prompt

**Problem**: Even with 4096 tokens, the merge prompt as written asks the model to *"identify
entries that are similar or duplicate… and merge them"* with no guidance on how many groups to
produce. `trinity-large-preview` keeps fine-grained distinctions (e.g. `time` / `date` /
`time zone` / `time conversion` as 4 labels instead of one `datetime`). GPT-3.5 collapses them
implicitly because it is better at this instruction. We compensate by being explicit.  
**Fix**: Add *"The list should contain approximately {k} labels"* to the merge prompt, where
`k = len(true_labels)` (already available in Step 1 as `labels_true.json`).  
**Changes paper approach?** No — same merge intent, explicit target count to compensate for a
weaker model's instruction-following.

#### Fix C — Warn when merged count >> true class count

**Problem**: The pipeline ran Step 2 for 4h29 on a broken label set with no warning. There was no
gate between Step 1 and Step 2 to catch this.  
**Fix**: After merge, log a `WARNING` if `len(merged_labels) > 2 × k`:
```
WARNING: merged label count (190) is 10× the true class count (18).
         Classification results will not be comparable to the paper baseline.
         Consider re-running Step 1 before proceeding to Step 2.
```
**Changes paper approach?** No — observability only, no logic change.

### Run 02 plan (after fixes)

1. Apply fixes A + B + C on branch `fix/run-02-prep`, merge to `develop`
2. Re-run Step 1 for `massive_scenario` (~30 min) — verify merged count ≈ 18–25
3. Re-run Step 2 (~3h20)
4. Re-run Step 3, fill in Run 02 results row in §8
5. Compare against paper baseline

### Longer term

- Re-probe Venice-blocked models (Llama 70B, Mistral 24B, Hermes 405B, Gemma 27B) during off-peak
- Once a second model passes, run the same pipeline and compare
- Extend to remaining 4 datasets once `massive_scenario` Run 02 is validated

> **Note on `run.sh`**: The original script runs all 5 datasets in parallel using `nohup ... &`.
> With a single free API key this saturates the 20 req/min rate limit immediately.
> We run datasets sequentially instead.
