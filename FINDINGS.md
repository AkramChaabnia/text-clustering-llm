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
4. [Code Issues & Fixes](#4-code-issues--fixes)
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

The paper reframes unsupervised text clustering as a **classification problem** driven by an LLM. Instead of relying on embeddings + k-means, the pipeline:

1. **Label generation** — given a small set of seed labels (20% of ground truth), the LLM proposes new label names by reading chunks of input texts. Duplicate/similar labels are merged in a second LLM call.
2. **Classification** — the LLM assigns each text to one of the generated labels, one text at a time.
3. **Evaluation** — standard clustering metrics: ACC (Hungarian alignment), NMI, ARI.

### Pipeline schema

```
Dataset (texts + true labels)
        │
        ▼
select_part_labels.py ──► chosen_labels.json  (seed: 20% of true labels)
        │
        ▼
label_generation.py
  ├─ Chunk texts (15/chunk) + seed labels → LLM → candidate labels
  └─ Merge similar labels → LLM → merged_labels_after_merge.json
        │
        ▼
given_label_classification.py
  └─ Each text + merged_labels → LLM → predicted label
        │
        ▼
evaluate.py ──► ACC / NMI / ARI
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
| `pyproject.toml` | Project metadata and all dependencies |
| `uv.lock` | Pinned lockfile for reproducibility |
| `requirements.txt` | Pinned fallback for pip users |
| `.env.example` | Documents every env variable (no secrets) |
| `openrouter_adapter.py` | Thin wrapper: loads `.env`, builds `openai.OpenAI` client for OpenRouter |
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

Output: `./generated_labels/chosen_labels.json`

---

## 4. Code Issues & Fixes

All changes are minimal — no structural modifications to the pipeline logic.

### Fix 1 — `ini_client()` call signature

**File**: `given_label_classification.py`  
**Issue**: `main()` called `ini_client(args.api_key)` but `ini_client()` accepted no arguments → `TypeError` on every run.

```python
# Before
client = ini_client(args.api_key)

# After
client = ini_client()  # API key comes from .env via openrouter_adapter
```

### Fix 2 — `response_format` not universally supported

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: The original code always sent a hardcoded `response_format={"type":"json_object"}`. Many free models return HTTP 400 or an empty body when this is set.  
**Fix**: Made it conditional via `LLM_FORCE_JSON_MODE` env var (default `false`).

```python
_FORCE_JSON_MODE = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"

if _FORCE_JSON_MODE:
    kwargs["response_format"] = {"type": "json_object"}
```

### Fix 3 — Markdown fence stripping

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Free models often wrap their JSON in markdown code fences (` ```json ... ``` `). The original `eval()` call cannot parse these - returns `None` or raises `SyntaxError`, causing all labels to be silently skipped.  
**Fix**: Added `_strip_fenced_json()` helper applied after every API call.

```python
def _strip_fenced_json(text: str) -> str:
    import re
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
```

### Fix 4 — No retry on rate limits

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Original code had no error handling on the API call level. A single 429 silently returned `None`, causing data loss across entire chunks.  
**Fix**: Added 5-attempt retry loop with exponential backoff (20s, 40s, 60s, 80s).

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

### Fix 5 — Hardcoded model name

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Model name `"gpt-3.5-turbo-0125"` was hardcoded.  
**Fix**: Read from env var `LLM_MODEL` (with original as default for backward compat).

```python
_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
```

### Fix 6 — Missing `.env` loading

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Scripts never called `load_dotenv()` - env vars from `.env` were not loaded.  
**Fix**: Added `load_dotenv()` at module top.

### Fix 7 — `LLM_REQUEST_DELAY` not consumed

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: `.env` defines `LLM_REQUEST_DELAY=4` but neither script read it. Without a delay between calls the pipeline fires requests as fast as the API responds, hitting the 20 req/min ceiling and triggering 429s.  
**Fix**: Read the env var at startup and sleep after each successful API call.

```python
_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))

# inside chat(), after a successful completion:
if _REQUEST_DELAY > 0:
    time.sleep(_REQUEST_DELAY)
```

With `LLM_REQUEST_DELAY=4` this caps throughput at 15 req/min, safely under the 20 req/min OpenRouter limit.

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

First run targets **`massive_scenario`** — the lightest dataset (2,974 samples, 18 classes). Goal is to validate the full pipeline end-to-end and get a first comparison with the paper before committing to the other datasets.

### Estimated cost per step

| Step | Script | API calls | Time @ 4s/call |
|------|--------|-----------|----------------|
| 0 | `select_part_labels.py` | 0 | ~2s |
| 1 | `label_generation.py` | ~200 (chunks + merge) | ~13 min |
| 2 | `given_label_classification.py` | 2,974 (one per text) | ~3h20 |
| 3 | `evaluate.py` | 0 | ~5s |

### Step 0 — `select_part_labels.py` ⏳

Needs to be run before Step 1 to generate the seed labels file.

### Step 1 — `label_generation.py --data massive_scenario` ⏳

```
Model  : arcee-ai/trinity-large-preview:free
Status : not started
Output : ./generated_labels/massive_scenario_small_llm_generated_labels_after_merge.json
```

### Step 2 — `given_label_classification.py --data massive_scenario` ⏳

Waiting on Step 1 output.

### Step 3 — `evaluate.py` ⏳

Waiting on Step 2 output.

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

### Reproduction — `massive_scenario` with `arcee-ai/trinity-large-preview:free`

| Model | ACC | NMI | ARI |
|-------|-----|-----|-----|
| `gpt-3.5-turbo-0125` (paper) | 71.75 | 78.00 | 56.86 |
| `arcee-ai/trinity-large-preview:free` | — | — | — |

*To be filled after the run completes.*

---

## 9. Next Steps

1. Run `select_part_labels.py` to generate the seed labels
2. Run Step 1 (`label_generation.py --data massive_scenario`)
3. Run Step 2 (`given_label_classification.py --data massive_scenario`) — ~3h20, run in background
4. Run Step 3 (`evaluate.py --data massive_scenario`), fill in §8 results table
5. Re-probe Venice-blocked models (Llama 70B, Mistral 24B, Hermes 405B, Gemma 27B) during off-peak hours
6. Once a second model is confirmed, run the same pipeline and compare against the paper baseline
7. Extend to the remaining 4 datasets once `massive_scenario` is validated end-to-end

> **Note on `run.sh`**: The original script runs all 5 datasets in parallel using `nohup ... &`. This is not viable with a single free-tier API key — running 5 datasets simultaneously would immediately saturate the 20 req/min rate limit and cause cascading 429s. We run them sequentially instead.
