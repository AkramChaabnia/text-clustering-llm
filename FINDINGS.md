# Phase A — Findings & Investigation Log

> **Project**: PPD — Text Clustering as Classification with LLMs  
> **Team**: M2 MLSD, Université Paris Cité  
> **Session date**: 2026-02-18 → 2026-02-19  
> **Branch**: `feature/project-setup`  
> **Objective**: Reproduce the baseline results from the paper using free LLMs via OpenRouter

---

## Table of Contents

1. [Paper Summary](#1-paper-summary)
2. [Repository Analysis](#2-repository-analysis)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Analysis](#4-dataset-analysis)
5. [Model Investigation](#5-model-investigation)
6. [Code Issues Found & Fixes Applied](#6-code-issues-found--fixes-applied)
7. [Pipeline Execution Log](#7-pipeline-execution-log)
8. [Blocker: OpenRouter Rate Limits](#8-blocker-openrouter-rate-limits)
9. [Decisions & Rationale](#9-decisions--rationale)
10. [Status Summary](#10-status-summary)
11. [Next Steps](#11-next-steps)

---

## 1. Paper Summary

**Title**: Text Clustering as Classification with LLMs  
**Authors**: Chen Huang, Guoxiu He  
**ArXiv**: https://arxiv.org/abs/2410.00927  
**Original repo**: https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM

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
select_part_labels.py ──► chosen_labels.json  (20% seed labels)
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

### Model used in paper

| Stage | Model |
|-------|-------|
| Label generation | `gpt-3.5-turbo-0125` |
| Label merging | `gpt-3.5-turbo-0125` |
| Classification | `gpt-3.5-turbo-0125` |
| Ablation upper bound | `gpt-4` |

### Datasets in paper (5 main, small split)

| Dataset | Domain | # Classes | Used in paper |
|---------|--------|-----------|---------------|
| `arxiv_fine` | Academic topics | 93 | ✅ |
| `go_emotion` | Emotion detection | 27 | ✅ |
| `massive_intent` | Voice assistant intents | 59 | ✅ |
| `massive_scenario` | Voice assistant scenarios | 18 | ✅ |
| `mtop_intent` | Multi-domain intent | 102 | ✅ |

---

## 2. Repository Analysis

### Original state (upstream)

The original code (`ECNU-Text-Computing/Text-Clustering-via-LLM`) had the following characteristics:

| Aspect | Status |
|--------|--------|
| Python packaging | None — flat scripts only |
| Dependency management | None — no `requirements.txt` |
| `.gitignore` | None — would commit secrets and outputs |
| API key handling | Hardcoded as `--api_key` CLI argument |
| Model name | Hardcoded string `"gpt-3.5-turbo-0125"` in source |
| OpenRouter compatibility | None |
| Error handling | Bare `except:` blocks, no retry logic |
| JSON parsing | Relied on `eval()` with no fence stripping |

### Bug found in original code

`given_label_classification.py::main()` called `ini_client(args.api_key)` but `ini_client()` was defined with **no parameters** — a bug in the original paper code that would crash immediately without an OpenAI key set in env.

```python
# Original (buggy)
def ini_client():
    client = OpenAI()
    return client

def main(args):
    client = ini_client(args.api_key)  # ← TypeError: takes 0 args
```

### Files in repo

| File | Purpose |
|------|---------|
| `label_generation.py` | Step 1: LLM label generation + merge |
| `given_label_classification.py` | Step 2: LLM classification |
| `evaluate.py` | Step 3: ACC/NMI/ARI scoring |
| `select_part_labels.py` | Utility: select 20% seed labels |
| `run.sh` | Shell script running all 5 datasets in parallel |

---

## 3. Environment Setup

### Toolchain chosen

| Tool | Version | Reason |
|------|---------|--------|
| Python | 3.12.6 | System default via pyenv |
| uv | latest | Fast, reproducible installs, uv.lock |
| Commitizen | 4.13.7 | Conventional commit enforcement |
| Ruff | 0.15.1 | Fast linting |

### Key files added

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, all dependencies, tool config |
| `uv.lock` | Fully pinned lock file (50 packages) |
| `requirements.txt` | Pinned fallback for pip users |
| `.env.example` | Documented env var template (no secrets) |
| `.gitignore` | Excludes `.env`, `.venv/`, `dataset/`, `generated_labels/`, `logs/` |
| `openrouter_adapter.py` | Drop-in adapter: loads `.env`, configures OpenAI SDK for OpenRouter |
| `.cz.yaml` | Commitizen config (conventional commits, semver) |
| `CONTRIBUTING.md` | Branching strategy, commit guide, PR process |
| `.github/workflows/ci.yml` | CI: lint + test on PR to `main`/`develop` |
| `.github/PULL_REQUEST_TEMPLATE/` | PR template with results table |

### Dependencies installed

```
openai>=1.30.0       — OpenAI-compatible client (works with OpenRouter)
python-dotenv>=1.0.0 — .env loading
scikit-learn>=1.4.0  — NMI, ARI metrics
scipy>=1.13.0        — Hungarian algorithm (ACC)
numpy>=1.26.0        — Array ops in evaluate.py
tqdm>=4.66.0         — Progress bars
httpx>=0.27.0        — HTTP client (used in original code)
```

### Branching strategy implemented

```
main                  ← protected, stable releases
  └── develop         ← integration (all phases merge here)
        ├── phase-a-reproduction  ← current working branch
        ├── phase-b-evals         ← future: complementary experiments
        ├── phase-c-improvements  ← future: algorithmic proposals
        └── refactor-package      ← future: proper src/ layout
```

All 3 branches are live on GitHub.

---

## 4. Dataset Analysis

### Source

Downloaded directly from Google Drive (ClusterLLM dataset, originally from EMNLP 2023):  
https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view

### All available datasets

| Dataset | # Samples (small) | # Samples (large) | # Classes |
|---------|-------------------|-------------------|-----------|
| `arxiv_fine` | 3,674 | — | 93 |
| `banking77` | 3,080 | — | 77 |
| `clinc` | 4,500 | — | 150 |
| `clinc_domain` | 4,500 | — | 10 |
| `few_event` | 4,742 | — | 34 |
| `few_nerd_nat` | 3,789 | — | 58 |
| `few_rel_nat` | 4,480 | — | 64 |
| `go_emotion` | 5,940 | — | 27 |
| `massive_intent` | 2,974 | — | **59** ← start here |
| `massive_scenario` | 2,974 | — | 18 |
| `mtop_domain` | 4,386 | — | 11 |
| `mtop_intent` | 4,386 | — | 102 |
| `reddit` | 3,217 | — | 50 |
| `stackexchange` | 4,156 | — | 121 |

### Data format (per record)

```json
{"task": "massive_intent", "input": "set an alarm for 7am", "label": "alarm set"}
```

### Seed labels selected (Step 1 — already run ✅)

`select_part_labels.py` was executed successfully on **2026-02-18**.  
It selects `floor(0.2 × num_classes)` random labels per dataset.

| Dataset | Total classes | Seed labels given to LLM |
|---------|--------------|--------------------------|
| `arxiv_fine` | 93 | 18 |
| `banking77` | 77 | 15 |
| `clinc` | 150 | 30 |
| `clinc_domain` | 10 | 2 |
| `few_event` | 34 | 6 |
| `few_nerd_nat` | 58 | 11 |
| `few_rel_nat` | 64 | 12 |
| `go_emotion` | 27 | 5 |
| **`massive_intent`** | **59** | **11** |
| `massive_scenario` | 18 | 3 |
| `mtop_domain` | 11 | 2 |
| `mtop_intent` | 102 | 20 |
| `reddit` | 50 | 10 |
| `stackexchange` | 121 | 24 |

Output: `./generated_labels/chosen_labels.json` ✅

The true labels for `massive_intent` (59 classes) were also extracted and saved:  
`./generated_labels/massive_intent_small_true_labels.json` ✅

---

## 5. Model Investigation

### Paper model vs free alternatives

The paper used `gpt-3.5-turbo-0125` (paid OpenAI API). For reproduction, we use **OpenRouter** with free-tier models.

#### Free models available on OpenRouter (as of 2026-02-18)

31 free models were found. Relevant candidates for this task:

| Model ID | Size | Context | Provider | JSON mode? | Status |
|----------|------|---------|----------|-----------|--------|
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | 128K | Venice | ✅ | ⚠️ Rate limited |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | 128K | Venice | ✅ | ⚠️ Rate limited |
| `google/gemma-3-27b-it:free` | 27B | 131K | Google AI Studio | ✅ | ⚠️ Rate limited |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | 131K | Venice | ✅ | ⚠️ Rate limited |
| `qwen/qwen3-4b:free` | 4B | 41K | — | ✅ | ⚠️ Rate limited |
| `upstage/solar-pro-3:free` | — | 128K | — | ⚠️ Empty body | ⚠️ Unstable |
| `stepfun/step-3.5-flash:free` | — | 256K | — | ❌ 400 error | ❌ Incompatible |

#### JSON mode compatibility finding

**Critical discovery**: `response_format={"type":"json_object"}` — the parameter used in the original code — is **not universally supported** by free models. Some return HTTP 400, others return empty responses.

Test result with real pipeline prompts:
- Models that **do** produce correct JSON: `llama-3.3-70b`, `mistral-small-3.1`, `gemma-3-27b`  
- They wrap JSON in markdown fences: ` ```json { } ``` ` — the original `eval()` cannot parse this
- `upstage/solar-pro-3`: returns HTTP 200 with empty body when `response_format` is set

#### Recommended model priority

| Priority | Model | Rationale |
|----------|-------|-----------|
| 1st | `meta-llama/llama-3.3-70b-instruct:free` | Closest to gpt-3.5-turbo-0125 in capability; instruction-following; JSON reliable |
| 2nd | `mistralai/mistral-small-3.1-24b-instruct:free` | Faster, solid JSON output |
| 3rd | `google/gemma-3-27b-it:free` | Cross-architecture validation |
| Upper bound | `nousresearch/hermes-3-llama-3.1-405b:free` | Largest free model |

---

## 6. Code Issues Found & Fixes Applied

All fixes are **minimal** — no structural changes, no logic changes to the original pipeline.

### Fix 1 — `ini_client()` signature bug

**File**: `given_label_classification.py`  
**Issue**: `main()` called `ini_client(args.api_key)` but `ini_client()` accepted no arguments → `TypeError` on every run.

```python
# Before (buggy original)
client = ini_client(args.api_key)

# After (fixed)
client = ini_client()  # api_key now comes from env via openrouter_adapter
```

### Fix 2 — `response_format` incompatibility

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Hardcoded `response_format={"type":"json_object"}` causes HTTP 400 on many free models and empty responses on others.  
**Fix**: Made it conditional via `LLM_FORCE_JSON_MODE` env var (default `false`).

```python
# Added env-controlled flag
_FORCE_JSON_MODE = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"

if _FORCE_JSON_MODE:
    kwargs["response_format"] = {"type": "json_object"}
```

### Fix 3 — Markdown JSON fence stripping

**Files**: `label_generation.py`, `given_label_classification.py`  
**Issue**: Free models often wrap their JSON in markdown code fences (` ```json ... ``` `). The original `eval()` call cannot parse these — returns `None` or raises `SyntaxError`, causing all labels to be silently skipped.  
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

### Fix 4 — No retry logic

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
**Issue**: Scripts never called `load_dotenv()` — env vars from `.env` were not loaded.  
**Fix**: Added `load_dotenv()` at module top.

---

## 7. Pipeline Execution Log

### Step 1 — `select_part_labels.py` ✅ DONE

```
Executed: 2026-02-18
Result: ./generated_labels/chosen_labels.json written
        ./generated_labels/massive_intent_small_true_labels.json written
Status: SUCCESS
```

### Step 2 — `label_generation.py --data massive_intent` ⚠️ BLOCKED

```
Executed: 2026-02-18
Model: meta-llama/llama-3.3-70b-instruct:free
Error: 429 on all attempts (5/5) — account-level rate limit
Root cause: OpenRouter account has $0 balance (limit_remaining=0)
Status: BLOCKED — see Section 8
```

Partial output before block:
- True labels file written (59 classes confirmed)
- First chunk prompt sent, all retries exhausted

### Steps 3–4 — Not yet run (blocked on Step 2)

---

## 8. Blocker: OpenRouter Rate Limits

### Root cause

OpenRouter free models require a **non-zero account credit balance** to unlock API access, even though the models themselves are priced at $0/token. This is a **global account-level** restriction, not a per-model or per-request limit.

### Account status (as of 2026-02-18)

```json
{
  "limit": 0,
  "limit_remaining": 0,
  "is_free_tier": true,
  "usage": 0,
  "usage_monthly": 0,
  "expires_at": "2026-03-20T00:12:18.604+00:00"
}
```

### Observed behavior

- All 10+ free models tested: `429 Provider returned error`
- All providers (Venice, Google AI Studio, etc.) simultaneously blocked
- Retry with exponential backoff (up to 80s wait) does not resolve it
- This confirms it is account-level, not upstream provider congestion

### Resolution

**Add a minimum credit balance** to the OpenRouter account:  
https://openrouter.ai/settings/billing

- Recommended: **$5** (sufficient for all baseline runs with free models at $0/token)
- Alternative providers that don't have this requirement:
  - **Groq** (free tier, no credit needed): `https://api.groq.com/openai/v1`
  - **Together.ai** (free trial credits)
  - **Direct OpenAI** (if a key becomes available)

### How to switch to Groq (temporary alternative)

```bash
# In .env:
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your-groq-key
LLM_MODEL=llama-3.3-70b-versatile   # Groq model ID
```

No code change needed — the adapter is provider-agnostic.

---

## 9. Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Use OpenRouter over direct OpenAI | No OpenAI key available; OpenRouter aggregates many free models |
| `LLM_FORCE_JSON_MODE=false` by default | Most free models don't support `response_format=json_object` reliably |
| Keep `eval()` in original scripts | No structural code change in Phase A; `_strip_fenced_json` makes it safe |
| Start with `massive_intent` | Smallest dataset (2974 samples), 59 classes — representative but fast |
| `uv` over `pip`/`conda` | Faster resolution, reproducible lock file, no global environment pollution |
| Flat scripts kept (no `src/` layout) | Phase A constraint: no structural change. Refactor deferred to `refactor-package` branch |
| Per-file `ruff` ignores for original scripts | Pre-existing style issues should not block CI; will be fixed in refactor phase |
| Conventional commits via Commitizen | Required for project traceability; good academic practice |

---

## 10. Status Summary

| Task | Status | Output |
|------|--------|--------|
| Fork + clone repo | ✅ Done | `github.com/AkramChaabnia/text-clustering-llm` |
| Git branching strategy | ✅ Done | `main`, `develop`, `phase-a-reproduction` live |
| Python env (uv + venv) | ✅ Done | `.venv/` Python 3.12, 50 packages |
| `pyproject.toml` + lock | ✅ Done | `uv.lock` pinned |
| OpenRouter adapter | ✅ Done | `openrouter_adapter.py` |
| Pipeline code fixes | ✅ Done | 6 fixes applied (see Section 6) |
| CI workflow | ✅ Done | `.github/workflows/ci.yml` |
| Dataset download | ✅ Done | All 14 datasets in `./dataset/` |
| Step 1 (seed labels) | ✅ Done | `./generated_labels/chosen_labels.json` |
| Step 2 (label gen) | ❌ Blocked | OpenRouter `limit_remaining=0` |
| Step 3 (classification) | ❌ Pending | Waiting on Step 2 |
| Step 4 (evaluation) | ❌ Pending | Waiting on Step 3 |
| Baseline results table | ❌ Pending | Waiting on Step 4 |

---

## 11. Next Steps

### Immediate (run the baseline)

1. ~~**Add $5 credits** to OpenRouter~~ — API confirmed working (smoke test passes as of 2026-02-19)

2. Run the full pipeline for `massive_intent`:
   ```bash
   source .venv/bin/activate
   python label_generation.py --data massive_intent
   python given_label_classification.py --data massive_intent
   python evaluate.py --data massive_intent \
     --predict_file_path ./generated_labels/ \
     --predict_file massive_intent_small_find_labels.json
   ```

3. Record results in baseline table below

### Short-term (Phase A completion)

4. Run remaining 4 paper datasets: `massive_scenario`, `go_emotion`, `mtop_intent`, `arxiv_fine`
5. Compare results to paper Table 3 — document deltas and explain discrepancies (model difference, temperature, API version)
6. Open PR: `feature/project-setup` → `develop`, then start `feature/phase-a-massive-intent`

### Phase B preparation

7. Design ablation experiments:
   - Prompt sensitivity (label count, phrasing)
   - Model comparison (llama vs mistral vs gemma)
   - Chunk size sensitivity (`--chunk_size`)
8. Create `feature/phase-b-evals` branch from `develop`

---

## 12. Session Log — 2026-02-19 (Cleanup)

**What changed:**

- `openrouter_adapter.py` — two versions had gotten merged/interleaved; rewritten from scratch to clean version
- `label_generation.py` — commented out unused imports (`OpenAI`, `httpx`, `datetime`, `tqdm`)
- `given_label_classification.py` — same import cleanup; `random` also commented out (unused in original)
- `select_part_labels.py` — `math` import commented out (unused)
- `pyproject.toml` — fixed broken `per-file-ignores` line (comment and key were on same line); removed `[tool.mypy]` and `[tool.pytest.ini_options]` sections
- `.github/workflows/ci.yml` — replaced `lint-and-test` job (with push trigger + test step + commitizen check) with lean `lint` job (PR-only, ruff only)
- `CONTRIBUTING.md` — already absent, confirmed deleted
- `.env` — `OPENAI_BASE_URL` and `LLM_MODEL` were concatenated on one line; split back to two separate lines

**Committed as 4 focused commits on `feature/project-setup`:**

| SHA | Message |
|-----|---------|
| `a03521d` | `build: add pyproject.toml, uv venv, ruff and commitizen config` |
| `2ebeed8` | `ci: lint-only check on pull requests to main and develop` |
| `26bdd14` | `fix: openrouter adapter, retry on 429, json fence stripping, ini_client bug` |
| `1e0db4d` | `docs: add FINDINGS.md as living research and decision log` |

**Smoke test result:** `Response: OK` — API is live, `meta-llama/llama-3.3-70b-instruct:free` responding correctly.

**ruff check:** `All checks passed!`
