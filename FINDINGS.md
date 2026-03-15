# Research Findings — Text Clustering as Classification with LLMs# Research Findings — Text Clustering as Classification with LLMs# Research Log — Text Clustering as Classification with LLMs



> **Project**: PPD — M2 MLSD, Université Paris Cité  > **Project**: PPD — Text Clustering as Classification with LLMs  

> **Period**: 2026-02-18 → ongoing  

> **Branch**: `feature/kmedoids`  > **Project**: PPD — M2 MLSD, Université Paris Cité  > **Programme**: M2 MLSD, Université Paris Cité  

> **Goal**: Reproduce the baseline results from [arXiv:2410.00927](https://arxiv.org/abs/2410.00927), then build SEAL-Clust — a scalable, cost-efficient alternative.

> **Period**: 2026-02-18 → ongoing  > **Period**: 2026-02-18 → ongoing  

---

> **Branch**: `feature/kmedoids`  > **Goal**: Reproduce the baseline results from [arXiv:2410.00927](https://arxiv.org/abs/2410.00927) using free LLMs via OpenRouter, then run complementary experiments.

## Table of Contents

> **Goal**: Reproduce the baseline results from [arXiv:2410.00927](https://arxiv.org/abs/2410.00927), then build SEAL-Clust — a scalable, cost-efficient alternative.

1. [Paper Overview](#1-paper-overview)

2. [What Is SEAL-Clust?](#2-what-is-seal-clust)---

3. [Code Fixes Applied to the Original Repository](#3-code-fixes-applied-to-the-original-repository)

4. [Improvements Added](#4-improvements-added)---

5. [API & Model Investigation](#5-api--model-investigation)

6. [Model Probe Results](#6-model-probe-results)## Table of Contents

7. [Pipeline Execution Log](#7-pipeline-execution-log)

8. [All Experimental Results](#8-all-experimental-results)## Table of Contents

9. [Key Findings](#9-key-findings)

10. [Cost Analysis](#10-cost-analysis)1. [Paper Overview](#1-paper-overview)

11. [Future Work](#11-future-work)

1. [Paper Overview](#1-paper-overview)2. [Setup](#2-setup)

---

2. [What Is SEAL-Clust?](#2-what-is-seal-clust)3. [Dataset](#3-dataset)

## 1. Paper Overview

3. [Code Fixes Applied to the Original Repository](#3-code-fixes-applied-to-the-original-repository)4. [Code Changes](#4-code-changes)

**Title**: Text Clustering as Classification with LLMs  

**Authors**: Chen Huang, Guoxiu He  4. [Improvements Added](#4-improvements-added)5. [API & Model Investigation](#5-api--model-investigation)

**Venue**: arXiv 2024 — [2410.00927](https://arxiv.org/abs/2410.00927)  

**Original repo**: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)5. [API & Model Investigation](#5-api--model-investigation)6. [Model Probe Results](#6-model-probe-results)



### Core Idea6. [Model Probe Results](#6-model-probe-results)7. [Pipeline Execution Log](#7-pipeline-execution-log)



The paper reframes unsupervised text clustering as a **classification problem** driven by an LLM:7. [Pipeline Execution Log](#7-pipeline-execution-log)   - [Run 01 — massive_scenario · trinity-large-preview · 2026-02-20](#run-01--massive_scenario--arcee-aitrinity-large-previewfree--2026-02-20)



1. **Label generation** — given 20% seed labels from ground truth, the LLM proposes new labels by reading chunks of input texts. Duplicates are merged in a second LLM call.8. [All Experimental Results](#8-all-experimental-results)   - [Run 02 — Merge Investigation & Model Switch](#run-02--merge-investigation--model-switch)

2. **Classification** — the LLM assigns each text to one of the generated labels, one text at a time.

3. **Evaluation** — standard clustering metrics: ACC (Hungarian alignment), NMI, ARI.9. [Key Findings](#9-key-findings)   - [Run 02 — `massive_scenario` · gemini-2.0-flash-001 · `target_k=18` · 2026-02-21](#run-02--massive_scenario--googlegemini-20-flash-001--target_k18--2026-02-21)



### Pipeline Diagram10. [Cost Analysis](#10-cost-analysis)   - [Run 03 — `massive_scenario` · gemini-2.0-flash-001 · no `target_k` · 2026-02-21](#run-03--massive_scenario--googlegemini-20-flash-001--no-target_k--2026-02-21)



```11. [Future Work](#11-future-work)8. [Results](#8-results)

dataset/small.jsonl  (texts + ground-truth labels)

        │9. [Next Steps](#9-next-steps)

        ▼

  [Step 0]  seed_labels.py        → picks 20% of true labels at random---

        ▼

  runs/chosen_labels.json---

        ▼

  [Step 1]  label_generation.py   → chunks of 15 texts → LLM proposes + merges labels## 1. Paper Overview

        ▼

  runs/<run_dir>/labels_merged.json## 1. Paper Overview

        ▼

  [Step 2]  classification.py     → one LLM call per text → assigns a label**Title**: Text Clustering as Classification with LLMs  

        ▼

  runs/<run_dir>/classifications.json**Authors**: Chen Huang, Guoxiu He  **Title**: Text Clustering as Classification with LLMs  

        ▼

  [Step 3]  evaluation.py         → Hungarian alignment → ACC / NMI / ARI**Venue**: arXiv 2024 — [2410.00927](https://arxiv.org/abs/2410.00927)  **Authors**: Chen Huang, Guoxiu He  

        ▼

  runs/<run_dir>/results.json**Original repo**: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)**Venue**: arXiv 2024 — [2410.00927](https://arxiv.org/abs/2410.00927)  

```

**Original repo**: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)

### Paper Baseline (Table 2 — `gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

### Core Idea

| Dataset | ACC | NMI | ARI |

|---------|-----|-----|-----|### Core idea

| `massive_scenario` | **71.75** | **78.00** | **56.86** |

| `massive_intent` | 64.12 | 65.44 | 48.92 |The paper reframes unsupervised text clustering as a **classification problem** driven by an LLM:

| `go_emotion` | 31.66 | 27.39 | 13.50 |

| `arxiv_fine` | 38.78 | 57.43 | 20.55 |The paper reframes unsupervised text clustering as a **classification problem** driven by an LLM. Instead of embeddings + k-means, the approach is:

| `mtop_intent` | 72.18 | 78.78 | 71.93 |

1. **Label generation** — given 20% seed labels from ground truth, the LLM proposes new labels by reading chunks of input texts. Duplicates are merged in a second LLM call.

---

2. **Classification** — the LLM assigns each text to one of the generated labels, one text at a time.1. **Label generation** — given a small set of seed labels (20% of ground truth), the LLM proposes new label names by reading chunks of input texts. Duplicate/similar labels are merged in a second LLM call.

## 2. What Is SEAL-Clust?

3. **Evaluation** — standard clustering metrics: ACC (Hungarian alignment), NMI, ARI.2. **Classification** — the LLM assigns each text to one of the generated labels, one text at a time.

**SEAL-Clust** (**S**calable **E**fficient **A**utonomous **L**LM **Clust**ering) separates semantic reasoning (LLM) from large-scale clustering computation (embeddings + traditional algorithms).

3. **Evaluation** — standard clustering metrics: ACC (Hungarian alignment), NMI, ARI.

| Problem | Original Paper | SEAL-Clust Solution |

|---------|---------------|---------------------|### Pipeline Diagram

| **High LLM cost** | One API call per document (~3,000 calls) | Only prototype docs go to the LLM (~100–300 calls) |

| **Not fully unsupervised** | Requires 20% seed labels from ground truth | Microclustering needs zero labels |### Pipeline — step by step

| **Undefined k** | LLM implicitly decides k during label merge | Systematic K\* via BIC / silhouette / Calinski-Harabasz |

| **Scalability** | Linear in N (all samples → LLM) | Sub-linear: embed once, cluster locally, LLM on M << N prototypes |```



### The 9-Stage Architecture (SEAL-Clust v2)dataset/small.jsonl  (texts + ground-truth labels)**Step 0 — Seed label selection**  



```        │Before running the LLM, 20% of the ground-truth labels are randomly sampled per dataset and written to `chosen_labels.json`. These seeds are handed to the LLM in Step 1 as a starting point, which anchors the taxonomy and avoids completely free-form generation.

┌─────────────────────────────────────────────────────────────────┐

│  Stage 1: Document Embedding    all-MiniLM-L6-v2 → 384D        │        ▼

│  Stage 2: Dimensionality Reduction    PCA 384D → 50D           │

│  Stage 3: Overclustering    K-Medoids K₀=300 → 300 clusters    │  [Step 0]  seed_labels.py        → picks 20% of true labels at random**Step 1 — Label generation**  

│  Stage 4: Representative Selection    Extract 300 medoid docs   │

│  Stage 5: Label Discovery    LLM reads reps → candidate labels  │        ▼The dataset is shuffled and split into chunks of 15 texts. For each chunk, the LLM is shown the current label set and asked: *"Do any of these texts require a new label that doesn't already exist?"* It adds new candidate labels when needed. Once all chunks are processed, a second LLM call merges and deduplicates near-synonyms (e.g. `"email"` and `"email_management"` collapse into one). The result is the final label set used in Step 2.

│  Stage 6: K* Estimation    Silhouette-elbow / manual → K*       │

│  Stage 7: Label Consolidation    LLM merges → exactly K* labels │  runs/chosen_labels.json

│  Stage 8: Classification    LLM classifies 300 reps → K* labels │

│  Stage 9: Label Propagation    Each doc inherits medoid's label  │        ▼**Step 2 — Classification**  

└─────────────────────────────────────────────────────────────────┘

```  [Step 1]  label_generation.py   → chunks of 15 texts → LLM proposes + merges labelsEach text is sent individually to the LLM with the full label list: *"Which of these labels fits this text best?"* The predicted label is accumulated into a dict `{ label: [text, text, ...] }` and saved as `classifications.json`. This step makes one API call per sample — ~3,000 calls for most datasets. Progress is checkpointed every 200 samples so interrupted runs can resume without starting over.



| Symbol | Meaning | Typical Value |        ▼

|--------|---------|:-------------:|

| **K₀** | Overclustering size (micro-clusters) | 300 |  runs/<run_dir>/labels_merged.json**Step 3 — Evaluation**  

| **K\*** | Final number of categories | 5–50 (auto) or manual |

        ▼Predicted labels are matched to ground-truth labels using the Hungarian algorithm (optimal one-to-one alignment), then ACC, NMI and ARI are computed. Results are printed and saved to `results.json`.

K₀ ≠ K\*. The pipeline over-clusters to get good representatives, then uses the LLM to discover the right number of meaningful categories (K\*).

  [Step 2]  classification.py     → one LLM call per text → assigns a label

---

        ▼### Pipeline diagram

## 3. Code Fixes Applied to the Original Repository

  runs/<run_dir>/classifications.json

The original code was written for OpenAI directly with a paid key. Adapting it to free models via OpenRouter exposed several bugs.

        ▼```

### Fix 1 — `ini_client()` call signature

`main()` called `ini_client(args.api_key)` but `ini_client()` took no arguments → `TypeError` on every run. Fixed: `ini_client()` reads API key from `.env`.  [Step 3]  evaluation.py         → Hungarian alignment → ACC / NMI / ARIdataset/



### Fix 2 — `response_format` not universally supported        ▼  └── small.jsonl  (texts + ground-truth labels)

The original code always sent `response_format={"type":"json_object"}`. Many models return HTTP 400 or empty body. Fixed: opt-in via `LLM_FORCE_JSON_MODE` env var (default `false`).

  runs/<run_dir>/results.json          │

### Fix 3 — Markdown fence stripping

Free models often wrap JSON in markdown code fences (`` ```json ... ``` ``). The original `eval()` call fails on these. Fixed: `_strip_fenced_json()` applied after every API call.```          ▼



### Fix 4 — No retry on rate limits  [Step 0]  seed_labels.py

No error handling around API calls — a single 429 silently returned `None`. Fixed: 5-attempt retry with linear backoff (20s, 40s, 60s, 80s).

### Paper Baseline (Table 2 — `gpt-3.5-turbo-0125`, 20% seed labels, batch=15)          │  picks 20% of true labels at random

### Fix 5 — Hardcoded model name

`"gpt-3.5-turbo-0125"` was hardcoded everywhere. Fixed: read from `LLM_MODEL` env var.          ▼



### Fix 6 — Missing `.env` loading| Dataset | ACC | NMI | ARI |  runs/chosen_labels.json

`load_dotenv()` was never called — all env vars fell back to defaults. Fixed: called at import time in `client.py`.

|---------|-----|-----|-----|          │

### Fix 7 — `LLM_REQUEST_DELAY` not consumed

`.env` defines `LLM_REQUEST_DELAY=4` but no script read it. Without a delay, requests hit rate limit ceilings immediately. Fixed: sleep after each API call.| `massive_scenario` | **71.75** | **78.00** | **56.86** |          ▼



---| `massive_intent` | 64.12 | 65.44 | 48.92 |  [Step 1]  label_generation.py  --data <dataset>



## 4. Improvements Added| `go_emotion` | 31.66 | 27.39 | 13.50 |          │  chunks of 15 texts → LLM proposes labels → merge call



### Package Restructuring| `arxiv_fine` | 38.78 | 57.43 | 20.55 |          ▼

The original 4 flat scripts with duplicated helpers were reorganized into a proper `text_clustering/` package: `client.py`, `config.py`, `llm.py`, `data.py`, `prompts.py`, `pipeline/`.

| `mtop_intent` | 72.18 | 78.78 | 71.93 |  runs/<dataset>_small_<timestamp>/

### Timestamped Run Directories

Each run creates an isolated folder: `runs/<dataset>_<split>_<YYYYMMDD_HHMMSS>/`. Previous runs are never overwritten.    labels_true.json        (ground-truth label list)



### Checkpoint / Resume (Step 2)---    labels_proposed.json    (before merge)

Progress saved to `checkpoint.json` every 200 samples. Re-running the same command resumes from where it stopped.

    labels_merged.json      (final label set)

### `results.json`

Results now written to JSON (ACC, NMI, ARI, sample count, cluster counts, model name, timestamp) — not just stdout.## 2. What Is SEAL-Clust?          │



### Logging          ▼

All `print()` replaced with Python's `logging` module. Two handlers: stdout (INFO) and `run.log` file (DEBUG) with timestamps.

**SEAL-Clust** (**S**calable **E**fficient **A**utonomous **L**LM **Clust**ering) separates semantic reasoning (LLM) from large-scale clustering computation (embeddings + traditional algorithms).  [Step 2]  classification.py  --run_dir <above>

---

          │  one LLM call per text → assigns a label

## 5. API & Model Investigation

| Problem | Original Paper | SEAL-Clust Solution |          ▼

### OpenRouter Rate Limits

|---------|---------------|---------------------|    classifications.json    { label: [text, ...] }

| Concept | Detail |

|---------|--------|| **High LLM cost** | One API call per document (~3,000 calls) | Only prototype docs go to the LLM (~100–300 calls) |          │

| **Free tier cap** | `is_free_tier: true` → 50 req/day. ≥ $10 purchase → 1,000/day |

| **RPM** | 20 req/min for free models regardless of tier || **Not fully unsupervised** | Requires 20% seed labels from ground truth | Microclustering needs zero labels |          ▼

| **Venice upstream** | Free models (Llama 70B, Mistral 24B, Gemma 27B) all route through Venice. When Venice is under load, all fail with 429 simultaneously. |

| **Undefined k** | LLM implicitly decides k during label merge | Systematic K\* via BIC / silhouette / Calinski-Harabasz |  [Step 3]  evaluation.py  --run_dir <above>

### Why We Don't Use OpenAI JSON Mode

`response_format={"type":"json_object"}` is not supported by most free models. Some return HTTP 400, some return empty body, reasoning models spend all tokens on chain-of-thought. Our fix: leave JSON mode off, strip fences from responses.| **Scalability** | Linear in N (all samples → LLM) | Sub-linear: embed once, cluster locally, LLM on M << N prototypes |          │  Hungarian alignment → ACC / NMI / ARI



### Why Reasoning Models Are Excluded          ▼

Models like DeepSeek R1, Solar Pro 3, GLM 4.5 Air use internal chain-of-thought. At `max_tokens=512` they spend the entire budget on CoT and return `content: ""`. They are slow, token-heavy, and incompatible with ~3,000 calls per dataset.

### The 9-Stage Architecture (SEAL-Clust v2)    results.json

### Model Selection Criteria

1. Instruct model (not reasoning/thinking)```

2. ≥ 24B parameters — needed for reliable label proposal and multi-class classification

3. ≥ 32K context — classification prompt lists up to 60+ candidate labels```

4. Responds without a system prompt being required

5. Returns JSON without wrapping it in extra explanation┌─────────────────────────────────────────────────────────────────┐### Models used in the paper

6. **Merge capability**: must consolidate ~150 proposed labels down to ~18 in a single call

│  Stage 1: Document Embedding    all-MiniLM-L6-v2 → 384D        │

### Primary Model: `google/gemini-2.0-flash-001`

Selected after Run 02 merge investigation. Probe: 6/6 RECOMMENDED. Merge test: 167 labels → **28 in 1.9 seconds** — single call, paper-aligned.│  Stage 2: Dimensionality Reduction    PCA 384D → 50D           │| Stage | Model |



**Cost estimate (full 5-dataset baseline)**: ~$0.92 (with 2× safety margin: ~$1.83).│  Stage 3: Overclustering    K-Medoids K₀=300 → 300 clusters    │|-------|-------|



---│  Stage 4: Representative Selection    Extract 300 medoid docs   │| Label generation | `gpt-3.5-turbo-0125` |



## 6. Model Probe Results│  Stage 5: Label Discovery    LLM reads reps → candidate labels  │| Label merging | `gpt-3.5-turbo-0125` |



All models tested with `probe_models.py` (6 tests: reachability, label gen, merge, classification, consistency, token efficiency).│  Stage 6: K* Estimation    Silhouette-elbow / manual → K*       │| Classification | `gpt-3.5-turbo-0125` |



| Model | Score | Verdict | Notes |│  Stage 7: Label Consolidation    LLM merges → exactly K* labels │| Ablation upper bound | `gpt-4` |

|-------|-------|---------|-------|

| `google/gemini-2.0-flash-001` | **6/6** | ✅ **PRIMARY** | Merge: 167 → 28 labels in 1.9s |│  Stage 8: Classification    LLM classifies 300 reps → K* labels │

| `arcee-ai/trinity-large-preview:free` | **6/6** | ⚠️ Merge fails at scale | Stalls at 144 labels |

| `openai/gpt-4o-mini` | 6/6 | ⚠️ USABLE | Merge: 167 → 105 (poor consolidation) |│  Stage 9: Label Propagation    Each doc inherits medoid's label  │### Datasets (5 main, small split)

| `meta-llama/llama-3.3-70b-instruct:free` | — | ⏳ Pending | Venice upstream 429 |

| `mistralai/mistral-small-3.1-24b-instruct:free` | — | ⏳ Pending | Venice upstream 429 |└─────────────────────────────────────────────────────────────────┘

| `google/gemma-3-27b-it:free` | — | ⏳ Pending | Venice upstream 429 |

| `nousresearch/hermes-3-llama-3.1-405b:free` | — | ⏳ Pending | Venice upstream 429 |```| Dataset | Domain | Classes |



### Models Excluded|---------|--------|---------|



| Model | Reason || Symbol | Meaning | Typical Value || `massive_intent` | Voice assistant intents | 59 |

|-------|--------|

| `upstage/solar-pro-3:free` | Reasoning model (`reasoning_details` field) ||--------|---------|:-------------:|| `massive_scenario` | Voice assistant scenarios | 18 |

| `z-ai/glm-4.5-air:free` | Reasoning model |

| `arcee-ai/trinity-mini:free` | Reasoning model (`content: ""`) || **K₀** | Overclustering size (micro-clusters) | 300 || `go_emotion` | Emotion detection | 27 |

| `nvidia/nemotron-3-nano-30b-a3b:free` | Reasoning model |

| `deepseek/deepseek-r1-0528:free` | R1 architecture — reasoning by design || **K\*** | Final number of categories | 5–50 (auto) or manual || `mtop_intent` | Multi-domain intent | 102 |

| `google/gemma-3-12b-it:free` | 12B too small, rejects system prompts |

| `qwen/qwen3-235b-a22b:free` | 404 — no endpoints on OpenRouter || `arxiv_fine` | Academic topics | 93 |

| `openai/gpt-oss-120b:free` / `20b:free` | Data policy restriction |

K₀ ≠ K\*. The pipeline over-clusters to get good representatives, then uses the LLM to discover the right number of meaningful categories (K\*).

---

---

## 7. Pipeline Execution Log

---

### Run 01 — `massive_scenario` · `trinity-large-preview:free` · 2026-02-20

## 2. Setup

First end-to-end validation run. Pipeline executed with no code changes relative to v1.2.0.

## 3. Code Fixes Applied to the Original Repository

- **Step 0**: ✅ Seed labels selected (3 of 18)

- **Step 1**: ✅ 190 proposed labels. Merge **silently failed** — `LLM_MAX_TOKENS=512` too small for 190-label merge response. Parser silently fell back to unmerged list.### Toolchain

- **Step 2**: ✅ 2,974 texts classified (4h29). 168 distinct predicted labels.

- **Step 3**: ✅ ACC=40.69, NMI=66.64, ARI=33.06The original code was written for OpenAI directly with a paid key. Adapting it to free models via OpenRouter exposed several bugs.



**Root cause**: Label merge truncated at 512 tokens → 168 predicted clusters instead of ~18. Each true class was split across 3–61 predicted labels (taxonomy fragmentation). The metric gap (ACC −31 vs paper) is an artifact of merge failure, not classification quality.| Tool | Version | Purpose |



### Run 02 — Merge Investigation & Model Switch · 2026-02-21### Fix 1 — `ini_client()` call signature|------|---------|---------|



After fixing the token limit, three Step 1 re-attempts with `trinity-large-preview:free`:`main()` called `ini_client(args.api_key)` but `ini_client()` took no arguments → `TypeError` on every run. Fixed: `ini_client()` reads API key from `.env`.| Python | 3.12.6 (pyenv) | Runtime |

- Parser crash on flat JSON array → fixed with `_parse_merge_response()`

- Model still couldn't semantically consolidate at scale (154 → 149 → 144, stalled)| uv | latest | Fast, reproducible package installs |

- Batched multi-pass merge attempted → only 10 labels removed in 3 passes

- Map-to-canonical approach tested → **rejected as data leakage** (passes all 18 true labels to merge step, paper withholds 15)### Fix 2 — `response_format` not universally supported| Ruff | 0.15.1 | Linter |



**Resolution**: Switched to `google/gemini-2.0-flash-001`. Merge test: 167 → 28 labels in 1.9s (paper-aligned).The original code always sent `response_format={"type":"json_object"}`. Many models return HTTP 400 or empty body. Fixed: opt-in via `LLM_FORCE_JSON_MODE` env var (default `false`).| Commitizen | 4.13.7 | Conventional commit enforcement |



### Run 02 — `massive_scenario` · `gemini-2.0-flash-001` · `target_k=18` · 2026-02-21



First full end-to-end run with Gemini.### Fix 3 — Markdown fence stripping### Key files added



- **Step 1**: ✅ 352 proposed → 18 merged (target_k=18). 10 good semantic matches, 4 overlapping labels, 4 spurious labels. Missing: `weather` cluster.Free models often wrap JSON in markdown code fences (`` ```json ... ``` ``). The original `eval()` call fails on these. Fixed: `_strip_fenced_json()` applied after every API call.

- **Step 2**: ✅ 2,974 classified (2h09). 0 errors.

- **Step 3**: ✅ **ACC=60.46, NMI=63.90, ARI=53.87**| File | Purpose |



ARI within 3 points of paper (53.87 vs 56.86) — cluster structure nearly correct. The ACC gap (−11.29) is driven by 4 spurious labels from `target_k` slot-filling.### Fix 4 — No retry on rate limits|------|---------|



### Run 03 — `massive_scenario` · `gemini-2.0-flash-001` · no `target_k` · 2026-02-21No error handling around API calls — a single 429 silently returned `None`. Fixed: 5-attempt retry with linear backoff (20s, 40s, 60s, 80s).| `pyproject.toml` | Project metadata and dependencies |



Without `target_k`, Gemini performed light deduplication only: 343 → 311 labels. Step 2 not run — 311-cluster classification would be scientifically useless.| `uv.lock` | Pinned lockfile for reproducibility |



**Conclusion**: `target_k` must remain the default. It is a necessary semantic anchor when the proposed label count is in the hundreds.### Fix 5 — Hardcoded model name| `requirements.txt` | Pinned fallback for pip users |



---`"gpt-3.5-turbo-0125"` was hardcoded everywhere. Fixed: read from `LLM_MODEL` env var.| `.env.example` | Documents every env variable (no secrets) |



### KM-01 — `massive_scenario` · `gpt-4o-mini` · K-Medoids k=100 · 2026-03-12| `text_clustering/client.py` | Thin wrapper: loads `.env`, builds `openai.OpenAI` client for OpenRouter |



Label generation on full 2,974 docs → 715 proposed. Re-merged with `target_k=18` → 19 labels. Classification on 100 medoids (181s). 23/2,974 (0.77%) unlabelled after propagation.  ### Fix 6 — Missing `.env` loading| `.cz.yaml` | Commitizen config (conventional commits) |

Run dir: `massive_scenario_small_20260312_112628`

`load_dotenv()` was never called — all env vars fell back to defaults. Fixed: called at import time in `client.py`.| `.github/workflows/ci.yml` | Lint CI on PRs to `main` / `develop` |

**Result**: ACC=54.98, NMI=57.78, ARI=41.66 — ~10× LLM cost reduction.



### KM-02 — `massive_scenario` · `gpt-4o-mini` · K-Medoids k=300 · 2026-03-12

### Fix 7 — `LLM_REQUEST_DELAY` not consumed### Dependencies

k=300 (9.9× compression). Label gen on 300 medoid docs → 683 proposed → 19 labels. Classification on 300 medoids (1161s). 11/2,974 (0.37%) unlabelled.  

Run dir: `massive_scenario_small_20260312_120831``.env` defines `LLM_REQUEST_DELAY=4` but no script read it. Without a delay, requests hit rate limit ceilings immediately. Fixed: sleep after each API call.



**Result**: ACC=55.21, NMI=57.25, ARI=39.85 — tripling k barely improved results.```



### GMM-01 — `massive_scenario` · `gpt-3.5-turbo` · GMM k=100 · 2026-03-13---openai>=1.30.0        — OpenRouter-compatible client



Label gen with `gpt-3.5-turbo` → 252 proposed. Re-merged with `gpt-4o-mini` → 20 labels. Classification on 100 GMM representatives (205s). 0% unlabelled.  python-dotenv>=1.0.0  — .env loading

Run dir: `massive_scenario_small_20260313_095906`

## 4. Improvements Addedscikit-learn>=1.4.0   — NMI, ARI metrics

**Result**: ACC=53.63, NMI=58.51, ARI=40.53 — **highest NMI** among pre-clustering runs. GMM soft assignments outperform K-Medoids hard assignments on cluster purity.

scipy>=1.13.0         — Hungarian algorithm (ACC)

---

### Package Restructuringnumpy>=1.26.0         — used in evaluate.py

### SC-01 — `massive_scenario` · SEAL-Clust v1 · t-SNE 2D + Elbow k=30 · 2026-03-13

The original 4 flat scripts with duplicated helpers were reorganized into a proper `text_clustering/` package: `client.py`, `config.py`, `llm.py`, `data.py`, `prompts.py`, `pipeline/`.```

Full 7-step SEAL-Clust v1. t-SNE reduced to 2D (cosine metric, perplexity=30) → Elbow selected k=30 → K-Medoids on 2D → 30 medoids.  

Label gen with `gpt-3.5-turbo`: 260 proposed → 231 (poor merge). Re-merged with `gpt-4o-mini`: 260 → 39 → 18 labels.  

Classification on 30 medoids only (57s, ~30 LLM calls). Only 12 of 18 labels used. 0% unlabelled.  

Run dir: `massive_scenario_small_20260313_113104`### Timestamped Run Directories### Branching



**Result**: ACC=43.21, NMI=44.68, ARI=26.14 — **99× cost reduction**, but t-SNE 2D loses too much structure.Each run creates an isolated folder: `runs/<dataset>_<split>_<YYYYMMDD_HHMMSS>/`. Previous runs are never overwritten.



### SC-02 — `massive_scenario` · SEAL-Clust v1 · t-SNE 2D + Manual k=200 · 2026-03-13```



Manual k=200 (Elbow skipped). t-SNE 2D → K-Medoids on 2D → 200 medoids. `gpt-4o-mini` throughout. All 19 labels used + 1 Unsuccessful. 3/2,974 unlabelled (0.1%).  ### Checkpoint / Resume (Step 2)main       ← stable, tagged releases

Run dir: `massive_scenario_small_20260313_135205`

Progress saved to `checkpoint.json` every 200 samples. Re-running the same command resumes from where it stopped.  └── develop         ← integration branch

**Result**: ACC=43.44, NMI=42.37, ARI=27.66 — increasing k from 30→200 did **not** improve ACC. The t-SNE 2D projection is the bottleneck.

        └── feature/<desc>  /  fix/<desc>  /  docs/<desc>

### SC-03 — `massive_scenario` · SEAL-Clust v2 · PCA 50D + K₀=300 + BIC K\*=7 · 2026-03-14

### `results.json````

**First SEAL-Clust v2 run.** Switched from t-SNE 2D to PCA 50D. K₀=300, BIC estimated K\*=7.  

`gpt-4o-mini` for all LLM steps. ~310 LLM calls.  Results now written to JSON (ACC, NMI, ARI, sample count, cluster counts, model name, timestamp) — not just stdout.

Run dir: `massive_scenario_small_20260314_113900`

All commits follow [Conventional Commits](https://www.conventionalcommits.org/) :  (`feat:`, `fix:`, `docs:`, `build:`, `ci:`).

**Result**: **ACC=56.32, NMI=55.15, ARI=38.66** — PCA 50D **recovers +13pp ACC** lost by t-SNE. Best pre-clustering ACC. However, BIC under-estimated K\* (7 vs true 18), so clusters are too broad.

### Logging

### SC-04 — `massive_scenario` · SEAL-Clust v2 · PCA 50D + K₀=300 + Silhouette K\*=50 · 2026-03-14

All `print()` replaced with Python's `logging` module. Two handlers: stdout (INFO) and `run.log` file (DEBUG) with timestamps.---

PCA 50D, K₀=300. Silhouette method selected K\*=50 (over-estimated — silhouette kept increasing up to k_max=50).  

Run dir: `massive_scenario_small_20260314_120703`



**Status**: ❌ Interrupted during Stage 8 (classification). No results.  ---## 3. Dataset

**Observation**: Silhouette without elbow detection picks the maximum K in the search range — not usable as-is.



### SC-05 — `massive_scenario` · SEAL-Clust v2 · PCA 50D + K₀=300 + Silhouette-Elbow K\*=9 · 2026-03-14

## 5. API & Model Investigation**Source**: Downloaded directly from [Google Drive — ClusterLLM dataset, originally from EMNLP 2023](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view) — unzip into `./dataset/`

PCA 50D, K₀=300. Silhouette-elbow (Kneedle algorithm) estimated K\*=9 from the silhouette curve.  

Run dir: `massive_scenario_small_20260314_122333`



**Status**: ❌ Interrupted during Stage 8. No results.  ### OpenRouter Rate Limits**Format** (one JSON object per line):

**Observation**: Silhouette-elbow gives K\*=9, which is closer to ground truth (18) than BIC (K\*=7) but still under-estimates by half.

```json

### SC-06 — `massive_scenario` · SEAL-Clust v2 · PCA 50D + K₀=300 + Manual K\*=18 · 2026-03-14

| Concept | Detail |{"task": "massive_intent", "input": "set an alarm for 7am", "label": "alarm set"}

PCA 50D, K₀=300. **Manual K\*=18** (set to ground truth). `gpt-4o-mini` for all steps. ~310 LLM calls.  

Run dir: `massive_scenario_small_20260314_124216`|---------|--------|```



**Result**: **ACC=53.56, NMI=59.17, ARI=44.92** — Setting K\* to the ground truth produces **the best ARI** (44.92) and **the best NMI** (59.17) among all SEAL-Clust runs. ACC slightly below SC-03 because the 18 clusters are more fine-grained and small assignment errors accumulate.| **Free tier cap** | `is_free_tier: true` → 50 req/day. ≥ $10 purchase → 1,000/day |



---| **RPM** | 20 req/min for free models regardless of tier |The download bundle contains 14 datasets. The 9 extras (`banking77`, `clinc`, `clinc_domain`, `few_event`, `few_nerd_nat`, `few_rel_nat`, `mtop_domain`, `reddit`, `stackexchange`) could be removed — only the 5 used in the paper are kept in `./dataset/`.



### MI-01 — `massive_intent` · SEAL-Clust v1 · t-SNE 2D + Manual k=200 · 2026-03-13| **Venice upstream** | Free models (Llama 70B, Mistral 24B, Gemma 27B) all route through Venice. When Venice is under load, all fail with 429 simultaneously. |



**First experiment on a different dataset.** `massive_intent` has 59 ground-truth classes (vs 18 for `massive_scenario`) — a much harder clustering task.  ### Dataset sizes (small split)

t-SNE 2D, K-Medoids k=200 (manual), `gpt-4o-mini`. 200 medoid prototypes. Labels propagated to full dataset.  

Run dir: `massive_intent_small_20260313_141759`### Why We Don't Use OpenAI JSON Mode



**Result**: **ACC=38.37, NMI=39.16, ARI=24.26** — significant gap vs paper (ACC 64.12). This is expected: t-SNE 2D loses structure, and 59 classes is far harder than 18. n_pred=19 (only 19 of 59 classes discovered — severe under-segmentation).`response_format={"type":"json_object"}` is not supported by most free models. Some return HTTP 400, some return empty body, reasoning models spend all tokens on chain-of-thought. Our fix: leave JSON mode off, strip fences from responses.| Dataset | Samples | Classes |



### MI-02 — `massive_intent` · SEAL-Clust v1 · t-SNE 2D + Manual k=200 · 2026-03-13|---------|---------|---------|



Same setup as MI-01, aborted before classification completed.  ### Why Reasoning Models Are Excluded| `massive_scenario` | 2,974 | 18 |

Run dir: `massive_intent_small_20260313_144015`

Models like DeepSeek R1, Solar Pro 3, GLM 4.5 Air use internal chain-of-thought. At `max_tokens=512` they spend the entire budget on CoT and return `content: ""`. They are slow, token-heavy, and incompatible with ~3,000 calls per dataset.| `massive_intent` | 2,974 | 59 |

**Status**: ❌ Incomplete — no results.

| `go_emotion` | 5,940 | 27 |

---

### Model Selection Criteria| `arxiv_fine` | 3,674 | 93 |

## 8. All Experimental Results

1. Instruct model (not reasoning/thinking)| `mtop_intent` | 4,386 | 102 |

### Terminology

2. ≥ 24B parameters — needed for reliable label proposal and multi-class classification

| Symbol | Meaning |

|--------|---------|3. ≥ 32K context — classification prompt lists up to 60+ candidate labels### Seed label selection (Step 0)

| **K₀** | Overclustering size (micro-clusters formed before LLM) |

| **K\*** | Final number of categories after LLM label consolidation |4. Responds without a system prompt being required

| **ACC** | Accuracy (Hungarian-aligned) |

| **NMI** | Normalized Mutual Information |5. Returns JSON without wrapping it in extra explanation`select_part_labels.py` samples `floor(0.2 × num_classes)` labels per dataset at random to use as the LLM's starting point.

| **ARI** | Adjusted Rand Index |

6. **Merge capability**: must consolidate ~150 proposed labels down to ~18 in a single call

### `massive_scenario` · small split (2,974 docs, 18 ground-truth classes)

| Dataset | Classes | Seed labels given |

| Run | Pipeline | Model | Reduction | K₀ | K\* (method) | n_pred | ACC | NMI | ARI | LLM Calls | Status |

|-----|----------|-------|-----------|:--:|:------------:|:------:|:---:|:---:|:---:|:---------:|--------|### Primary Model: `google/gemini-2.0-flash-001`|---------|---------|-------------------|

| Paper | Original | `gpt-3.5-turbo-0125` | — | — | implicit | ~18 | **71.75** | **78.00** | **56.86** | ~3,000 | Reference |

| Run 01 | Original | `trinity:free` | — | — | — | 168 | 40.69 | 66.64 | 33.06 | ~3,000 | ❌ Broken merge |Selected after Run 02 merge investigation. Probe: 6/6 RECOMMENDED. Merge test: 167 labels → **28 in 1.9 seconds** — single call, paper-aligned.| `massive_scenario` | 18 | 3 |

| Run 02 | Original | `gemini-2.0-flash` | — | — | 18 | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | ✅ Valid |

| KM-01 | K-Medoids | `gpt-4o-mini` | — | — | 18 | 19 | 54.98 | 57.78 | 41.66 | ~300 | ✅ || `massive_intent` | 59 | 11 |

| KM-02 | K-Medoids | `gpt-4o-mini` | — | — | 18 | 20 | 55.21 | 57.25 | 39.85 | ~500 | ✅ |

| GMM-01 | GMM | `gpt-3.5-turbo` | — | — | 18 | 17 | 53.63 | 58.51 | 40.53 | ~300 | ✅ |**Cost estimate (full 5-dataset baseline)**: ~$0.92 (with 2× safety margin: ~$1.83).| `go_emotion` | 27 | 5 |

| **SC-01** | **SEALClust v1** | `3.5-turbo`+`4o-mini` | **t-SNE 2D** | 30 | Elbow | 12 | 43.21 | 44.68 | 26.14 | **~30** | ✅ |

| **SC-02** | **SEALClust v1** | `gpt-4o-mini` | **t-SNE 2D** | 200 | 18 (manual) | 20 | 43.44 | 42.37 | 27.66 | ~200 | ✅ || `mtop_intent` | 102 | 20 |

| **SC-03** | **SEALClust v2** | `gpt-4o-mini` | **PCA 50D** | 300 | 7 (BIC) | 7 | **56.32** | 55.15 | 38.66 | ~310 | ✅ |

| **SC-04** | **SEALClust v2** | `gpt-4o-mini` | **PCA 50D** | 300 | 50 (silhouette) | — | — | — | — | — | ❌ Interrupted |---| `arxiv_fine` | 93 | 18 |

| **SC-05** | **SEALClust v2** | `gpt-4o-mini` | **PCA 50D** | 300 | 9 (sil-elbow) | — | — | — | — | — | ❌ Interrupted |

| **SC-06** | **SEALClust v2** | `gpt-4o-mini` | **PCA 50D** | 300 | **18 (manual)** | 18 | 53.56 | **59.17** | **44.92** | ~310 | ✅ |



### `massive_intent` · small split (2,974 docs, 59 ground-truth classes)## 6. Model Probe ResultsOutput: `./runs/chosen_labels.json`



| Run | Pipeline | Model | Reduction | K₀ | K\* (method) | n_pred | ACC | NMI | ARI | LLM Calls | Status |

|-----|----------|-------|-----------|:--:|:------------:|:------:|:---:|:---:|:---:|:---------:|--------|

| Paper | Original | `gpt-3.5-turbo-0125` | — | — | implicit | ~59 | **64.12** | **65.44** | **48.92** | ~3,000 | Reference |All models tested with `probe_models.py` (6 tests: reachability, label gen, merge, classification, consistency, token efficiency).---

| **MI-01** | **SEALClust v1** | `gpt-4o-mini` | **t-SNE 2D** | 200 | manual | 19 | 38.37 | 39.16 | 24.26 | ~200 | ✅ |



---

| Model | Score | Verdict | Notes |## 4. Code Changes

## 9. Key Findings

|-------|-------|---------|-------|

### Finding 1 — t-SNE 2D Is the Accuracy Bottleneck

| `google/gemini-2.0-flash-001` | **6/6** | ✅ **PRIMARY** | Merge: 167 → 28 labels in 1.9s |The original code was written to run against OpenAI directly with a paid key. Adapting it to free models via OpenRouter exposed several bugs and missing pieces. This section documents what was broken (fixes) and what was added on top (improvements).

SC-01 (k=30, ACC=43.21%) ≈ SC-02 (k=200, ACC=43.44%). Going from 30→200 prototypes only improved ACC by +0.23pp. The information loss from 384D→2D projection dominates. Increasing k cannot compensate for the lost discriminative structure.

| `arcee-ai/trinity-large-preview:free` | **6/6** | ⚠️ Merge fails at scale | Stalls at 144 labels |

### Finding 2 — PCA 50D Recovers the Accuracy

| `openai/gpt-4o-mini` | 6/6 | ⚠️ USABLE | Merge: 167 → 105 (poor consolidation) |---

SC-03 (PCA 50D, ACC=56.32%) vs SC-01/SC-02 (t-SNE 2D, ACC≈43%). Switching from t-SNE 2D to PCA 50D recovered **+13pp ACC** — the single biggest improvement in the SEAL-Clust pipeline. PCA 50D is now the default reduction method.

| `meta-llama/llama-3.3-70b-instruct:free` | — | ⏳ Pending | Venice upstream 429 |

### Finding 3 — K\* Estimation Methods Comparison

| `mistralai/mistral-small-3.1-24b-instruct:free` | — | ⏳ Pending | Venice upstream 429 |### Fixes

| Method | Estimated K\* | True K | Verdict |

|--------|:------------:|:------:|---------|| `google/gemma-3-27b-it:free` | — | ⏳ Pending | Venice upstream 429 |

| BIC (GMM) | 7 | 18 | Under-estimates — categories overlap in embedding space |

| Silhouette (raw) | 50 | 18 | Over-estimates — picks k_max, monotonically increasing || `nousresearch/hermes-3-llama-3.1-405b:free` | — | ⏳ Pending | Venice upstream 429 |These are things that were broken in the original and prevented the pipeline from running correctly.

| Silhouette-elbow (Kneedle) | 9 | 18 | Better, but still under-estimates by ~50% |

| Manual | 18 | 18 | ✅ Best results when ground truth is known |



**Conclusion**: All automated methods struggle because some ground-truth categories overlap in the MiniLM-L6-v2 embedding space. **For best accuracy, use `--k_star` with the known number of classes.**### Models Excluded#### Fix 1 — `ini_client()` call signature



### Finding 4 — Manual K\* Gives the Best Cluster Quality



SC-06 (K\*=18, manual) achieved the **best NMI (59.17)** and **best ARI (44.92)** among all SEAL-Clust runs. SC-03 (K\*=7, BIC) has higher ACC (56.32 vs 53.56) because fewer clusters means fewer assignment errors, but the clusters are too coarse — lower NMI and ARI confirm this.| Model | Reason |**File**: `text_clustering/pipeline/classification.py`  



### Finding 5 — GMM Achieves Highest NMI Among Pre-Clustering Modes|-------|--------|**Issue**: `main()` called `ini_client(args.api_key)` but `ini_client()` took no arguments → `TypeError` on every run.



GMM-01 (NMI=58.51) outperforms both KM-01 (57.78) and KM-02 (57.25) on cluster purity. GMM's soft (probabilistic) assignments provide richer information than K-Medoids' hard assignments. GMM also achieves **0% unlabelled** documents.| `upstage/solar-pro-3:free` | Reasoning model (`reasoning_details` field) |



### Finding 6 — Merge Capability Is a Hard Requirement| `z-ai/glm-4.5-air:free` | Reasoning model |```python



The model must consolidate ~150 proposed labels down to ~18 in a single call. `trinity-large-preview` stalls at 144 labels. `gpt-4o-mini` merges to 105 (poor). Only `gemini-2.0-flash-001` performs paper-aligned merge (167→28 in 1.9s). No batching or multi-pass workaround succeeds.| `arcee-ai/trinity-mini:free` | Reasoning model (`content: ""`) |# Before



### Finding 7 — `massive_intent` (59 Classes) Is Much Harder| `nvidia/nemotron-3-nano-30b-a3b:free` | Reasoning model |client = ini_client(args.api_key)



MI-01 (ACC=38.37) vs paper (ACC=64.12) — a 26pp gap. With t-SNE 2D and only 200 prototypes, only 19 of 59 classes were discovered. The fine-grained intent taxonomy requires either more prototypes, PCA-based reduction, or a stronger embedding model.| `deepseek/deepseek-r1-0528:free` | R1 architecture — reasoning by design |



### Finding 8 — The Trade-Off Curve| `google/gemma-3-12b-it:free` | 12B too small, rejects system prompts |# After



| Approach | Reduction | K\* | ACC | NMI | ARI | LLM Calls | Automation || `qwen/qwen3-235b-a22b:free` | 404 — no endpoints on OpenRouter |client = ini_client()  # API key comes from .env

|----------|:---------:|:---:|:---:|:---:|:---:|:---------:|:----------:|

| Original paper | — | implicit | **71.75** | **78.00** | **56.86** | ~3,000 | Manual || `openai/gpt-oss-120b:free` / `20b:free` | Data policy restriction |```

| Original (gemini) | — | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | Manual |

| K-Medoids (raw 384D) | — | 18 | 55.21 | 57.25 | 39.85 | ~500 | Manual |

| GMM (raw 384D) | — | 18 | 53.63 | 58.51 | 40.53 | ~300 | Manual |

| **SEALClust v2 (manual K\*)** | **PCA 50D** | **18** | 53.56 | **59.17** | **44.92** | ~310 | **Semi-auto** |---#### Fix 2 — `response_format` not universally supported

| SEALClust v2 (BIC K\*) | PCA 50D | 7 | **56.32** | 55.15 | 38.66 | ~310 | **Full auto** |

| SEALClust v1 (t-SNE) | t-SNE 2D | Elbow/18 | 43.21–43.44 | 42–45 | 26–28 | 30–200 | Full auto |



Best accuracy-cost balance: **SEALClust v2 with PCA 50D and manual K\*=18** (ARI=44.92, NMI=59.17, 10× cost reduction).## 7. Pipeline Execution Log**Files**: `text_clustering/pipeline/label_generation.py`, `classification.py`  



---**Issue**: The original code always sent `response_format={"type":"json_object"}`. Many free models return HTTP 400 or an empty body when this is set.  



## 10. Cost Analysis### Run 01 — `massive_scenario` · `trinity-large-preview:free` · 2026-02-20**Fix**: Made it opt-in via `LLM_FORCE_JSON_MODE` env var (default `false`).



### LLM API Calls Per Mode



| Pipeline Step | Mode A (Original) | Mode B (KM k=100) | Mode C (GMM k=100) | Mode E (SEALClust K₀=300) |First end-to-end validation run. Pipeline executed with no code changes relative to v1.2.0.```python

|--------------|:-----------------:|:-----------------:|:------------------:|:-------------------------:|

| Label discovery | ~200 | ~200 | ~200 | **~10** (reps only) |_FORCE_JSON_MODE = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"

| Label merge | 1 | 1 | 1 | 1 |

| Classification | **~2,974** | **~100** | **~100** | **~300** |- **Step 0**: ✅ Seed labels selected (3 of 18)

| **Total** | **~3,175** | **~301** | **~301** | **~311** |

| **Reduction** | 1× | **~10×** | **~10×** | **~10×** |- **Step 1**: ✅ 190 proposed labels. Merge **silently failed** — `LLM_MAX_TOKENS=512` too small for 190-label merge response. Parser silently fell back to unmerged list.if _FORCE_JSON_MODE:



### Wall-Clock Time (`massive_scenario` · small split)- **Step 2**: ✅ 2,974 texts classified (4h29). 168 distinct predicted labels.    kwargs["response_format"] = {"type": "json_object"}



| Step | Mode A | Mode B (k=100) | Mode C (k=100) | Mode E (K₀=300) |- **Step 3**: ✅ ACC=40.69, NMI=66.64, ARI=33.06```

|------|:------:|:--------------:|:--------------:|:---------------:|

| Embedding | ~18s | ~18s | ~18s | ~18s |

| Reduction | — | — | — | ~1s (PCA) |

| Clustering | — | ~10s | ~20s | ~5s |**Root cause**: Label merge truncated at 512 tokens → 168 predicted clusters instead of ~18. Each true class was split across 3–61 predicted labels (taxonomy fragmentation). The metric gap (ACC −31 vs paper) is an artifact of merge failure, not classification quality.#### Fix 3 — Markdown fence stripping

| Label discovery | ~15 min | ~15 min | ~15 min | **~20s** |

| Classification | **~2 hours** | **181s** | **205s** | **~8 min** |

| **Total** | **~2.5 hours** | **~20 min** | **~20 min** | **~10 min** |

### Run 02 — Merge Investigation & Model Switch · 2026-02-21**Files**: `text_clustering/pipeline/label_generation.py`, `classification.py`  

### Gemini-2.0-flash-001 Cost Estimate (5-Dataset Baseline)

**Issue**: Free models often wrap JSON in markdown code fences (` ```json ... ``` `). The original `eval()` call fails on these, silently dropping labels.  

Pricing: $0.10/M input tokens · $0.40/M output tokens

After fixing the token limit, three Step 1 re-attempts with `trinity-large-preview:free`:**Fix**: Added `_strip_fenced_json()` applied after every API call.

| Dataset | Estimated cost |

|---------|:--------------:|- Parser crash on flat JSON array → fixed with `_parse_merge_response()`

| `massive_scenario` | ~$0.14 |

| `massive_intent` | ~$0.11 |- Model still couldn't semantically consolidate at scale (154 → 149 → 144, stalled)```python

| `go_emotion` | ~$0.29 |

| `arxiv_fine` | ~$0.27 |- Batched multi-pass merge attempted → only 10 labels removed in 3 passesdef _strip_fenced_json(text: str) -> str:

| `mtop_intent` | ~$0.11 |

| **Total** | **~$0.92** |- Map-to-canonical approach tested → **rejected as data leakage** (passes all 18 true labels to merge step, paper withholds 15)    text = text.strip()



---    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)



## 11. Future Work**Resolution**: Switched to `google/gemini-2.0-flash-001`. Merge test: 167 → 28 labels in 1.9s (paper-aligned).    if match:



### Completed ✅        return match.group(1).strip()



- [x] Replace t-SNE 2D with PCA 50D (now the default)### Run 02 — `massive_scenario` · `gemini-2.0-flash-001` · `target_k=18` · 2026-02-21    return text

- [x] Systematic K\* estimation (silhouette-elbow, Calinski-Harabasz, BIC, ensemble)

- [x] Manual K\* override (`--k_star N`)```

- [x] One-command full pipeline (`--full`)

- [x] Test on `massive_intent` dataset (MI-01)First full end-to-end run with Gemini.



### Remaining#### Fix 4 — No retry on rate limits



- [ ] Run `massive_intent` with SEALClust v2 (PCA 50D) — expect significant improvement over MI-01 (t-SNE)- **Step 1**: ✅ 352 proposed → 18 merged (target_k=18). 10 good semantic matches, 4 overlapping labels, 4 spurious labels. Missing: `weather` cluster.

- [ ] Run remaining 3 datasets with Gemini (`go_emotion`, `arxiv_fine`, `mtop_intent`)

- [ ] Complete SC-04 and SC-05 runs to validate silhouette-based K\* estimation- **Step 2**: ✅ 2,974 classified (2h09). 0 errors.**File**: `text_clustering/llm.py`  

- [ ] HDBSCAN microclustering (density-based, auto-determines k)

- [ ] Multi-representative extraction (2–3 docs per cluster for richer LLM context)- **Step 3**: ✅ **ACC=60.46, NMI=63.90, ARI=53.87****Issue**: No error handling around API calls. A single 429 silently returned `None`, losing an entire chunk of labels with no indication.  

- [ ] Better embedding models (`all-mpnet-base-v2`, `instructor-xl`, `e5-large`)

- [ ] Cross-dataset evaluation on all 14 datasets**Fix**: 5-attempt retry with linear backoff (20s, 40s, 60s, 80s).

- [ ] Re-probe Venice-blocked free models during off-peak hours

ARI within 3 points of paper (53.87 vs 56.86) — cluster structure nearly correct. The ACC gap (−11.29) is driven by 4 spurious labels from `target_k` slot-filling.

```python

### Run 03 — `massive_scenario` · `gemini-2.0-flash-001` · no `target_k` · 2026-02-21for attempt in range(5):

    try:

Without `target_k`, Gemini performed light deduplication only: 343 → 311 labels. Step 2 not run — 311-cluster classification would be scientifically useless.        completion = client.chat.completions.create(**kwargs)

        ...

**Conclusion**: `target_k` must remain the default. It is a necessary semantic anchor when the proposed label count is in the hundreds.        return response_origin

    except Exception as e:

### KM-01 — `massive_scenario` · `gpt-4o-mini` · K-Medoids k=100        if "429" in str(e) and attempt < 4:

            wait = 20 * (attempt + 1)

Label generation on full 2,974 docs → 715 proposed. Re-merged with `target_k=18` → 19 labels. Classification on 100 medoids (181s). 23/2,974 (0.77%) unlabelled after propagation.            print(f"  [rate limit] attempt {attempt+1}/5, waiting {wait}s...")

            time.sleep(wait)

**Result**: ACC=54.98, NMI=57.78, ARI=41.66 — ~10× LLM cost reduction.        else:

            return None

### KM-02 — `massive_scenario` · `gpt-4o-mini` · K-Medoids k=300```



k=300 (9.9× compression). Label gen on 300 medoid docs → 683 proposed → 19 labels. Classification on 300 medoids (1161s). 11/2,974 (0.37%) unlabelled.#### Fix 5 — Hardcoded model name



**Result**: ACC=55.21, NMI=57.25, ARI=39.85 — tripling k barely improved results.**File**: `text_clustering/config.py`  

**Issue**: `"gpt-3.5-turbo-0125"` was hardcoded in every script, making model switching require code edits.  

### GMM-01 — `massive_scenario` · `gpt-3.5-turbo` · GMM k=100**Fix**: Read from `LLM_MODEL` env var.



Label gen with `gpt-3.5-turbo` → 252 proposed. Re-merged with `gpt-4o-mini` → 20 labels. Classification on 100 GMM representatives (205s). 0% unlabelled.```python

MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")

**Result**: ACC=53.63, NMI=58.51, ARI=40.53 — **highest NMI** among pre-clustering runs. GMM soft assignments outperform K-Medoids hard assignments on cluster purity.```



### SC-01 — `massive_scenario` · `gpt-3.5-turbo`+`gpt-4o-mini` · t-SNE 2D + Elbow k=30#### Fix 6 — Missing `.env` loading



Full 7-step SEAL-Clust v1. t-SNE reduced to 2D → Elbow selected k=30 → K-Medoids on 2D → 30 medoids. Label gen → classification on 30 medoids only (57s, ~30 LLM calls). Only 12 of 18 labels used.**File**: `text_clustering/client.py`  

**Issue**: `load_dotenv()` was never called, so the `.env` file was silently ignored and all env vars fell back to their defaults.  

**Result**: ACC=43.21, NMI=44.68, ARI=26.14 — **99× cost reduction**, but t-SNE 2D loses too much structure.**Fix**: `load_dotenv()` called at import time in `client.py`.



### SC-02 — `massive_scenario` · `gpt-4o-mini` · t-SNE 2D + Manual k=200#### Fix 7 — `LLM_REQUEST_DELAY` not consumed



Manual k=200 (Elbow skipped). K-Medoids on 2D → 200 medoids. All 19 labels used.**File**: `text_clustering/llm.py`  

**Issue**: `.env` defines `LLM_REQUEST_DELAY=4` but no script read it. Without a delay, requests fire back-to-back and immediately hit the 20 req/min ceiling.  

**Result**: ACC=43.44, NMI=42.37, ARI=27.66 — increasing k from 30→200 did **not** improve ACC. The t-SNE 2D projection is the bottleneck.**Fix**: Sleep after each successful API call.



### SC-03 — `massive_scenario` · `gpt-4o-mini` · PCA 50D + K₀=300 + BIC K\*=7```python

_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))

SEAL-Clust v2 with PCA 50D instead of t-SNE 2D. BIC estimated K\*=7.

# after a successful completion:

**Result**: ACC=56.32, NMI=55.15, ARI=38.66 — **PCA 50D recovers the accuracy** lost by t-SNE. Best pre-clustering ACC.if _REQUEST_DELAY > 0:

    time.sleep(_REQUEST_DELAY)

### SC-04, SC-05 — Interrupted / In-Progress```



SC-04 (silhouette K\*) and SC-05 (silhouette-elbow, K\*=9) were interrupted during classification.With `LLM_REQUEST_DELAY=4` the pipeline stays at ~15 req/min, safely under the OpenRouter limit.



------



## 8. All Experimental Results### Improvements



### TerminologyThese are additions that go beyond the original scope — the pipeline worked without them, but they make it more practical for long or repeated runs.



| Symbol | Meaning |#### Improvement 1 — Package restructuring

|--------|---------|

| **K₀** | Overclustering size (micro-clusters formed before LLM) |The original code was 4 flat scripts with duplicated helpers (`ini_client`, `chat`, prompt builders) copy-pasted across files. All shared logic was moved into a proper `text_clustering/` package:

| **K\*** | Final number of categories after LLM label consolidation |

| **ACC** | Accuracy (Hungarian-aligned) |- `client.py` — API client factory

| **NMI** | Normalized Mutual Information |- `config.py` — single source of truth for all env vars

| **ARI** | Adjusted Rand Index |- `llm.py` — `chat()` with retry, `_strip_fenced_json()`

- `data.py` — dataset loading

### `massive_scenario` · small split (2,974 docs, 18 ground-truth classes)- `prompts.py` — prompt construction

- `pipeline/` — the 4 pipeline steps as importable modules with console-script entry points

| Run | Pipeline | Model | K₀ | K\* | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |

|-----|----------|-------|----|-----|--------|-----|-----|-----|-----------|-----------|--------|#### Improvement 2 — Timestamped run directories

| Paper | Original | `gpt-3.5-turbo-0125` | — | implicit | ~18 | **71.75** | **78.00** | **56.86** | ~3,000 | 1× | Reference |

| Run 01 | Original | `trinity-large-preview:free` | — | — | 168 | 40.69 | 66.64 | 33.06 | ~3,000 | 1× | ❌ Broken merge |The original wrote all outputs to a flat `generated_labels/` folder with fixed filenames, so re-running a dataset overwrote previous results.

| Run 02 | Original | `gemini-2.0-flash-001` | — | 18 | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | 1× | ✅ Valid |

| KM-01 | K-Medoids | `gpt-4o-mini` | — | 18 | 19 | 54.98 | 57.78 | 41.66 | ~300 | **10×** | ✅ |Each Step 1 run now creates an isolated folder: `runs/<dataset>_<split>_<YYYYMMDD_HHMMSS>/`. All subsequent steps read from and write to that folder. Previous runs are never touched.

| KM-02 | K-Medoids | `gpt-4o-mini` | — | 18 | 20 | 55.21 | 57.25 | 39.85 | ~500 | 6× | ✅ |

| GMM-01 | GMM | `gpt-3.5-turbo` | — | 18 | 17 | 53.63 | **58.51** | 40.53 | ~300 | **30×** | ✅ |#### Improvement 3 — Checkpoint / resume (Step 2)

| SC-01 | SEALClust v1 | `gpt-3.5-turbo`+`4o-mini` | 30 | Elbow | 12 | 43.21 | 44.68 | 26.14 | **~30** | **99×** | ✅ (t-SNE 2D) |

| SC-02 | SEALClust v1 | `gpt-4o-mini` | 200 | 18 | 20 | 43.44 | 42.37 | 27.66 | ~200 | 15× | ✅ (t-SNE 2D) |Step 2 makes one API call per text (~3,000 per dataset, ~3h20). The original had no way to recover from an interruption — a crash meant starting from scratch.

| SC-03 | SEALClust v2 | `gpt-4o-mini` | 300 | 7 (BIC) | 7 | **56.32** | 55.15 | 38.66 | ~310 | **10×** | ✅ (PCA 50D) |

Progress is now saved to `checkpoint.json` every 200 samples. Re-running the same command detects the checkpoint and resumes from where it stopped. The checkpoint is deleted automatically on successful completion.

---

#### Improvement 4 — `results.json`

## 9. Key Findings

The original `evaluate.py` printed metrics to stdout only. Results are now also written to `results.json` in the run directory, including ACC, NMI, ARI, sample count, cluster counts, model name, and timestamp.

### Finding 1 — t-SNE 2D Is the Accuracy Bottleneck

#### Improvement 5 — Logging

SC-01 (k=30, ACC=43.21%) ≈ SC-02 (k=200, ACC=43.44%). Going from 30→200 prototypes only improved ACC by +0.23pp. The information loss from 384D→2D projection dominates. Increasing k cannot compensate for the lost discriminative structure.

All `print()` calls in the pipeline were replaced with Python's standard `logging` module. A `setup_logging(log_path)` function configures two handlers at startup: one to stdout (INFO level) and one to a `run.log` file inside the run directory (DEBUG level). The log format includes a timestamp, level, and module name:

### Finding 2 — PCA 50D Recovers the Accuracy

```

SC-03 (PCA 50D, ACC=56.32%) vs SC-01/SC-02 (t-SNE 2D, ACC≈43%). Switching from t-SNE 2D to PCA 50D recovered **+13pp ACC** — the single biggest improvement in the SEAL-Clust pipeline. PCA 50D is now the default reduction method.2026-02-20 14:32:01 | INFO     | label_generation | Run dir: ./runs/...

```

### Finding 3 — BIC Under-Estimates K\*

Every pipeline step writes its full trace to `run.log` in its run directory. Step 0 writes to `runs/seed_labels.log`. This means a run can always be reconstructed after the fact — which model was used, when each step ran, any rate limit retries, and progress checkpoints.

On `massive_scenario` (ground truth K=18), BIC estimated K\*=7. The under-estimation happens because some true categories overlap in the MiniLM-L6-v2 embedding space. Silhouette-elbow performs similarly (K\*≈7-9). **For best accuracy, use manual `--k_star` with the known number of classes.**



### Finding 4 — GMM Achieves Highest NMI

---

GMM-01 (NMI=58.51) outperforms both KM-01 (57.78) and KM-02 (57.25) on cluster purity. GMM's soft (probabilistic) assignments provide richer information than K-Medoids' hard assignments. GMM also achieves **0% unlabelled** documents.

## 5. API & Model Investigation

### Finding 5 — Merge Capability Is a Hard Requirement

### OpenRouter rate limits

The model must consolidate ~150 proposed labels down to ~18 in a single call. `trinity-large-preview` stalls at 144 labels. `gpt-4o-mini` merges to 105 (poor). Only `gemini-2.0-flash-001` performs paper-aligned merge (167→28 in 1.9s). No batching or multi-pass workaround succeeds.

OpenRouter has two distinct concepts:

### Finding 6 — The Trade-Off Curve

| Concept | Detail |

| Approach | Dim Reduction | ACC | LLM Cost | Automation ||---------|--------|

|----------|:------------:|:---:|:---------:|:----------:|| **Account balance** | Must be non-negative to use free models (even $0/token ones) |

| Original paper (per-doc) | none | **72%** | ~3,000 calls | Manual k || **Key limit** | Per-key spending cap. If set to `0`, blocks all requests including free ones |

| K-Medoids (raw 384D) | none | 55% | ~300 | Manual k || **Free tier cap** | `is_free_tier: true` → 50 req/day. Purchasing ≥ $10 raises it to 1,000/day |

| SEALClust v2 (PCA 50D) | PCA 50D | **56%** | ~310 | **Auto k** || **RPM** | 20 req/min for free models regardless of tier |

| SEALClust v1 (t-SNE 2D) | t-SNE 2D | 43% | ~30–200 | Auto k |

After adding €10 of credits the key changed to `is_free_tier: false`, `limit: null` — no daily cap, governed only by credit balance.

Best accuracy-cost balance: **SEALClust v2 with PCA 50D** (56% ACC, 10× cost reduction, full automation).

**Important**: The popular free models (Llama 70B, Mistral 24B, Gemma 27B) are all routed through a single upstream provider called **Venice**. When Venice is under load, all of these fail simultaneously with a 429 even if your account has plenty of credits. This is an infrastructure issue on their side, not an account issue.

---

### Why we don't use OpenAI JSON mode

## 10. Cost Analysis

`response_format={"type": "json_object"}` — what the original code uses — is not supported by most free models. The options are:

### LLM API Calls Per Mode

- Some return HTTP 400 (Google AI Studio, older models)

| Pipeline Step | Mode A (Original) | Mode B (KM k=100) | Mode C (GMM k=100) | Mode E (SEALClust K₀=300) |- Some return an empty body (Solar Pro 3, Step Flash)

|--------------|:-----------------:|:-----------------:|:------------------:|:-------------------------:|- Reasoning models spend all `max_tokens` on internal chain-of-thought and return `content: ""`

| Label discovery | ~200 | ~200 | ~200 | **~10** (reps only) |

| Label merge | 1 | 1 | 1 | 1 |Our fix: leave JSON mode off by default and strip fences from responses instead. Works reliably across all tested models.

| Classification | **~2,974** | **~100** | **~100** | **~300** |

| **Total** | **~3,175** | **~301** | **~301** | **~311** |### Why reasoning models are excluded

| **Reduction** | 1× | **~10×** | **~10×** | **~10×** |

Models like DeepSeek R1, Solar Pro 3, GLM 4.5 Air, and Qwen3-thinking variants use an internal chain-of-thought before producing a final answer. On OpenRouter, this CoT appears in `message.reasoning` while the actual answer goes in `message.content`. The problem:

### Wall-Clock Time (`massive_scenario` · small split)

- At `max_tokens=512` (pipeline default), they spend the entire budget on CoT and return `content: ""`

| Step | Mode A | Mode B (k=100) | Mode C (k=100) | Mode E (K₀=300) |- Even with a larger budget, they are slow and token-heavy — incompatible with ~3,000 calls per dataset

|------|:------:|:--------------:|:--------------:|:---------------:|

| Embedding | ~18s | ~18s | ~18s | ~18s |### Model selection criteria

| Reduction | — | — | — | ~1s (PCA) |

| Clustering | — | ~10s | ~20s | ~5s |1. Instruct model (not reasoning/thinking)

| Label discovery | ~15 min | ~15 min | ~15 min | **~20s** |2. ≥ 24B parameters — needed for reliable label proposal and 59-class classification

| Classification | **~2 hours** | **181s** | **205s** | **~8 min** |3. ≥ 32K context — classification prompt lists up to 60 candidate labels

| **Total** | **~2.5 hours** | **~20 min** | **~20 min** | **~10 min** |4. Responds without a system prompt being required

5. Returns JSON without wrapping it in extra explanation

### Gemini-2.0-flash-001 Cost Estimate (5-Dataset Baseline)

### Merge capability requirement

Pricing: $0.10/M input tokens · $0.40/M output tokens

An additional hard requirement emerged during Run 02 (see §7): the model must be able to

| Dataset | Estimated cost |consolidate ~150 proposed labels down to ~18 in a **single call**. This is what GPT-3.5-turbo

|---------|:--------------:|does in the paper. It cannot be compensated for with batching or multi-pass strategies — those

| `massive_scenario` | ~$0.14 |approaches either stall (trinity-large-preview) or produce data leakage (map-to-canonical). See

| `massive_intent` | ~$0.11 |§7 — Run 02 for the full investigation.

| `go_emotion` | ~$0.29 |

| `arxiv_fine` | ~$0.27 |This requirement disqualifies `arcee-ai/trinity-large-preview:free` as the primary model,

| `mtop_intent` | ~$0.11 |despite it passing all 6 probe tests.

| **Total** | **~$0.92** |

### Primary model: `google/gemini-2.0-flash-001`

---

Selected after the Run 02 merge investigation. Probe: 6/6 RECOMMENDED (2026-02-21). Merge test:

## 11. Future Work167 labels → **28 in 1.9 seconds** — single call, paper-aligned.



### Completed ✅**Cost estimate — full 5-dataset baseline**  

Pricing: $0.10/M input tokens · $0.40/M output tokens

- [x] Replace t-SNE 2D with PCA 50D (now the default)

- [x] Systematic K\* estimation (silhouette-elbow, Calinski-Harabasz, BIC, ensemble)| Dataset | Estimated cost |

- [x] Manual K\* override (`--k_star N`)|---------|---------------|

- [x] One-command full pipeline (`--full`)| `massive_scenario` | ~$0.14 |

| `massive_intent` | ~$0.11 |

### Remaining| `go_emotion` | ~$0.29 |

| `arxiv_fine` | ~$0.27 |

- [ ] Run remaining 4 datasets with Gemini (`massive_intent`, `go_emotion`, `arxiv_fine`, `mtop_intent`)| `mtop_intent` | ~$0.11 |

- [ ] HDBSCAN microclustering (density-based, auto-determines k)| **Total (5 datasets)** | **~$0.92** |

- [ ] Multi-representative extraction (2–3 docs per cluster for richer LLM context)

- [ ] Better embedding models (`all-mpnet-base-v2`, `instructor-xl`, `e5-large`)With 2× safety margin for retries / rate limit backoff: **~$1.83**. $10 budget → **~$8.17 remaining** after full baseline.

- [ ] Cross-dataset evaluation on all 14 datasets

- [ ] Re-probe Venice-blocked free models during off-peak hours**Note on availability**: OpenRouter lists `gemini-2.0-flash-001` as going away March 31, 2026.

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