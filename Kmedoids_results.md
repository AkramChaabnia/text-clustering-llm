# Clustering Results — K-Medoids & GMM

> **Branch**: `feature/kmedoids`
> **Goal**: Reduce LLM API calls by ~30× via document compression (K-Medoids or GMM) while preserving clustering quality.

---

## Paper Baseline (Table 2 — `gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `massive_scenario` | 71.75 | 78.00 | 56.86 |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |

---

## Previous Runs (Original Pipeline — No Pre-Clustering)

### `massive_scenario` · small split

| Run | Model | Pipeline | target_k | n_pred | ACC | NMI | ARI | LLM Calls | Status |
|-----|-------|----------|----------|--------|-----|-----|-----|-----------|--------|
| Paper | `gpt-3.5-turbo-0125` | Original | implicit | ~18 | **71.75** | **78.00** | **56.86** | ~3,000 | Reference |
| Run 01 | `trinity-large-preview:free` | Original | — | 168 | 40.69 | 66.64 | 33.06 | ~3,000 | ❌ Broken merge |
| Run 02 | `gemini-2.0-flash-001` | Original | 18 | 18 | 60.46 | 63.90 | 53.87 | ~3,000 | ✅ Valid |
| Run 03 | `gemini-2.0-flash-001` | Original | none | — | — | — | — | — | ⚠️ Merge failed |

---

## K-Medoids Runs

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` |
| K-Medoids k | 100 / 300 |
| Metric | cosine |
| Init | k-medoids++ |
| Random seed | 42 |

### `massive_scenario` · small split (2,974 docs)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| KM-01 | `gpt-4o-mini` | 100 | 19 | 54.98 | 57.78 | 41.66 | ~300 | ~10× | ✅ Done |
| KM-02 | `gpt-4o-mini` | 300 | 20 | 55.21 | 57.25 | 39.85 | ~500 | ~6× | ✅ Done |

> **KM-01 notes**: Label generation ran on full 2974 docs (not medoids only), producing 715 proposed labels.
> Re-merged with `target_k=18` → 19 labels. Classification ran on 100 medoids only (181s).
> 23/2974 docs (0.77%) received no label during propagation.
> Run dir: `./runs/massive_scenario_small_20260312_112628`
>
> **KM-02 notes**: k=300 (9.9× reduction). Label generation on 300 medoid docs produced 683 proposed labels.
> Re-merged with `target_k=18` → 19 labels (20 predicted clusters incl. "Unsuccessful").
> Classification ran on 300 medoids (1161.3s, includes ~10min API retry pause).
> 11/2974 docs (0.37%) received no label during propagation.
> Run dir: `./runs/massive_scenario_small_20260312_120831`

---

## GMM Runs

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` |
| GMM k | 100 |
| Covariance type | tied |
| n_init | 3 |
| L2-normalised | yes (cosine ≈ euclidean) |
| Random seed | 42 |

### `massive_scenario` · small split (2,974 docs)

| Run | Model | k | n_pred | ACC | NMI | ARI | LLM Calls | Reduction | Status |
|-----|-------|---|--------|-----|-----|-----|-----------|-----------|--------|
| GMM-01 | `gpt-3.5-turbo` | 100 | 17 | 53.63 | 58.51 | 40.53 | ~300 | ~30× | ✅ Done |

> **GMM-01 notes**: Label generation with `gpt-3.5-turbo` on full 2974 docs produced 252 proposed labels.
> `gpt-3.5-turbo` couldn't merge well (229 labels) — re-merged with `gpt-4o-mini` at `target_k=18` → 20 labels.
> Classification with `gpt-3.5-turbo` on 100 GMM representatives (205s). 0 unsuccessful.
> Hard propagation: 0/2974 docs (0%) received no label.
> Run dir: `./runs/massive_scenario_small_20260313_095906`

---

## Summary Comparison

| Method | Model | k | ACC | NMI | ARI | LLM Calls | Unlabelled |
|--------|-------|---|-----|-----|-----|-----------|------------|
| Paper (original) | `gpt-3.5-turbo-0125` | — | **71.75** | **78.00** | **56.86** | ~3,000 | 0% |
| KM-01 | `gpt-4o-mini` | 100 | 54.98 | 57.78 | 41.66 | ~300 | 0.77% |
| KM-02 | `gpt-4o-mini` | 300 | 55.21 | 57.25 | 39.85 | ~500 | 0.37% |
| **GMM-01** | `gpt-3.5-turbo` | 100 | 53.63 | **58.51** | 40.53 | ~300 | **0%** |

**Key observations:**
- GMM-01 achieves the **highest NMI** (58.51) among all pre-clustering runs, beating both KM-01 and KM-02.
- GMM produces **0% unlabelled** documents (every cluster gets a representative), vs 0.37–0.77% for K-Medoids.
- All pre-clustering methods achieve ~10× cost reduction with ~16–18 pp ACC gap vs the paper baseline.
- Tripling k (100→300) in K-Medoids barely improved results while costing more.
- GMM with k=100 + `gpt-3.5-turbo` is the cheapest option with competitive quality.

---

## Full Tutorial — How to Run Each Mode

### Prerequisites

```bash
# 1. Activate environment
conda activate ppd

# 2. Install the package
pip install -e .

# 3. Seed labels (run once — produces runs/chosen_labels.json)
tc-seed-labels
```

### Switching the LLM Provider

Edit `.env` at the project root:

```bash
# ── For OpenAI (direct) ──
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=           ← leave blank or comment out
LLM_MODEL=gpt-3.5-turbo      # or gpt-4o-mini, gpt-4o

# ── For OpenRouter ──
LLM_PROVIDER=openrouter
OPENAI_API_KEY=or-...your-openrouter-key...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-001
```

Save `.env` — all commands read it automatically. No code changes needed.

---

### Mode A — Original Pipeline (No Pre-Clustering)

```bash
# Step 1: Label generation (~200 LLM calls)
tc-label-gen --data massive_scenario
# → prints run_dir, e.g. ./runs/massive_scenario_small_20260313_...

# Step 2: Classification (~2974 LLM calls — one per document)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir>

# Step 3: Evaluation (local, instant)
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~3,000 LLM calls.  **Typical time**: 1–3 hours.

---

### Mode B — K-Medoids Pre-Clustering

```bash
# Step 0: Pre-cluster (embeddings + K-Medoids, ~10–40s, no LLM)
tc-kmedoids --data massive_scenario --kmedoids_k 100
# → prints run_dir. Saves embeddings.npy + kmedoids_metadata.json

# Step 1: Label generation (writes into the same run_dir)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>
# If labels are too many: python tools/remerge_labels.py ./runs/<run_dir> 18

# Step 2: Classify medoids only (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode

# Step 3: Propagate medoid labels → full dataset (local, instant)
tc-kmedoids --data massive_scenario --run_dir ./runs/<run_dir> --propagate

# Step 4: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls.  **Typical time**: 5–15 minutes.

---

### Mode C — GMM Pre-Clustering

```bash
# Step 0: Pre-cluster (embeddings + GMM, ~20–40s, no LLM)
tc-gmm --data massive_scenario --gmm_k 100
# → prints run_dir. Saves embeddings.npy + gmm_metadata.json + gmm_probs.npy

# Step 1: Label generation (writes into the same run_dir)
tc-label-gen --data massive_scenario --run_dir ./runs/<run_dir>
# If labels are too many: python tools/remerge_labels.py ./runs/<run_dir> 18

# Step 2: Classify GMM representatives only (~100 LLM calls)
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode

# Step 3: Propagate labels → full dataset (local, instant)
# Hard propagation (default):
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate
# Soft propagation (uses posterior probabilities, marks low-confidence as Unsuccessful):
tc-gmm --data massive_scenario --run_dir ./runs/<run_dir> --propagate --soft --confidence_threshold 0.4

# Step 4: Evaluate
tc-evaluate --data massive_scenario --run_dir ./runs/<run_dir>
```

**Cost**: ~300 LLM calls.  **Typical time**: 5–15 minutes.

---

### GMM Auto-Select k (via BIC)

```bash
tc-gmm --data massive_scenario --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200 --selection_criterion bic
```

This tries k ∈ [10, 200], picks the k with lowest BIC, and proceeds.

---

### Reusing Embeddings Between Modes

Both K-Medoids and GMM save `embeddings.npy` in the run directory. To reuse:

```bash
# 1. Run K-Medoids first
tc-kmedoids --data massive_scenario --kmedoids_k 100
# run_dir = ./runs/massive_scenario_small_20260313_100000

# 2. Now run GMM on the SAME run_dir — skips embedding recomputation
tc-gmm --data massive_scenario --gmm_k 100 --run_dir ./runs/massive_scenario_small_20260313_100000
```

The GMM step detects `embeddings.npy` and loads it directly (~0s instead of ~18s).

---

### Changing Datasets

Replace `massive_scenario` with any dataset name under `./datasets/`:

```bash
tc-gmm --data banking77 --gmm_k 50
tc-kmedoids --data clinc --kmedoids_k 100
tc-label-gen --data arxiv_fine --run_dir ./runs/<run_dir>
```

Available: `arxiv_fine`, `banking77`, `clinc`, `clinc_domain`, `few_event`, `few_nerd_nat`, `few_rel_nat`, `go_emotion`, `massive_intent`, `massive_scenario`, `mtop_domain`, `mtop_intent`, `reddit`, `stackexchange`.

---

### Resume After Interruption

All expensive steps have checkpoints:
- **Embeddings**: cached as `embeddings.npy` — pass `--run_dir` to reuse
- **K-Medoids metadata**: `kmedoids_metadata.json`
- **GMM metadata**: `gmm_metadata.json` + `gmm_probs.npy`
- **Classification**: checkpoint after every API call in medoid/representative mode

```bash
# Re-run the same command — it picks up from where it stopped
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --medoid_mode
tc-classify --data massive_scenario --run_dir ./runs/<run_dir> --representative_mode
```

---

### Quick Reference — Which Mode to Use

| Scenario | Recommended Mode | Key Command |
|----------|-----------------|-------------|
| Best quality, no budget constraint | Mode A (Original) | `tc-label-gen` + `tc-classify` |
| Fast experiments, K-Medoids | Mode B | `tc-kmedoids` → `tc-classify --medoid_mode` |
| Fast experiments, GMM (0% unlabelled) | Mode C | `tc-gmm` → `tc-classify --representative_mode` |
| Let GMM pick best k automatically | Mode C + auto-k | `tc-gmm --gmm_k 0 --gmm_k_min 10 --gmm_k_max 200` |
| Reuse embeddings from another run | Pass `--run_dir` | `tc-gmm --run_dir ./runs/<existing_dir>` |
| Switch provider to OpenRouter | Edit `.env` | `LLM_PROVIDER=openrouter` + `OPENAI_BASE_URL=https://openrouter.ai/api/v1` |
| Switch LLM model | Edit `.env` | `LLM_MODEL=gpt-4o-mini` or `gpt-3.5-turbo` etc. |
