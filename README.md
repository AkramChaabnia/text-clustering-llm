# Text Clustering as Classification with LLMs

> PPD reproduction — M2 MLSD, Université Paris Cité  
> Based on: [Text Clustering as Classification with LLMs](https://arxiv.org/abs/2410.00927) (Chen Huang, Guoxiu He, 2024)  
> Original code: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)

For a full description of the pipeline, the fixes applied to the original code, and the execution log, see [FINDINGS.md](./FINDINGS.md).

---

## Output Structure

All outputs go under `./runs/`:

```
runs/
  chosen_labels.json                         ← Step 0: seed labels (shared, reused)
  massive_scenario_small_20260220_143012/    ← one folder per Step 1 run
    labels_true.json                         ← ground-truth label list
    labels_proposed.json                     ← LLM proposals before merge
    labels_merged.json                       ← merged labels → input to Step 2
    classifications.json                     ← Step 2 output → input to Step 3
    checkpoint.json                          ← Step 2 live checkpoint (auto-deleted on success)
    results.json                             ← Step 3 metrics (ACC, NMI, ARI + metadata)
```

The folder name encodes `<dataset>_<split>_<YYYYMMDD_HHMMSS>`, so each run is isolated and timestamped.

---

## Models & Datasets

### Models

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| `gpt-3.5-turbo-0125` | OpenAI (paper) | Reference | Main model in paper experiments |
| `google/gemini-2.0-flash-001` | OpenRouter (paid) | ✅ **PRIMARY** | 6/6 probe + merge test 167→28 labels. ~$0.92 for full 5-dataset baseline. |
| `arcee-ai/trinity-large-preview:free` | OpenRouter (free) | ⚠️ Merge fails | 6/6 probe but stalls at 144 labels — cannot consolidate at scale |
| `openai/gpt-4o-mini` | OpenRouter (paid) | ⚠️ Merge weak | 6/6 probe but merge test: 167→105 (poor consolidation) |
| `meta-llama/llama-3.3-70b-instruct:free` | OpenRouter (free) | ⏳ Pending | Venice upstream congestion — retry off-peak, then confirm merge |
| `mistralai/mistral-small-3.1-24b-instruct:free` | OpenRouter (free) | ⏳ Pending | Same Venice block |
| `google/gemma-3-27b-it:free` | OpenRouter (free) | ⏳ Pending | Same Venice block |
| `nousresearch/hermes-3-llama-3.1-405b:free` | OpenRouter (free) | ⏳ Pending | Same Venice block |

Run `tools/probe_models.py` to check eligibility before switching. Passing the 6-test probe is
necessary but not sufficient — also verify the merge step consolidates to ≈ true class count.

See `FINDINGS.md §5–§6` for full selection rationale and probe log.

> **JSON mode**: The original code forces `response_format={"type":"json_object"}` on every call.
> Most models (including gemini-2.0-flash) don't support this flag. We keep it off by default
> (`LLM_FORCE_JSON_MODE=false`) and strip markdown fences from responses instead.

### Datasets used in the paper

| Dataset | Domain | Classes | Samples (small) |
|---------|--------|---------|-----------------|
| `massive_scenario` | Voice assistant scenarios | 18 | 2,974 |
| `massive_intent` | Voice assistant intents | 59 | 2,974 |
| `go_emotion` | Emotion detection | 27 | 5,940 |
| `arxiv_fine` | Academic topics | 93 | 3,674 |
| `mtop_intent` | Multi-domain intent | 102 | 4,386 |

Download: [Google Drive (ClusterLLM)](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing) — unzip into `./dataset/`

### Paper baseline (`gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

| Dataset | ACC | NMI | ARI |
|---------|-----|-----|-----|
| `massive_scenario` | 71.75 | 78.00 | 56.86 |
| `massive_intent` | 64.12 | 65.44 | 48.92 |
| `go_emotion` | 31.66 | 27.39 | 13.50 |
| `arxiv_fine` | 38.78 | 57.43 | 20.55 |
| `mtop_intent` | 72.18 | 78.78 | 71.93 |

---

## Setup

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai) API key (free tier works)

### Installation

```bash
git clone https://github.com/AkramChaabnia/text-clustering-llm.git
cd text-clustering-llm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
# Edit .env: set OPENAI_API_KEY to your OpenRouter key
python -m text_clustering.client   # smoke test: verifies the API key and model
```

---

## Running the Pipeline

### Using Make

```bash
# Step 0 — seed labels (run once)
make run-step0

# Step 1 — generate labels
#   Prints the run directory path — you'll need it for steps 2 and 3.
make run-step1 data=massive_scenario

# Step 2 — classify (runs in background, resumes if interrupted)
make run-step2 data=massive_scenario run=./runs/massive_scenario_small_20260220_143012

# Step 3 — evaluate
make run-step3 data=massive_scenario run=./runs/massive_scenario_small_20260220_143012
```

### Using entry points directly

```bash
# Step 0 — seed labels
tc-seed-labels

# Step 1 — label generation  (prints the run_dir at startup)
tc-label-gen --data massive_scenario

# Step 2 — classification  (--run_dir = folder created by Step 1)
tc-classify --data massive_scenario \
            --run_dir ./runs/massive_scenario_small_20260220_143012

# Step 3 — evaluation
tc-evaluate --data massive_scenario \
            --run_dir ./runs/massive_scenario_small_20260220_143012
```

### Checkpoint & resume (Step 2)

Step 2 makes one API call per text (~3,000 calls for most datasets). Progress is saved to `checkpoint.json` every 200 samples. If the run is interrupted, re-run the same command — the script picks up from where it left off and prints:

```
[checkpoint] Resuming from sample 1400 / loaded from ./runs/.../checkpoint.json
```

The checkpoint file is removed once the run finishes.

### Test mode

Pass `--print_details True --test_num N` to limit LLM calls to N and print prompts/responses:

```bash
tc-label-gen  --data massive_scenario --print_details True --test_num 2
tc-classify   --data massive_scenario --run_dir ./runs/... --print_details True --test_num 3
```

---

## Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenRouter (or OpenAI) key |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | API endpoint |
| `LLM_MODEL` | `google/gemini-2.0-flash-001` | Model ID |
| `LLM_TEMPERATURE` | `0` | Temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per call (merge step needs ≥2300) |
| `LLM_FORCE_JSON_MODE` | `false` | Set `true` only for models that support `response_format=json_object` |
| `LLM_REQUEST_DELAY` | `2` | Seconds to wait between LLM calls (rate limiting) |
| `OR_SITE_URL` | — | Optional: sent as `HTTP-Referer` header to OpenRouter |
| `OR_APP_NAME` | `text-clustering-llm` | Optional: sent as `X-Title` header |

See `.env.example` for the full template.

---

## Repository Structure

```
text-clustering-llm/
├── text_clustering/               # Python package (installed via `uv pip install -e .`)
│   ├── client.py                  # OpenRouter/OpenAI client factory (reads .env)
│   ├── config.py                  # Centralised env-var config
│   ├── llm.py                     # Shared LLM helpers: ini_client, chat, retry logic
│   ├── data.py                    # Dataset loading helpers
│   ├── prompts.py                 # Prompt construction functions (no I/O)
│   └── pipeline/
│       ├── seed_labels.py         # Step 0: select 20% seed labels → runs/chosen_labels.json
│       ├── label_generation.py    # Step 1: LLM proposes + merges labels → runs/<run_dir>/
│       ├── classification.py      # Step 2: LLM classifies texts + checkpoint/resume
│       └── evaluation.py          # Step 3: ACC/NMI/ARI + saves results.json
├── paper/                         # Thin shims for backward compat (python label_generation.py …)
├── tools/
│   └── probe_models.py            # Dev tool: 6-test model compatibility probe
├── dataset/                       # Datasets (not in git — download separately)
├── runs/                          # All outputs (not in git)
├── logs/                          # Logs from background runs (not in git)
├── Makefile                       # Convenience targets: setup, lint, run-step*, release
├── pyproject.toml                 # Project metadata + dependencies + entry points
├── uv.lock                        # Pinned lock file
├── .env.example                   # Env variable template (no secrets)
├── .cz.yaml                       # Commitizen config
└── FINDINGS.md                    # Research log: decisions, fixes, results
```

---

## Development

Branching: `main` ← `develop` ← `feature/<desc>` / `fix/<desc>` / `docs/<desc>`  
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

```bash
make setup            # venv + install + git hooks
make lint             # ruff check
cz commit             # conventional commit prompt
make branch name=my-feature type=feature
make release          # bump version, merge develop→main, push tags
```

---

## Citation

```bibtex
@inproceedings{huang2024text,
  title={Text Clustering as Classification with LLMs},
  author={Huang, Chen and He, Guoxiu},
  year={2024},
  url={https://arxiv.org/abs/2410.00927}
}
```
