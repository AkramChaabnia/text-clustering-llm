# Text Clustering as Classification with LLMs

> **PPD reproduction & experiments** — M2 MLSD, Université Paris Cité  
> Based on the paper: [Text Clustering as Classification with LLMs](https://arxiv.org/abs/2410.00927) (Chen Huang, Guoxiu He, 2024)  
> Original repository: [ECNU-Text-Computing/Text-Clustering-via-LLM](https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM)

---

## Models & Datasets — Reference Table

### Models used in the paper

| Model (paper) | Type | Notes |
|---------------|------|-------|
| `gpt-3.5-turbo-0125` | OpenAI | Main model used in all experiments |
| `gpt-4` | OpenAI | Used in ablation/upper bound |

### Free OpenRouter alternatives for reproduction

| Model (OpenRouter ID) | Size | Status | Notes |
|-----------------------|------|--------|-------|
| `arcee-ai/trinity-large-preview:free` | ~70B | ✅ Confirmed | Passed all 6 probes — use for baseline runs |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | ⏳ Pending | Venice upstream congestion — retry off-peak |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | ⏳ Pending | Same Venice block |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | ⏳ Pending | Same Venice block |
| `google/gemma-3-27b-it:free` | 27B | ⏳ Pending | Same Venice block |

See `FINDINGS.md §6` for the full probe log and exclusion reasons.

> **Note on JSON mode**: The original code forces `response_format={"type":"json_object"}` on every call.
> Most free models don't reliably support this — some return HTTP 400, others return empty bodies.
> We default `LLM_FORCE_JSON_MODE=false` and strip markdown fences from responses instead.

### Datasets used in the paper

| Dataset | Domain | Classes | Samples (small) |
|---------|--------|---------|-----------------|
| `massive_scenario` | Voice assistant scenarios | 18 | 2,974 |
| `massive_intent` | Voice assistant intents | 59 | 2,974 |
| `go_emotion` | Emotion detection | 27 | 5,940 |
| `arxiv_fine` | Academic topics | 93 | 3,674 |
| `mtop_intent` | Multi-domain intent | 102 | 4,386 |

Download: [Google Drive (ClusterLLM)](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing) — unzip into `./dataset/`

### Paper baseline results (`gpt-3.5-turbo-0125`, 20% seed labels, batch=15)

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
python openrouter_adapter.py   # smoke test
```

---

## Running the Pipeline

```bash
# Step 1 — seed labels (20% of ground truth)
python select_part_labels.py

# Step 2 — label generation
python label_generation.py --data massive_intent

# Step 3 — classification
python given_label_classification.py --data massive_intent

# Step 4 — evaluate
python evaluate.py \
  --data massive_intent \
  --predict_file_path ./generated_labels/ \
  --predict_file massive_intent_small_find_labels.json

# Or run all datasets at once:
bash run.sh
```

---

## Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenRouter (or OpenAI) key |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | API endpoint |
| `LLM_MODEL` | `arcee-ai/trinity-large-preview:free` | Model ID |
| `LLM_TEMPERATURE` | `0` | Temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `512` | Max tokens per call |
| `LLM_FORCE_JSON_MODE` | `false` | Set `true` only for models that support `response_format=json_object` |
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `openai` |
| `OR_SITE_URL` | — | Optional: sent as `HTTP-Referer` header to OpenRouter |
| `OR_APP_NAME` | `text-clustering-llm` | Optional: sent as `X-Title` header |

See `.env.example` for the full template.

---

## Development

Branching: `main` ← `develop` ← `feature/<desc>` / `fix/<desc>` / `docs/<desc>`  
Commits follow [Conventional Commits](https://www.conventionalcommits.org/) — use `cz commit` or write them manually.

```bash
ruff check .          # lint
cz commit             # interactive conventional commit
```

---

## Repository Structure

```
├── label_generation.py           # Step 1: LLM generates candidate labels
├── given_label_classification.py # Step 2: LLM assigns labels to texts
├── evaluate.py                   # Step 3: ACC / NMI / ARI scoring
├── select_part_labels.py         # Utility: sample 20% of ground-truth labels
├── openrouter_adapter.py         # Thin adapter: reads .env, returns openai.OpenAI client
├── run.sh                        # End-to-end script for all datasets
├── pyproject.toml                # Project metadata + dependencies
├── uv.lock                       # Pinned lock file
├── .env.example                  # Env variable template (no secrets)
├── .cz.yaml                      # Commitizen config
└── FINDINGS.md                   # Research log: decisions, fixes, results
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
