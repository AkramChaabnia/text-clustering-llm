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

| Model (OpenRouter ID) | Size | Quality tier | Recommended for |
|-----------------------|------|-------------|-----------------|
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | ⭐⭐⭐ Best free | Primary baseline |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | ⭐⭐ Good | Secondary check |
| `google/gemma-3-27b-it:free` | 27B | ⭐⭐ Good | Cross-validation |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | ⭐⭐⭐ Largest free | Upper-bound test |
| `qwen/qwen3-4b:free` | 4B | ⭐ Cheap | Smoke tests only |

> **Note on JSON mode**: The original code uses `response_format={"type":"json_object"}`.
> Not all free models support it — Llama 3.3 70B does.

### Datasets used in the paper

| Dataset | Domain | Classes | Split used | Priority |
|---------|--------|---------|-----------|----------|
| `massive_intent` | Intent detection | 60 | small | Start here |
| `massive_scenario` | Scenario detection | 18 | small | Second |
| `go_emotion` | Emotion | 27 | small | Third |
| `mtop_intent` | Intent (multi-domain) | 104 | small | 4th |
| `arxiv_fine` | Topic classification | 20+ | small | 5th |

Download: [Google Drive (ClusterLLM)](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing) — unzip into `./dataset/`

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
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model |
| `LLM_TEMPERATURE` | `0` | Temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `512` | Max tokens per call |

See `.env.example` for the full template.

---

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branching strategy and commit conventions.

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
├── openrouter_adapter.py         # OpenRouter/env adapter
├── run.sh                        # End-to-end script for all datasets
├── pyproject.toml                # Project metadata + dependencies
├── uv.lock                       # Pinned lock file
├── .env.example                  # Env variable template (no secrets)
├── .cz.yaml                      # Commitizen config
└── CONTRIBUTING.md               # Branching + commit guide
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
