## v1.3.0 (2026-02-21)

### Feat

- **model**: merge fix/model-gemini-flash into develop (gemini + target_k fix + Run 02/03 docs)
- **tools**: add tc-preflight pre-run check command
- **model**: switch primary model to google/gemini-2.0-flash-001

### Fix

- **prompts**: restore paper-faithful merge (no default target_k)
- merge run-02-prep into develop (token budget, target-k prompt, merge warning, per-step logs)
- **logging**: use separate log file per pipeline step
- **pipeline**: warn when merged label count exceeds 2x true class count
- **prompts**: add target_k to merge prompt to guide consolidation count
- **llm**: allow per-call max_tokens override; use 4096 for merge step

## v1.2.0 (2026-02-20)

### Feat

- **logging**: merge logging into develop
- **logging**: replace print calls with structured logging to run.log
- **runs**: merge run management into develop
- **runs**: add timestamped run dirs, checkpoint/resume, and results.json
- **package**: merge package restructure into develop
- **package**: restructure into text_clustering package with entry points

## v1.1.0 (2026-02-20)

### Feat

- **openrouter**: add probe_models CLI for model eligibility testing
- **openrouter**: add OpenRouter adapter and env template

### Fix

- **scripts**: rewrite given_label_classification to work with OpenRouter
- **scripts**: rewrite label_generation to work with OpenRouter
- **scripts**: fix dataset path resolution in select_part_labels

## v1.0.0 (2026-02-19)
