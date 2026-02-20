.DEFAULT_GOAL := help
.PHONY: help setup lint branch release run-step0 run-step1 run-step2 run-step3

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  setup                        create venv, install deps and git hooks"
	@echo "  lint                         run ruff on all Python files"
	@echo "  branch name=<n> type=<t>     create and push a new branch off develop"
	@echo "  release                      bump version, merge develop->main, push tags"
	@echo ""
	@echo "  run-step0                    select seed labels (run once)"
	@echo "  run-step1 data=<dataset>     generate candidate labels"
	@echo "  run-step2 data=<dataset>     classify (runs in background)"
	@echo "  run-step3 data=<dataset>     evaluate results"
	@echo ""

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	.venv/bin/pre-commit install

lint:
	.venv/bin/ruff check .

# usage: make branch name=openrouter-retry type=fix
branch:
ifndef name
	$(error name is required, e.g. make branch name=my-feature type=feature)
endif
ifndef type
	$(error type is required: feature | fix | docs)
endif
	git checkout develop
	git pull origin develop
	git checkout -b $(type)/$(name)
	git push -u origin $(type)/$(name)

release:
	@[ "$$(git branch --show-current)" = "develop" ] || \
		(echo "error: must be on develop" && exit 1)
	.venv/bin/cz bump
	$(eval NEW_TAG := $(shell git describe --tags --abbrev=0))
	git checkout main
	git merge --no-ff develop -m "release: merge develop into main for $(NEW_TAG)"
	git checkout develop
	git push origin main
	git push origin develop
	git push origin --tags

run-step0:
	.venv/bin/python select_part_labels.py

# usage: make run-step1 data=massive_scenario
run-step1:
ifndef data
	$(error data is required, e.g. make run-step1 data=massive_scenario)
endif
	mkdir -p logs
	.venv/bin/python label_generation.py --data $(data) 2>&1 | tee logs/$(data)_label_gen.log

# usage: make run-step2 data=massive_scenario  (runs in background)
run-step2:
ifndef data
	$(error data is required, e.g. make run-step2 data=massive_scenario)
endif
	mkdir -p logs
	nohup .venv/bin/python given_label_classification.py --data $(data) \
		> logs/$(data)_classification.log 2>&1 &
	@echo "running in background — tail logs/$(data)_classification.log"

# usage: make run-step3 data=massive_scenario
run-step3:
ifndef data
	$(error data is required, e.g. make run-step3 data=massive_scenario)
endif
	.venv/bin/python evaluate.py \
		--data $(data) \
		--predict_file_path ./generated_labels/ \
		--predict_file $(data)_small_find_labels.json
