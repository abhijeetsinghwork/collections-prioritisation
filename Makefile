.DEFAULT_GOAL := help
SHELL := /bin/bash

# PySpark 3.5 needs Java 8/11/17. src/spark.py also falls back to config candidates.
JAVA17 := $(shell /usr/libexec/java_home -v 17 2>/dev/null)
ifneq ($(JAVA17),)
export JAVA_HOME ?= $(JAVA17)
endif

RUN      := uv run
INTERIM  := data/interim
STAMPS   := data/.stamps

INGEST_DONE   := $(INTERIM)/.s1_ingest.done
LABELS_DONE   := data/processed/.s2_labels.done
FEATURES_DONE := data/processed/.s3_features.done
TRAIN_DONE    := data/processed/.s4_fit.done
POLICY_DONE   := data/processed/.s5_policy.done
MACRO_DONE    := data/raw/macro/MORTGAGE30US.csv
# Test scores are written by the frozen models' (logged) test evaluation.
EQ := =
TEST_SCORES := data/processed/scores/split$(EQ)test/scores.parquet

.PHONY: help setup schema macro lint format typecheck test check ingest labels features train evaluate-test policy drift all clean

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## create env and install pinned deps, install git hooks
	uv sync --locked
	$(RUN) pre-commit install

schema:  ## regenerate src/utils/schema.py from the official layout workbook
	$(RUN) python -m src.utils.gen_schema

lint:  ## ruff check + format --check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## apply ruff fixes and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:  ## mypy src
	$(RUN) mypy src

test:  ## pytest on synthetic fixtures
	$(RUN) pytest

check: lint typecheck test  ## lint + typecheck + test

# Each stage depends on one content stamp (src/utils/config_stamp.py): its
# config sections, the bytes of its sources, its raw inputs, and the upstream
# stage's stamp. Stamps are recomputed on every run but only touched when that
# content changes, so a checkout or merge does not trigger a rebuild. Upstream
# markers are order-only (after the |): they must exist, their mtime is ignored.
# A marker is written only after every acceptance check passes.
.PHONY: FORCE
$(STAMPS)/%.json: FORCE
	@$(RUN) python -m src.utils.config_stamp $* $@

ingest: $(INGEST_DONE)  ## stage 1
$(INGEST_DONE): $(STAMPS)/ingest.json
	$(RUN) python -m src.s1_ingest

labels: $(LABELS_DONE)  ## stage 2
$(LABELS_DONE): $(STAMPS)/labels.json | $(INGEST_DONE)
	$(RUN) python -m src.s2_labels

macro: $(MACRO_DONE)  ## download FRED macro series (no API key needed)
$(MACRO_DONE):
	$(RUN) python -m src.utils.fetch_macro

features: $(FEATURES_DONE)  ## stage 3
$(FEATURES_DONE): $(STAMPS)/features.json | $(LABELS_DONE) $(MACRO_DONE)
	$(RUN) python -m src.s3_features

train: $(TRAIN_DONE)  ## stage 4: fit on train, select + calibrate on validation, freeze
$(TRAIN_DONE): $(STAMPS)/models.json | $(FEATURES_DONE)
	$(RUN) python -m src.s4_models

evaluate-test:  ## stage 4: score the FROZEN models on test (appends to the test log)
	$(RUN) python -m src.s4_models --evaluate-test

# Re-scored on test only when the frozen models change (each run is logged).
$(TEST_SCORES): $(STAMPS)/models.json | $(TRAIN_DONE)
	$(RUN) python -m src.s4_models --evaluate-test

policy: $(POLICY_DONE)  ## stage 5: capture of each contact policy on the test window
$(POLICY_DONE): $(STAMPS)/policy.json $(TEST_SCORES) | $(TRAIN_DONE)
	$(RUN) python -m src.s5_policy

drift:  ## stage 6
	@echo "stage 6 not built yet" && exit 1

all: ingest labels features train policy  ## full pipeline

clean:  ## remove interim and processed, keep raw
	rm -rf $(INTERIM) data/processed $(STAMPS)
	mkdir -p $(INTERIM) data/processed
