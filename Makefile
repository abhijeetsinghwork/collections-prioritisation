.DEFAULT_GOAL := help
SHELL := /bin/bash

# PySpark 3.5 needs Java 8/11/17. src/spark.py also falls back to config candidates.
JAVA17 := $(shell /usr/libexec/java_home -v 17 2>/dev/null)
ifneq ($(JAVA17),)
export JAVA_HOME ?= $(JAVA17)
endif

RUN      := uv run
INTERIM  := data/interim
RAW      := $(wildcard data/raw/sample_*.txt)
CONFIG   := config/config.yaml
COMMON   := src/spark.py src/utils/config.py src/utils/schema.py
INGEST_SRC := src/s1_ingest.py src/pipeline/ingest.py src/pipeline/checks.py $(COMMON)
FEATURES_EXTRA := src/utils/leakage.py
LABELS_SRC := src/s2_labels.py src/pipeline/labels.py src/pipeline/label_checks.py $(COMMON)
FEATURES_SRC := src/s3_features.py src/pipeline/features.py src/pipeline/feature_audit.py \
	src/pipeline/macro.py src/pipeline/labels.py $(FEATURES_EXTRA) $(COMMON)

INGEST_DONE := $(INTERIM)/.s1_ingest.done
LABELS_DONE := data/processed/.s2_labels.done
FEATURES_DONE := data/processed/.s3_features.done
TRAIN_DONE := data/processed/.s4_fit.done
MODELS_SRC := src/s4_models.py src/pipeline/modeling.py src/pipeline/evaluation.py \
	src/utils/plots.py src/utils/leakage.py src/pipeline/features.py $(COMMON)
POLICY_DONE := data/processed/.s5_policy.done
# Test scores are written by the frozen models' (logged) test evaluation. Make
# re-runs that evaluation only when the scores are older than the freeze.
EQ := =
TEST_SCORES := data/processed/scores/split$(EQ)test/scores.parquet
POLICY_SRC := src/s5_policy.py src/pipeline/policy.py src/utils/plots.py $(COMMON)
MACRO_DONE := data/raw/macro/MORTGAGE30US.csv
STAMPS := data/.stamps

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

# Stage targets rebuild when code, raw inputs or the stage's own config section
# are newer than the marker. The marker is written only after every acceptance
# check passes. Stamps are refreshed on every run but only touched on change.
.PHONY: FORCE
$(STAMPS)/%.json: FORCE
	@$(RUN) python -m src.utils.config_stamp $* $@

ingest: $(INGEST_DONE)  ## stage 1
$(INGEST_DONE): $(RAW) $(STAMPS)/ingest.json $(INGEST_SRC)
	$(RUN) python -m src.s1_ingest

labels: $(LABELS_DONE)  ## stage 2
$(LABELS_DONE): $(INGEST_DONE) $(STAMPS)/labels.json $(LABELS_SRC)
	$(RUN) python -m src.s2_labels

macro: $(MACRO_DONE)  ## download FRED macro series (no API key needed)
$(MACRO_DONE):
	$(RUN) python -m src.utils.fetch_macro

features: $(FEATURES_DONE)  ## stage 3
$(FEATURES_DONE): $(LABELS_DONE) $(MACRO_DONE) $(STAMPS)/features.json $(FEATURES_SRC)
	$(RUN) python -m src.s3_features

train: $(TRAIN_DONE)  ## stage 4: fit on train, select + calibrate on validation, freeze
$(TRAIN_DONE): $(FEATURES_DONE) $(STAMPS)/models.json $(MODELS_SRC)
	$(RUN) python -m src.s4_models

evaluate-test:  ## stage 4: score the FROZEN models on test (appends to the test log)
	$(RUN) python -m src.s4_models --evaluate-test

$(TEST_SCORES): $(TRAIN_DONE)
	$(RUN) python -m src.s4_models --evaluate-test

policy: $(POLICY_DONE)  ## stage 5: capture of each contact policy on the test window
$(POLICY_DONE): $(TEST_SCORES) $(STAMPS)/policy.json $(POLICY_SRC)
	$(RUN) python -m src.s5_policy

drift:  ## stage 6
	@echo "stage 6 not built yet" && exit 1

all: ingest labels features train policy  ## full pipeline

clean:  ## remove interim and processed, keep raw
	rm -rf $(INTERIM) data/processed $(STAMPS)
	mkdir -p $(INTERIM) data/processed
