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
LIB      := $(wildcard src/*.py src/utils/*.py src/pipeline/*.py)

INGEST_DONE := $(INTERIM)/.s1_ingest.done

.PHONY: help setup schema lint format typecheck test check ingest labels features train policy drift all clean

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

# Stage targets rebuild when code, config or raw inputs are newer than the
# marker. The marker is written only after every acceptance check passes.
ingest: $(INGEST_DONE)  ## stage 1
$(INGEST_DONE): $(RAW) $(CONFIG) $(LIB)
	$(RUN) python -m src.s1_ingest

labels:  ## stage 2
	@echo "stage 2 not built yet" && exit 1

features:  ## stage 3
	@echo "stage 3 not built yet" && exit 1

train:  ## stage 4
	@echo "stage 4 not built yet" && exit 1

policy:  ## stage 5
	@echo "stage 5 not built yet" && exit 1

drift:  ## stage 6
	@echo "stage 6 not built yet" && exit 1

all: ingest labels features train policy  ## full pipeline

clean:  ## remove interim and processed, keep raw
	rm -rf $(INTERIM) data/processed
	mkdir -p $(INTERIM) data/processed
