# Collections Prioritisation Engine

> Work in progress. The full README (capture curve, headline result,
> limitations) is written once the pipeline is complete.

Given every mortgage that is already delinquent in month *T*, and a
collections team that can contact only a fixed share of them, rank the queue
to preserve the most balance that would otherwise deteriorate. The score is
`P(rolls deeper within 3 months) × exposure`; this is a ranking problem under
a capacity constraint, not a default model.

## Data

Freddie Mac Single-Family Loan-Level Dataset (sample files). The raw data is
**not** in this repository: Freddie Mac's terms permit publishing code and
analytical results, not redistributing the dataset. See
[docs/download.md](docs/download.md) to obtain it.

## Running

```bash
brew install openjdk@17   # PySpark 3.5 needs Java 8/11/17
make setup                # uv env from uv.lock + git hooks
make check                # lint, type-check, tests (no data needed)
make ingest               # stage 1, needs data/raw/
```

Decisions and their reasons are recorded in
[docs/methodology.md](docs/methodology.md).
