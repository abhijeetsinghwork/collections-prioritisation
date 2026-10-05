# Collections Prioritisation Engine — Build Spec

> Reference document for Claude Code. Build stage by stage. Do not skip the
> acceptance checks — several of them exist to catch leakage, which is the main
> failure mode of this project.
>
> **No expected values appear in this document, deliberately.** Every check is
> structural: a relationship that must hold regardless of what the numbers turn
> out to be. Do not introduce target metrics, do not tune toward a figure, and
> do not describe any result as good or bad relative to an assumed benchmark.
> Whatever comes out of the data is the result.

---

## 0. What this project is

Given every loan account that is **already delinquent** as of month *T*, and a
collections team with capacity to contact only *C* of them this cycle, produce a
ranked contact list that preserves the most money.

This is **not** a default prediction model. It is a ranking problem under a
capacity constraint. The deliverable is a policy comparison, not an AUC.

**Core score:**

```
score(account, T) = P(rolls deeper within 3 months) × exposure_at_T
```

An account that will cure on its own is worth zero calls regardless of balance.
That reframe is the whole point of the project.

**Final headline metric:** at *C* = 20% of the delinquent queue, what fraction of
the balance that actually deteriorated did we capture, versus the baseline
policies?

---

## 0b. Stage 0 — Bootstrap the repo

Do this before any code.

1. `git init` in the project folder.
2. Write `.gitignore` **first**, before any data lands. It must contain at
   minimum: `data/`, `mlruns/`, `.venv/`, `__pycache__/`, `.DS_Store`,
   `*.parquet`, `.env`, `outputs/figures/*.png` (keep the final README figures,
   ignore scratch).
3. Create the GitHub repo and push. Use the `gh` CLI if it is authenticated
   (`gh auth status` to check, `gh auth login` if not):
   ```bash
   gh repo create collections-prioritisation --public --source=. --remote=origin
   git add -A && git commit -m "chore: project skeleton and build spec"
   git push -u origin main
   ```
   If `gh` is unavailable, create the repo in the browser and add the remote
   manually.
4. Commit the empty directory structure with `.gitkeep` files so the layout is
   visible from the first commit.
5. **Verify `data/` is ignored before the dataset is placed in it.** Run
   `git check-ignore -v data/raw/test.txt` and confirm it reports a match.
   Freddie Mac's terms permit publishing code and analytical results but not
   redistributing the raw dataset, and a committed data file is hard to purge
   from git history later.

Commit at the end of every stage, not just at the end of the project. The commit
history is part of what a reviewer looks at.

---

## 1. Environment

- Python 3.11
- `uv` for env management
- PySpark 3.5 in **local mode** (requires a JDK — install `openjdk@17` if absent)
- Everything runs CPU-only on a Mac. No GPU, no cloud, no paid services.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install pyspark==3.5.* pandas polars pyarrow lightgbm scikit-learn \
               mlflow shap matplotlib seaborn optbinning pyyaml tqdm
```

**Spark local config** (put in `src/spark.py`, reuse everywhere):

```python
SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "64") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
```

Tune `driver.memory` to roughly half the machine's RAM. If the machine has 8GB,
set 4g and reduce the number of vintages loaded.

---

## 2. Repo structure

```
collections-prioritisation/
├── README.md
├── config/
│   └── config.yaml              # all tunable parameters, no magic numbers in code
├── data/                        # GITIGNORED — never commit raw data
│   ├── raw/                     # downloaded Freddie Mac files
│   ├── interim/                 # parquet panel
│   └── processed/               # modelling frames
├── docs/
│   ├── download.md              # how a reader gets the data themselves
│   └── methodology.md           # written writeup
├── notebooks/                   # exploration only, nothing load-bearing
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── spark.py
│   ├── s1_ingest.py
│   ├── s2_labels.py
│   ├── s3_features.py
│   ├── s4_models.py
│   ├── s5_policy.py
│   └── utils/
│       ├── schema.py            # Freddie Mac column definitions
│       ├── leakage.py           # blocklist + automated guard
│       └── plots.py
├── tests/
└── Makefile
```

**`.gitignore` must include `data/`.** Freddie Mac's terms permit publishing
analytical results and code but not redistributing the raw dataset.

---

## 3. Data acquisition (manual step — cannot be automated)

Full instructions live in `docs/download.md`. Summary:

1. Register at Freddie Mac Clarity Data Intelligence (free) and go to the SFLLD
   Data Download page.
2. Download the **sample files**, not the full dataset. The sample is a random
   subset per vintage year and is laptop-sized.
3. Suggested vintages: **2000–2019 and 2022–2024**. Deliberately skip 2020–2021
   at training time (see COVID handling below) but download them anyway — they
   are needed for the drift case study.
4. Also download the **File Layout** workbook and the **General User Guide** from
   the same page. These are the authoritative schema reference.
5. Unzip into `data/raw/`.

Each vintage arrives as `sample_YYYY.zip` containing two pipe-delimited files
with **no header row**:
- `sample_orig_YYYY.txt` — one row per loan, static attributes
- `sample_perf_YYYY.txt` — one row per loan-month, performance

The sample is a simple random sample of 50,000 loans per full vintage year.

**Instruction to Claude Code:** do not trust the column list in section 4 below.
Parse the official layout document in `data/raw/` and generate
`src/utils/schema.py` from it. Column counts have changed across layout
revisions and the authoritative source is that document.

---

## 4. Schema reference (verify against official layout)

### Origination file — fields to retain

| Field | Use |
|---|---|
| Loan Sequence Number | join key |
| Credit Score | feature |
| First Time Homebuyer Flag | feature |
| MSA | feature (high cardinality — target encode or drop) |
| Number of Units | feature |
| Occupancy Status | feature |
| Original CLTV | feature |
| Original DTI | feature |
| Original UPB | feature + denominator |
| Original LTV | feature |
| Original Interest Rate | feature |
| Channel | feature |
| Property State | feature + macro join key |
| Property Type | feature |
| Loan Purpose | feature |
| Original Loan Term | feature |
| Number of Borrowers | feature |
| First Payment Date | derive loan age sanity check |

Known sentinel values: `9999` for missing credit score, `999` for missing DTI,
`999` for missing LTV/CLTV. Convert these to nulls explicitly — LightGBM handles
nulls natively, but leaving 9999 in will silently corrupt splits.

### Monthly performance file — fields to retain

| Field | Use |
|---|---|
| Loan Sequence Number | join key |
| Monthly Reporting Period (YYYYMM) | time index |
| Current Actual UPB | exposure |
| **Current Loan Delinquency Status** | the spine — target and population |
| Loan Age | feature |
| Remaining Months to Legal Maturity | feature |
| Modification Flag | feature + treatment proxy |
| Zero Balance Code | terminal state detection |
| Zero Balance Effective Date | terminal state timing |
| Current Interest Rate | feature |
| Current Deferred UPB | feature |
| Step Modification Flag | feature |
| Deferred Payment Plan | feature |
| Estimated LTV | feature |
| Delinquency Due to Disaster | feature + COVID/disaster handling |
| Borrower Assistance Status Code | feature |

### Delinquency status encoding

Values are zero-padded strings, not integers:

| Raw | Meaning | Bucket |
|---|---|---|
| `00` | Current | 0 |
| `01` | 30–59 days | 1 |
| `02` | 60–89 days | 2 |
| `03` | 90–119 days | 3 |
| `04`…`NN` | increasing | 4+ |
| `RA` | REO acquisition | terminal |
| `XX` | Unknown | null — do not coerce to 0 |

Map to an ordinal integer `dq_bucket`. Treat `XX` as null and exclude those
loan-months from both the population and the label window.

### Zero Balance Codes (terminal states)

| Code | Meaning | Treat as |
|---|---|---|
| `01` | Prepaid / matured | cure (good) |
| `02` | Third party sale | roll (bad) |
| `03` | Short sale | roll (bad) |
| `06` | Repurchase | exclude |
| `09` | REO disposition | roll (bad) |
| `15` | Note sale | roll (bad) |
| `96` | Removal (non-credit) | exclude |

---

## 5. Stage 1 — Ingest (`s1_ingest.py`)

**Goal:** turn raw pipe-delimited text into a clean partitioned Parquet panel.

Steps:
1. Read origination files with explicit schema (never `inferSchema` — it will
   type `Loan Sequence Number` inconsistently across vintages).
2. Read performance files the same way.
3. Clean sentinels to nulls.
4. Parse `Monthly Reporting Period` into a proper date (first of month).
5. Map delinquency status to `dq_bucket`.
6. Join origination onto performance on loan sequence number (broadcast the
   origination side — it is small relative to performance).
7. Write Parquet partitioned by `reporting_year`.

```python
df.write.mode("overwrite") \
  .partitionBy("reporting_year") \
  .parquet("data/interim/panel")
```

**Acceptance checks:**
- Row count of the panel equals the raw performance row count minus excluded
  `XX` rows. Print both.
- `loan_sequence_number` is unique within each `(loan, reporting_period)` pair.
  Assert zero duplicates.
- Distribution of `dq_bucket` printed as a table. Current should dominate
  heavily (>95%). If it doesn't, the status mapping is wrong.
- Spot-check 5 random loans: print their full monthly sequence and confirm the
  delinquency path is monotonic-ish and the dates are contiguous.

---

## 6. Stage 2 — Population and labels (`s2_labels.py`)

### Population

Loan-months where `dq_bucket IN (1, 2, 3)`.

Rationale to record in `docs/methodology.md`: current accounts are not contacted
by collections, and 120+ is past the point where routine telephony is the right
lever.

### Label

```
rolls_deeper = 1 if, within the next 3 reporting months, the account either:
                 - reaches a dq_bucket strictly greater than its bucket at T, or
                 - hits a terminal Zero Balance Code classified as "roll"
               else 0
```

Implementation: a window function over `(loan_sequence_number ORDER BY
reporting_period ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING)`, taking `max(dq_bucket)`
and checking terminal codes in the same window.

### Censoring

Drop any loan-month where fewer than 3 future months exist in the dataset. Do
**not** impute. Compute the global `max(reporting_period)` and drop the final
three months outright, plus any loan whose record ends early without a terminal
code in the window.

### Splits

Temporal, and grouped by loan:

```yaml
train:      reporting_period <= 2015-12
validation: 2016-01 .. 2017-12
test:       2018-01 .. 2019-12
drift_study: 2020-01 .. 2021-12   # held out entirely, not for training or test
late_holdout: 2022-01 .. 2024-12  # optional second test window
```

Then: **remove from validation and test any loan that appears in train.** A
temporal split alone is not enough, because a long-lived loan straddles the
cutoff and the model will have memorised its idiosyncrasies.

**Acceptance checks:**
- `assert len(set(train.loan_ids) & set(test.loan_ids)) == 0`
- Print base rate of `rolls_deeper` per split and per `dq_bucket`. Do not assume
  a value. Check the **structure**: the base rate should be non-degenerate
  (neither near 0 nor near 1), and it should be monotonically higher in deeper
  buckets. A flat or inverted pattern across buckets means the label logic is
  wrong.
- Print row counts per split. Confirm each is large enough to train on.
- Print `max(reporting_period)` per split and confirm no overlap.
- Manually trace 3 accounts labelled 1 and 3 labelled 0 through their monthly
  sequences and confirm by eye that the label matches what happened.

---

## 7. Stage 3 — Features (`s3_features.py`)

Target ~80 features in four families. All computed in Spark with window
functions, all strictly backward-looking from month *T*.

### A. Static origination (~16)
Direct pass-through from the origination file: credit score, original LTV, CLTV,
DTI, original UPB, original rate, original term, loan purpose, occupancy,
channel, property type, number of units, number of borrowers, first-time buyer
flag, property state, MSA.

### B. Current state (~10)
`loan_age`, `current_upb`, `current_upb / original_upb`, `current_interest_rate`,
`current_rate - original_rate`, `remaining_months`, `current_deferred_upb`,
`estimated_ltv`, `modification_flag`, `deferred_payment_plan`.

### C. Trajectory (~40) — highest signal family

Rolling windows of 3, 6 and 12 months, all computed over rows strictly **before**
*T* (`ROWS BETWEEN n PRECEDING AND 1 PRECEDING` — note the `1 PRECEDING`, not
`CURRENT ROW`):

- `months_delinquent_last_{3,6,12}`
- `max_bucket_last_{3,6,12}`
- `mean_bucket_last_{3,6,12}`
- `n_cure_events_last_{6,12}` — transitions to bucket 0
- `n_roll_events_last_{6,12}` — transitions to a higher bucket
- `months_since_last_current`
- `months_since_first_delinquency`
- `longest_consecutive_delinquent_run`
- `delinquency_velocity_3m` — `dq_bucket[T] - dq_bucket[T-3]`
- `upb_change_pct_{3,6,12}`
- `times_entered_delinquency_lifetime`
- `ever_modified_before_T`

### D. Macro overlay (~6)
Join on `(property_state, reporting_period)`:
- state unemployment rate (FRED series `{STATE}UR`)
- state house price index (FRED series `{STATE}STHPI`)
- 3m and 12m change in each
- national 30-year mortgage rate (`MORTGAGE30US`)

FRED is free. Register for an API key, or download CSVs manually and commit
them under `data/raw/macro/` (small, and not covered by Freddie's terms).

### Leakage guard (`src/utils/leakage.py`)

Two mechanisms, both mandatory:

**1. Static blocklist.** Any column containing these substrings is banned from
the feature set and the check raises on violation:

```python
BLOCKED = [
    "zero_balance", "net_sales_proceeds", "mi_recoveries",
    "non_mi_recoveries", "expenses", "legal_costs", "maintenance",
    "taxes_and_insurance", "actual_loss", "delinquent_accrued_interest",
    "defect_settlement", "ddlpi", "zero_balance_removal_upb",
]
```

These are all post-outcome fields. Including any of them produces an
artificially strong model that is worthless, because at scoring time in the real
world none of these values exist yet.

**2. Automated single-feature AUC check.** For every candidate feature, fit a
univariate model and compute AUC. Print the full table sorted descending.

The check is **relative, not absolute**: a single feature that separates the
outcome far better than everything else around it is the signature of leakage.
Review the top of the table manually and ask, for each one, *could this value
have been known at the start of month T?* If the answer is no or unclear, trace
how the column is computed before keeping it.

Do not auto-drop on a threshold, and do not assume a "normal" AUC for an honest
feature. Require an explicit keep-or-drop decision recorded in `config.yaml`,
with a one-line reason.

**Acceptance checks:**
- Print null rate per feature and decide on each explicitly.
- Print the single-feature AUC table and review the top entries by hand.
- For each of the top features, confirm in code that its window ends strictly
  before month *T*.
- Confirm no feature derives from any column in the blocklist.

---

## 8. Stage 4 — Models (`s4_models.py`)

### Models to fit

1. **Baseline 1:** rank by `dq_bucket` alone (no model). This is the honest
   benchmark — it is what collections operations actually use.
2. **Baseline 2:** logistic regression on WOE-binned features (`optbinning`).
   This is the credit-risk-convention interpretable benchmark.
3. **Primary:** LightGBM binary classifier.
4. **Segmented:** separate LightGBM per entry bucket (1, 2, 3), compared against
   the pooled model with `dq_bucket` as a feature.

### LightGBM starting parameters

```yaml
objective: binary
metric: auc
learning_rate: 0.03
num_leaves: 63
min_data_in_leaf: 200
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 5
lambda_l2: 1.0
num_boost_round: 3000
early_stopping_rounds: 100
```

Early stop on the validation split. Never touch test until the model is frozen.

Check the measured base rate from stage 2 before reaching for imbalance
handling. Resampling and `scale_pos_weight` both damage calibration, and this
project depends on calibration more than most, so apply them only if the
measured base rate genuinely warrants it — and if you do, recalibrate
afterwards and report the effect.

### Calibration

Fit isotonic regression on the **validation** split, apply to test. Report before
and after:
- reliability curve (10 bins)
- Brier score
- predicted vs actual roll rate per decile, as a table

Record in the writeup why calibration matters here specifically: the score is
multiplied by a balance, so miscalibration reorders the queue even when ranking
is correct.

### Tracking

Every run to MLflow with a local file backend:

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("collections-prioritisation")
```

Log: params, AUC, PR-AUC, Brier, per-bucket metrics, feature importance plot,
calibration plot, and the exact feature list used.

**Acceptance checks:**

These are structural. Record whatever numbers come out and judge them on the
relationships below, not against any expected value.

- **Every model must beat Baseline 1 (`dq_bucket` alone).** If LightGBM cannot
  beat an ordinal variable with three levels, something is broken.
- **Validation and test metrics should be in the same neighbourhood.** A large
  gap means overfitting or a split problem.
- **Compare segmented against pooled per bucket and report both**, whichever
  wins. A result where segmentation does not help is a real finding and goes in
  the writeup as-is.
- **Calibration must improve after isotonic scaling.** Compare the reliability
  curve and Brier score before and after and report both numbers.
- **Inspect the top 20 features by importance by hand.** If the ordering is not
  explicable in domain terms, investigate before trusting the model.
- **Run the leakage guard again after training**, on the final feature set, not
  just the candidate set.

---

## 9. Stage 5 — Policy simulation (`s5_policy.py`)

**This is the deliverable. Everything above exists to serve this stage.**

### Mechanic

For each month *m* in the test window:

1. Take the delinquent population at *m*.
2. Compute `actual_money_at_risk(account) = exposure × 1[rolls_deeper == 1]`.
   Sum across the month to get `total_at_risk[m]`.
3. For each policy, rank the same accounts using **only information knowable at
   the start of month *m***.
4. Take the top `C%`, sum their `actual_money_at_risk`.
5. `capture[policy, m] = captured / total_at_risk[m]`.

Average across months, weighted by `total_at_risk[m]`.

The outcomes are used **only to score**, never to rank. Assert this in code: the
ranking function must not receive the label column.

### Policies

| Key | Ranking |
|---|---|
| `random` | random permutation, averaged over 20 seeds |
| `by_dpd` | `dq_bucket` descending, ties broken randomly |
| `by_balance` | `current_upb` descending |
| `by_prob` | `P(roll)` descending |
| `by_expected_value` | `P(roll) × current_upb` descending |
| `oracle` | `actual_money_at_risk` descending — the ceiling |

### Outputs

1. **Capture curve plot:** x-axis = capacity *C* from 5% to 50%, y-axis = capture
   rate, one line per policy. This is the README hero image.
2. **Table at C = 10%, 20%, 30%.**
3. **Per-month capture** for the headline policy, to show stability rather than
   one lucky month.

### Validity checks

No expected values. Whatever the numbers are, these **structural relationships**
must hold, and any violation is a bug rather than a result:

1. `random` capture should land close to the capacity fraction itself. At
   *C* = 20%, random should be near 20%. This is a mechanical property of
   drawing a random subset, so a large deviation means the simulation is wrong.
2. `oracle` must be the highest of every policy, at every capacity.
3. Every modelled policy must sit strictly **below** `oracle`. Any policy that
   matches or exceeds it has the label leaking into the ranking.
4. Every policy must sit at or **above** `random`. A policy below random is
   ranking in the wrong direction — check the sort order.
5. All curves must be monotonically increasing in *C*, and all must reach 100%
   at *C* = 100%.
6. Rank-by-exposure-only (`by_balance`) should roughly track the share of total
   balance held by the largest accounts. Compute that share independently and
   compare — they should be similar, since that policy carries no risk
   information at all.

Report whatever gap exists between `by_expected_value` and the baselines. If the
gap is small, that is the result, and the writeup says so. If it is very large,
re-run checks 2 and 3 before believing it.

Also record, for the writeup: how stable the headline capture is month to month,
and how much of the gap between `by_balance` and `oracle` the model closes.

---

## 10. Stage 6 — Drift study (optional but high-value)

Uses the held-out 2020–2021 window as a real, dated concept-drift event.

1. Compute PSI per feature between the training window and each month of
   2020–2021.
2. Plot PSI over time. Expect a sharp break around 2020-03 to 2020-06 as
   forbearance enters the data.
3. Score the frozen model on those months and plot AUC and calibration error
   alongside PSI.
4. Write up which degrades first. Calibration usually goes before
   discrimination, which is the argument for a separate recalibration trigger.

This section maps directly to production monitoring and is worth building if
time allows.

---

## 10b. Engineering standards

These are not optional extras. A reviewer looking at this repo is assessing the
engineering as much as the modelling.

### Configuration
- **No magic numbers in code.** Every threshold, path, date cutoff, window
  length and hyperparameter lives in `config/config.yaml` and is loaded once.
- Config is validated on load (a dataclass or Pydantic model), so a typo fails
  loudly at startup rather than silently halfway through a Spark job.
- The exact config used for a run is logged to MLflow as an artifact.

### Reproducibility
- A single `random_seed` in config, threaded through Spark, numpy, scikit-learn
  and LightGBM.
- Pin dependencies. Commit `uv.lock` or `requirements.txt` with exact versions.
- `make all` from a clean checkout plus `data/raw/` must reproduce every number
  in the README. Test this at least once before publishing.
- Log the git commit SHA with every MLflow run.

### Code quality
- `ruff` for linting and formatting, `mypy` in non-strict mode for type hints on
  public functions. Configure both in `pyproject.toml`.
- `pre-commit` hooks running ruff and a check that blocks committing anything
  under `data/`.
- Functions do one thing. Stage scripts are thin orchestration over testable
  functions in modules — not 400-line top-to-bottom scripts.
- Docstrings on every public function stating what it returns and, where
  relevant, what time window it is allowed to look at.

### Testing (`tests/`, run with `pytest`)

Unit tests on small synthetic frames, not on the real data. Build a fixture of
a dozen hand-written loan-months where you know the right answer.

Minimum set:
- `test_dq_mapping` — every documented delinquency code maps correctly,
  including `XX` to null.
- `test_label_logic` — a loan that rolls, one that cures, one that holds, one
  that hits each terminal zero-balance code. Assert the expected label.
- `test_censoring` — loans near the end of the window are dropped, not labelled.
- `test_no_future_leakage` — the headline test. Build a fixture where a feature
  would change if it accidentally read month *T* or later, and assert it does
  not.
- `test_split_disjoint` — no loan ID appears in two splits.
- `test_policy_ranking_has_no_label` — assert the ranking function raises if
  passed a frame containing the outcome column.
- `test_capture_math` — a tiny worked example where the capture rate is
  calculable by hand, asserted exactly.

### CI
A GitHub Actions workflow on push: install, lint, type-check, run tests. It
cannot run the pipeline (no data in CI), and that is fine — the tests are
designed to run on synthetic fixtures precisely so CI is possible.

### Modelling discipline
- **Three splits, used correctly.** Train fits the model. Validation does early
  stopping, hyperparameter choice, calibration fitting and all comparisons
  between candidates. Test is touched **once**, at the end, after the model is
  frozen. Every test-set evaluation is logged with a timestamp so repeated
  peeking is visible.
- If you find yourself wanting to change something after seeing a test number,
  that change belongs to a new experiment evaluated on validation, and the test
  result you already saw is reported alongside it.
- Hyperparameter search runs against validation only, with the search space
  recorded in config.
- Feature selection decisions are made on train and validation, never on test.
- Every experiment, including the failed ones, goes to MLflow. A run that made
  things worse is evidence of a real process.

### Documentation
- `docs/methodology.md` is written as you go, not reconstructed at the end. It
  records each decision and the reason for it, including decisions that turned
  out to be wrong.
- `docs/download.md` lets a stranger obtain the data themselves.
- Every notebook in `notebooks/` is exploratory and explicitly marked as such.
  Nothing in the pipeline imports from a notebook.

---

## 11. Makefile

```make
setup:     ## create env and install deps
lint:      ## ruff check + format --check
typecheck: ## mypy src
test:      ## pytest
check:     lint typecheck test
ingest:    ## stage 1
labels:    ## stage 2
features:  ## stage 3
train:     ## stage 4
policy:    ## stage 5
drift:     ## stage 6
all:       ingest labels features train policy
clean:     ## remove interim and processed, keep raw
```

Each target must be independently re-runnable and must not silently reuse stale
outputs. Write a `.done` marker or check input mtimes.

---

## 12. README requirements

The README is a deliverable, not an afterthought. It must contain:

1. **The capture curve plot, first thing after the title.**
2. One-paragraph problem statement framing this as ranking under capacity.
3. The headline number in one sentence.
4. Dataset section pointing to `docs/download.md`, with an explicit note that raw
   data is not committed per Freddie Mac's terms.
5. **A limitations section**, written plainly and placed prominently:
   - This is secured US mortgage data. Collateral changes the loss economics
     completely and the arrears cycle is far slower than unsecured lending. The
     method transfers; the numbers do not.
   - There is no randomised contact assignment in the data, so this measures
     targeting quality, not treatment effect. It assumes contact helps and helps
     roughly equally. Measuring true call effectiveness needs a randomised
     holdout.
6. Reproduction instructions.

Do not soften the limitations section. It is the part that signals judgement.

---

## 13. Things that will go wrong

| Symptom | Cause |
|---|---|
| One feature separates the outcome far better than all others | Post-outcome field in features. Re-run the leakage guard. |
| Model performs no better than chance | Trajectory windows include the current row. Check `1 PRECEDING`. |
| Base rate degenerate (near 0 or near 1) | Label window off by one, or terminal codes not mapped. |
| Capture matches or exceeds `oracle` | Label column leaked into the ranking function. |
| A policy falls below `random` | Sort order reversed somewhere. |
| `by_balance` beats the modelled policies | Exposure read at the wrong month. |
| Train and test metrics far apart | Loan overlap across splits, or overfitting. |
| Spark OOM | Too many vintages at once. Process year by year, write, then union. |
| Inconsistent loan ID types | `inferSchema` used somewhere. Use explicit schemas. |
| Sentinel values appearing in importance plots | Sentinel cleaning skipped. |

---

## 14. Build order

Do not jump ahead. Each stage's acceptance checks must pass before the next.

0. Stage 0 bootstrap: git, `.gitignore`, GitHub repo, directory skeleton.
   **Stop here and tell the user `data/raw/` exists so they can place the
   dataset**, then continue with the environment while waiting.
1. Environment + `config.yaml` + lint/test/CI scaffolding
2. Stage 1 ingest, on **one vintage only**, until checks pass
3. Stage 1 on all vintages
4. Stage 2 labels and splits
5. Stage 3 features, starting with family C (trajectory) since it carries most
   of the signal
6. Leakage guard, run before any model is fitted
7. Stage 4 baselines first, then LightGBM, then segmentation, then calibration
8. Stage 5 policy simulation
9. README and methodology writeup
10. Stage 6 drift study if time allows

Write the tests for each stage **in the same session as the stage itself**, not
at the end. A test written after the fact tends to encode whatever the code
already does.

A working end-to-end pipeline on one vintage is worth more than a half-built
pipeline on twenty.
