# Methodology

A running record of each decision and the reason for it, written as the
project is built. Decisions that later turn out to be wrong stay here, with
what replaced them.

---

## Stage 1 — Ingest

### Schema comes from the official layout, not from the spec

The SFLLD file layout has been revised across releases. `src/utils/schema.py`
is generated from the File Layout workbook (`file_layout_july_2026.xlsx`) by
`src/utils/gen_schema.py`: positions and data types come from the workbook,
and short column names come from a curated map keyed on the workbook's
attribute names. If Freddie Mac adds, drops or renames a field, generation
fails and names the mismatch, instead of silently shifting every later column.

The July 2026 layout differs from the field list in the original build spec:

- The loan key is now called *Loan Identifier*. It is kept as
  `loan_sequence_number` in code so the column name stays stable.
- The credit score field is *Classic FICO®*; a *VantageScore® 4.0* field has
  been added. Both use `9999` for not-available.
- Performance files have 35 fields; origination files 31. Every vintage from
  2000 to 2024 has the same field count.

### Read as text, then cast, and count failures

Files are read with an explicit all-string schema (never `inferSchema`), then
cleaned, then cast. Reading as text first means a value that fails its cast
is *counted* rather than silently becoming null. Ingest refuses to write the
panel if any field has a cast failure, if any row has the wrong number of
fields, or if any delinquency status is unrecognised.

Layout fields typed "Numeric" are treated as integers only when short (≤ 4
characters). Longer ones are dollar amounts that the files write with
decimals (`0.00`), and are read as doubles. MSA, zero-balance code and
property-valuation method are numeric-looking *codes*, kept as strings so
`"01"` keeps its leading zero and no ordering is implied.

### Sentinels

Every not-available sentinel documented in the General User Guide is mapped to
null, not only the four listed in the spec: `9999` (FICO, VantageScore), `999`
(MI %, CLTV, DTI, LTV, ELTV), `99` (units, property type, borrowers), `9`
(first-time buyer, occupancy, channel, loan purpose), `7` (valuation method,
MI cancellation) and `000` (postal code). Sentinels are matched per column
on the trimmed raw text, so a balance that happens to equal `9999` is never
nulled.

### Delinquency status → `dq_bucket`

- Numeric statuses are months delinquent (the guide documents a cap at 99).
  `dq_months` keeps the uncapped value; `dq_bucket = min(dq_months, 4)`.
- `RA` (REO acquisition) is terminal and maps to `dq_bucket = 4`, with
  `is_reo = true`. REO is strictly worse than any delinquency, so treating it
  as the deepest bucket makes "rolls deeper" behave correctly without a
  special case. On the 2015 vintage, every loan that reaches `RA` stays `RA`
  until its last record.
- `XX` (not available) is null and those loan-months are removed from the
  panel. They are never coerced to current.

### Columns dropped at ingest

The disposition fields (actual loss, MI and non-MI recoveries, net sales
proceeds, expenses, zero-balance removal UPB, delinquent accrued interest,
modification costs, cramdown costs) are not carried into the panel. They are
only populated at or after the terminal event, so they can only ever leak the
outcome. Dropping them at the source means no later stage can pick them up
by accident.

DDLPI (due date of last paid installment) and the defect settlement date are
also dropped, matching the Stage 3 leakage blocklist. DDLPI largely restates
the delinquency status and can be revised after the fact, so keeping it in
the panel would only create a second place for the leakage guard to police.

### The source has month gaps

The 2015 vintage has no `XX` rows, yet 2 of 50,000 loans have a missing month
inside their history (e.g. one jumps from 2017-05 to 2017-08 while 12 months
delinquent; another shows a $0 balance with no zero-balance code and then
reappears two months later). These are source artefacts.

**Consequence for Stage 2:** label and trajectory windows must step through
time by calendar month, not by row offset. A `ROWS BETWEEN 1 FOLLOWING AND 3
FOLLOWING` window would silently reach four or five months ahead across a gap.

### Zero Balance Codes have changed since the spec

The July 2026 guide no longer lists `06` (repurchase) and adds `15` (whole
loan sales) and `16` (reperforming loan securitisations). Classification
(settled in Stage 2, see below):

| Code | Meaning | Treat as |
|---|---|---|
| 01 | Prepaid or matured | cure |
| 02 | Third party sale | roll |
| 03 | Short sale or charge off | roll |
| 09 | REO disposition | roll |
| 15 | Whole loan sale | roll |
| 16 | Reperforming loan securitisation | exclude |
| 96 | Confirmed defect | exclude |

Code `16` matters more than it looks: in the 2015 vintage it terminates 108
loans, more than codes 02, 03, 09 and 15 combined. Across all 25 vintages it
terminates 5,850 loans, against 11,550 REO dispositions (09), 4,564 short
sales (03), 3,286 third-party sales (02) and 1,284 whole loan sales (15).
The loans it removes
have delinquency histories (one spot-checked loan was 20 months delinquent,
was modified, re-performed, then left with code 16). It is a sale of a
reperforming loan out of the portfolio, not an observed credit outcome, which
argues for treating it as censoring rather than as a roll or a cure.

### Full ingest (all 25 vintages, 2000–2024)

- 1,250,000 loans; 72,011,687 performance rows; 5 `XX` rows removed, giving
  a 72,011,682-row panel covering reporting periods 2000-01 to 2026-03.
- Zero malformed rows, cast failures, unmatched loans or duplicate
  `(loan, period)` keys.
- 188 loans have a missing month inside their history (see above).
- 98,148 loan-months are `RA`.

### Environment

PySpark 3.5 supports Java 8/11/17 only; the machine default was Java 24.
`src/spark.py` locates a supported JDK (respecting `JAVA_HOME`) and fails with
an install instruction if there is none.

---

## Stage 2 — Population, label, splits

### Population

Loan-months with `dq_bucket` in 1, 2 or 3 (30–119 days), not REO, and not
carrying a zero-balance code that month. Current accounts are not worked by
collections, and at 120+ days routine telephony is no longer the right lever.

### Label: calendar months, not rows

`rolls_deeper = 1` if, in months T+1 .. T+3, the loan reaches a bucket deeper
than at T (REO counts as deepest) or terminates with a "roll" zero-balance
code. The window is a Spark `RANGE` window over a calendar month index, not
the spec's `ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING`: the panel has missing
months (Stage 1), and a row window would silently reach T+4 or T+5 across a
gap. `tests/test_labels.py` has fixtures where the two give different answers.

A label of 0 needs evidence. In order:

| Situation | Outcome |
|---|---|
| T is in the last 3 months of the data | dropped (`end_of_data`) |
| deeper bucket seen in the window | 1 |
| roll zero-balance code in the window | 1 |
| excluded code (16, 96) in the window, no roll before it | dropped |
| prepaid / matured (01) in the window | 0, terminal outcome observed |
| all 3 months observed, no roll | 0 |
| anything else (history ends or has a gap, no roll seen) | dropped (`incomplete_window`) |

A roll seen across a gap is kept as 1: it is an observed fact. A non-roll
across a gap is dropped: the missing month could have hidden a roll.

### Code 16 is excluded

Decided with the project owner: a reperforming-loan securitisation is a sale
out of the portfolio, not an observed credit outcome, so it is treated as
censoring. In practice it drops 1,593 population rows (0.13%), together with
code 96.

### Splits are fully disjoint by loan

The spec removes train loans from validation and test. This goes one step
further: each loan stays only in the **first** split in which it has a
labelled row, and its rows in every later split are dropped. Otherwise a loan
could sit in both validation (used for tuning) and test, which would leak
tuning into the test number, and `test_split_disjoint` requires no loan in
two splits.

The cost is real and is recorded here rather than hidden: 267,629 labelled
rows are dropped by this rule, and validation (27,436 rows, 11,254 loans) and
test (31,061 rows, 13,566 loans) are much smaller than train (706,490 rows,
97,424 loans). It also changes *who* is in the later splits: validation and
test hold only loans whose first delinquency episode falls in that window,
so repeat delinquents, which are mostly in train, are under-represented
there. Any drop in base rate across splits partly reflects this selection,
not only the passage of time.

55,021 labelled rows fall after the last split (2025) and are unused.

### Results of the checks

| Split | Rows | Loans | Base rate | Bucket 1 | Bucket 2 | Bucket 3 |
|---|---|---|---|---|---|---|
| train | 706,490 | 97,424 | 0.398 | 0.314 | 0.524 | 0.724 |
| validation | 27,436 | 11,254 | 0.330 | 0.242 | 0.599 | 0.726 |
| test | 31,061 | 13,566 | 0.257 | 0.195 | 0.481 | 0.720 |
| drift_study | 71,544 | 25,325 | 0.576 | 0.453 | 0.716 | 0.732 |
| late_holdout | 63,186 | 21,543 | 0.328 | 0.243 | 0.553 | 0.698 |

The base rate rises with bucket depth in every split, which is the structural
check that the label logic is the right way round. The drift-study window has
a visibly different structure (bucket 1 rolls far more often), consistent with
forbearance changing what delinquency means; it is never trained or tested on.
Three label-1 and three label-0 rows were traced month by month and match
what happened.

---

## Stage 3 — Features

71 candidate features in four families, computed in Spark for every
labelled row (899,717 rows, one feature row per label row).

| Family | Candidates | Kept |
|---|---|---|
| A — static origination | 17 | 16 |
| B — current state at T | 14 | 9 |
| C — trajectory | 31 | 28 |
| D — macro | 9 | 5 |

### Every feature declares its window

`src/pipeline/features.py` has a registry giving each feature's source
columns and the newest month it reads relative to T: static, `T+0`, or `T-k`.
Trajectory windows are `RANGE` windows on the calendar month index ending at
T-1 (the spec's `1 PRECEDING`), so a missing month cannot pull month T in.

Five trajectory features read month T **by definition**, as the spec writes
them: `delinquency_velocity_3m` (bucket at T minus bucket at T-3),
`upb_change_pct_{3,6,12}` (balance at T against T-n), and
`current_run_length`. Month T's own values are reported for month T and are
known when the month-T contact list is built.

### The guard is tested, not only declared

`tests/test_features.py` changes everything after T and asserts no feature
moves. Then, for k = 0, 1, 2, it changes month T-k (once pushing values up,
once down) and asserts that every feature declared to end before T-k does not
move. As a check on the check, widening one window to include month T made
this test fail and name the six affected features.

The static blocklist (spec section 7) is applied to feature names and to
every declared source column; post-disposition fields were already dropped at
ingest. The label columns are blocked too.

### Macro: lagged to publication date

FRED values are lagged to when they were public: state unemployment 2 months,
quarterly state house prices 5 months from the quarter's start date (≈2
months after quarter end), national mortgage rate 1 month. FRED serves today's
revised values, not what was published at the time (that would need ALFRED
vintages); this is a small, acknowledged look-ahead in the macro family.
Guam and the Virgin Islands have no state series, Puerto Rico has no house
price index; those features are null there. The series are downloaded by
`make macro`, not committed: `data/` stays entirely out of git.

### Audit and decisions

`outputs/tables/feature_audit.csv` has null rates and single-feature AUC
(each feature alone, fit on train, scored on validation) for all 71
candidates. Reviewed by hand:

- **No leakage signature.** The top feature is `dq_bucket` (validation AUC
  0.658), followed by a smooth run of trajectory features from 0.652 down to
  about 0.60. Nothing stands apart from its neighbours.
- **Fields Freddie Mac only started populating recently are dropped:** ELTV
  (2017+), payment deferral (2020+), disaster and borrower-assistance flags
  (2014+). They are 98–100% null in train, so there is nothing to learn; they
  belong to the drift study.
- **National series are dropped.** The 30-year mortgage rate and its changes
  are identical for every loan in a month, so they cannot change a ranking
  made within a month, and univariately they flip from train (0.55–0.57) to
  validation (0.48–0.49): they act as a stand-in for the time period.
- **State house-price level is dropped:** each state's index has its own
  base, so the level encodes state × time. Its percentage changes are kept.
- **MSA is dropped** (≈400 codes, 24% null, overfits univariately);
  `property_state` carries geography. **Step-modification flag** duplicates
  the modification flag.
- `house_price_index_chg_12m` inverts between train (0.58) and validation
  (0.47), because 2016–17 had rising prices everywhere. It is kept, since it
  is economically meaningful and varies across states, and is flagged for the
  drift study.
- `orig_dti` is 6% null in train but 26% in validation (HARP refinances
  carry no DTI, and they are concentrated among first-episode loans in the
  later splits). Kept, with native null handling.

Every one of the 71 decisions, with its one-line reason, is in
`config.yaml` under `features.decisions`; Stage 3 fails if a candidate has
no decision.

### Pipeline rebuilds

Each stage's `make` target depends on a stamp of only its own config section
(`src/utils/config_stamp.py`), so editing a feature decision re-runs Stage 3
but not the 15-minute ingest.

**Revised after Stage 5: content, not file times.** Stage targets also
depended on the *mtimes* of their source files. A git checkout or merge
rewrites mtimes, so after the Stage 4 merge every stage looked stale, and
`make policy` would have re-run the whole pipeline, including a fourth
(logged, unnecessary) test evaluation. Each stage now depends on one stamp
holding its validated config sections, a SHA-256 of each of its source
files, its external inputs (raw and macro files: name, size, mtime; never
touched by git), and the hash of the upstream stage's stamp, so a real
change propagates down the chain. Upstream markers are order-only
prerequisites. `src/utils/config.py` is deliberately not hashed: the stamp
already holds the validated values, so adding a config class for one stage
does not invalidate the rest.

The switch was adopted without re-running: no stage's logic changed after
its last run (the only later edits were a refactor that extracted
`load_manifest` in `s4_models.py` and new plot functions for Stage 5), so
the new stamps were written and the existing markers kept. Checked with a
dry run: nothing to do in the clean state; editing a Stage 5 source re-runs
only Stage 5; editing a Stage 2 source re-runs Stages 2–5 but not ingest.

---

## Stage 4 — Models

### Protocol

`python -m src.s4_models` fits on train, early-stops and selects on
validation, fits isotonic calibrators on validation, and freezes everything
with a manifest (feature list, best iterations, config hash, git SHA). It
never reads test. `python -m src.s4_models --evaluate-test` loads the frozen
models, refuses to run if the kept features or model config changed since the
freeze, and appends every evaluation to `outputs/tables/test_evaluations.csv`
with a timestamp, so repeated looks at test are visible in git history.

Training is deterministic: a second fit reproduced every validation number
exactly. The train base rate is 0.398, so no resampling or class weights were
used (both would damage calibration).

MLflow 3.16 refuses the file store named in the spec (`file:./mlruns`); runs
go to its local SQLite store under `mlruns/` instead, still git-ignored.

### Validation

| Model | AUC | PR-AUC | Brier (raw) |
|---|---|---|---|
| Baseline: `dq_bucket` alone | 0.6578 | 0.4766 | – |
| WOE logistic | 0.7135 | 0.5843 | 0.1896 |
| LightGBM pooled | 0.7392 | 0.6138 | 0.1833 |
| LightGBM segmented by bucket | 0.7396 | 0.6138 | 0.1828 |

**Segmented vs pooled is a tie.** Segmented won each bucket on validation by
0.0006, 0.0013 and 0.0040 AUC (0.0004 overall); its bucket-2 and bucket-3
models early-stopped at 69 and 50 rounds on small validation slices. The
original rule ("higher validation AUC wins") would have picked segmented on
noise. **After seeing those validation numbers, and before any test
evaluation,** the rule was changed: segmented must beat pooled by at least
0.005 validation AUC, otherwise the simpler pooled model is primary. Pooled
was chosen. On test, pooled beat segmented in every bucket (0.6460 vs 0.6454,
0.6182 vs 0.6122, 0.6299 vs 0.6237), which is consistent with segmentation
having fitted validation noise; that test result was not used to choose.

**Top features (gain, pooled)** are explicable: 3-month bucket velocity
(20% of gain), recent maximum bucket, 3-month balance change (a balance that
has not fallen means a missed payment), property state (foreclosure process
and timelines differ sharply by state), the 12-month and lifetime trajectory,
and the state house-price trend.

The leakage guard was re-run on the final feature set: every booster uses
exactly the 58 kept features and all pass the blocklist and window checks.

### Test (evaluated once, 2026-10-05)

| Model | AUC | PR-AUC | Validation AUC | Gap |
|---|---|---|---|---|
| Baseline: `dq_bucket` alone | 0.6393 | 0.3918 | 0.6578 | 0.019 |
| WOE logistic | 0.7040 | 0.5059 | 0.7135 | 0.010 |
| LightGBM pooled (primary) | 0.7238 | 0.5227 | 0.7392 | 0.015 |
| LightGBM segmented | 0.7241 | 0.5222 | 0.7396 | 0.016 |

Every model beats the baseline on test, and validation and test AUC are
within 0.02 for every model.

### Calibration did not survive to test (acceptance check failed)

| Pooled LightGBM, test | Raw | Isotonic (fit on validation) |
|---|---|---|
| Brier | 0.1627 | 0.1646 |
| ECE (10 bins) | 0.0135 | 0.0444 |

Isotonic calibration fitted on validation made test calibration **worse**.
The raw model was already well calibrated on test: its decile table tracks
the actual roll rate within about 0.01–0.03 except the top decile
(predicted 0.707, actual 0.656). The isotonic map raised predictions across
the middle of the range (deciles 3–9 over-predict by 0.03–0.07 after
calibration).

The likely mechanism is that the calibration map learned something specific
to the 2016–17 validation window that did not hold in 2018–19: the raw
model under-predicted on validation and over-corrected for test. The splits
also differ in composition (Stage 2: later splits contain only loans whose
first delinquency falls in that window), so a single two-year calibration
window is a weak basis for a map.

Why it matters here: the score is P(roll) × balance, so miscalibration
reorders the queue even when ranking by P(roll) alone is correct. A
probability that is uniformly 0.05 too high does not change the order; one
that is too high only in the middle of the range does, once multiplied by
balances of different sizes.

This is recorded as observed. Choosing raw over isotonic *because* of this
test result would be a test-informed decision; any change to the calibration
approach has to be a new experiment judged on data other than test, with
this test result reported alongside it.

### Follow-up experiment: does a calibration map carry forward in time?

Decided with the project owner after the failure above, and fixed in config
before it was run: fit isotonic on validation minus its last 12 months
(2016, 9,155 rows), score the last 12 months (2017, 18,281 rows), and keep
isotonic only if it improves **both** Brier and ECE over raw scores there.
No test data is involved.

| 2017 holdout (calibrator fit on 2016) | Brier | ECE |
|---|---|---|
| Raw | 0.1931 | 0.0470 |
| Isotonic | 0.1930 | 0.0377 |

Isotonic passed the rule, so it stays the downstream calibration. The Brier
improvement is 0.0001, which is effectively nil; the rule was not tightened
after the fact, because that would be choosing the rule from its result.

The frozen models were re-fitted (identical boosters and validation AUCs) and
re-evaluated on test; the test log shows both evaluations. Test numbers are
unchanged and **the calibration acceptance check still fails** (Brier 0.1627
raw vs 0.1646 isotonic; ECE 0.0135 vs 0.0444).

What this shows: within 2016–17, a one-year-old calibration map still helped,
so validation could not see the problem; the shift that breaks it happens
between the validation and test windows. A single two-year window is not
enough to estimate a calibration map that holds in the next period. Stage 5
should therefore report its policy results under both calibrated and raw
probabilities, so the reader can see how much the ranking depends on this
choice, with the frozen (isotonic) result as the primary number.

### Second follow-up: is it the method, or the window?

Decided with the project owner: try other calibration methods, chosen by a
rule fixed in config before running, still without test data. Candidates fit
on validation minus its last 12 months and scored on those months; the
lowest held-out ECE among methods beating raw on both Brier and ECE wins.

| 2017 holdout (fit on 2016) | Brier | ECE | Beats raw |
|---|---|---|---|
| Raw | 0.1931 | 0.0470 | – |
| Isotonic | 0.1930 | 0.0377 | yes |
| **Platt (chosen)** | 0.1922 | 0.0367 | yes |
| Intercept shift | 0.1924 | 0.0386 | yes |

Platt was frozen (refit on all of validation) and test was evaluated a third
time (logged):

| Test, pooled LightGBM | Raw | Isotonic (2nd eval) | Platt (3rd eval) |
|---|---|---|---|
| Brier | **0.1627** | 0.1646 | 0.1643 |
| ECE | **0.0135** | 0.0444 | 0.0428 |

**The calibration acceptance check fails for every method fitted on
validation, by about the same amount.** After Platt scaling every test decile
over-predicts by 0.02–0.07. This is a level shift, not a shape problem: in
2016–17, loans rolled more often than the model predicted, so any map fitted
there raises predictions; in 2018–19 they rolled about as often as the raw
model (trained on 2000–2015) predicts. Every map was lifting predictions by
roughly the 2016–17 shortfall, and that shortfall did not continue into
2018–19. No method can fix that from the validation window alone.

A longer calibration window was considered and not attempted: validation is
the only out-of-sample period for the frozen model before test, so a longer
window would mean calibrating a different model trained on fewer years.

**Conclusion.** Calibration from the most recent out-of-sample window does
not transfer forward here. The frozen pipeline uses Platt; Stage 5 reports
policy results under both the frozen calibrated probabilities and raw
probabilities, so the effect of this choice on the ranking is visible. Test
has now been evaluated three times; every evaluation is in
`outputs/tables/test_evaluations.csv`, and the AUC results never changed
(the boosters are identical across all three).

### Accepted failure

Agreed with the project owner: stop searching for a calibrator (each attempt
is another look at test, and the evidence points at the window, not the
method) and accept the calibration check as a documented failure. It is
recorded in `config.yaml` under `acceptance.accepted_failures` with its
reason; the config refuses an accepted failure without a substantive reason
or for an unknown check. That section sits outside `models:`, so it does not
change the frozen model. No fourth test evaluation was run: the third
evaluation (Platt) is the one being accepted.

---

## Stage 5 — Policy simulation

### Protocol (fixed before the test run)

`python -m src.s5_policy` reads the per-row scores the frozen Stage 4 models
wrote; nothing is fitted or chosen here. It refuses to run if the scores are
older than the frozen models or the model config changed since the freeze.

For each month in the window, every policy ranks the same delinquent
accounts. The ranking function receives the month's frame **with the label
removed** and raises if it sees `rolls_deeper` or `money_at_risk`; a test
spies on every call to confirm. The label is used only afterwards, to score:
`money_at_risk = exposure × rolls_deeper`, and capture at capacity *C* is the
money at risk in the first ⌊C·n⌋ accounts divided by the month's total.
Months are pooled (sum captured / sum at risk), which is the risk-weighted
mean the spec asks for. The oracle is the only ordering built from the
outcome; it is a separate function, not a policy.

Every policy breaks ties randomly and is averaged over 20 seeds (this is
what makes `random` a random permutation and `by_dpd` a bucket sort with
random tie-breaks). Exposure is `current_upb` at T, the same month-T value
the features treat as known at T.

Policies are the spec's six plus one sensitivity line committed to in Stage 4:
`by_expected_value_raw`, the same score using the raw LightGBM probability
instead of the frozen Platt-calibrated one. `by_prob` needs no raw twin:
Platt is monotone, so it cannot change a ranking by probability alone.

Fixed in `config.yaml` (`policy:`) before any run: capacities (5–50% for the
curve; 10/20/30% for the table; 20% headline), the headline policy
(`by_expected_value` with the frozen probabilities), 20 seeds, and two
check tolerances: random within 0.02 of the capacity, and `by_balance`
within 0.05 of the balance share of the largest accounts (check 6).

The mechanics were rehearsed on **validation** first (all checks passed).
One observation from that rehearsal: `by_balance` sat about 0.04 below the
balance share, because the largest 20% of accounts by balance rolled less
often (0.286 vs 0.341). That is a real relationship, not a bug, and it is
close to the check-6 tolerance. The tolerance was **not** changed; instead
the script now prints the roll rate of the largest accounts vs the rest, so
a check-6 result on test can be explained either way.

### Results (test window, 2018-01 to 2019-12)

31,061 delinquent account-months over 24 months. 25.7% of accounts rolled
deeper, holding 24.9% of the queue's balance.

**All 23 structural checks passed.** Random captured within 0.003 of the
capacity at every point; every policy sat strictly below the oracle and at or
above random; every curve was non-decreasing and reached 100% at full
capacity. Check 6 also held: `by_balance` captured 0.390 at C = 20% against
a balance share of 0.401. As in validation it sits slightly below, because
the largest 20% of accounts rolled slightly less often (0.240 vs 0.262).

| Capture of deteriorating balance | C = 10% | C = 20% | C = 30% |
|---|---|---|---|
| `random` | 0.099 | 0.200 | 0.301 |
| `by_dpd` | 0.226 | 0.384 | 0.465 |
| `by_balance` | 0.224 | 0.390 | 0.524 |
| `by_prob` | 0.237 | 0.411 | 0.521 |
| **`by_expected_value`** (frozen, Platt) | **0.336** | **0.502** | **0.623** |
| `by_expected_value_raw` | 0.332 | 0.506 | 0.625 |
| `oracle` | 0.643 | 0.920 | 0.999 |
| Gap closed, `by_balance` → oracle | 27% | 21% | 21% |
| Gap closed, `by_dpd` → oracle | 26% | 22% | 30% |

**Headline:** contacting 20% of the delinquent queue ranked by
P(roll) × balance captured 50.2% of the balance that went on to roll deeper,
against 38.4% for ranking by days past due, 39.0% by balance, and 20.0% at
random. It closes about a fifth of the gap between the best baseline and the
oracle.

What the table says about *where* the gain comes from: ranking by probability
alone (`by_prob`, 0.411) barely beats days past due or balance alone. Most of
the gain comes from the product. Neither a risk ranking nor an exposure
ranking alone is much better than the other, but combining them is. This is
the spec's reframe showing up in the data. The oracle's ceiling is high
because only a quarter of the balance rolls, so a perfect 20% list holds
most of it; the model is far from that.

**Calibrated vs raw probabilities.** The two `by_expected_value` lines differ
by at most 0.004 at the table capacities (Platt ahead at 10%, raw ahead at
20% and 30%). Stage 4's accepted calibration failure, a level shift of a few
points, barely reorders the queue here. That fits: multiplying every
probability by a constant would not change the ordering of P × balance at
all, and the frozen Platt map is mild (log-odds slope 0.86, intercept
+0.05), so it reorders only accounts whose P × balance values were already
close.
The frozen (Platt) number stays the headline, as decided in Stage 4.

**Month-to-month stability (C = 20%).** `by_expected_value` captured
0.499 ± 0.042 per month (range 0.377–0.559). It beat `by_dpd` in all 24
months and `by_balance` in 23. The exception is **2018-01**, the first test
month: because the splits are disjoint by loan, every loan delinquent in
January 2018 that was also delinquent before 2018 belongs to validation, so
the test window's first month is 410 bucket-1 first-time delinquents out of
411. Days past due carries no information there (`by_dpd` 0.214, about random),
the model's strongest trajectory features have nothing to work with, and
`by_prob` fell below random for that month (0.154, on 411 accounts). By
2018-03 buckets 2 and 3 have refilled and the month looks like the rest. The
aggregate checks are unaffected; the effect is a property of the split
design, and it is recorded rather than trimmed.

Artifacts: `outputs/tables/policy_*_test.csv`,
`outputs/figures/capture_curve_test.png` (README hero) and
`monthly_capture_test.png`; MLflow run `stage5-policy-test`.
