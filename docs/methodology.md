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
