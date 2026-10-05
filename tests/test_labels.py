"""Stage 2 label, censoring and split logic on hand-written loans with known answers."""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import functions as F

from src.pipeline import labels as L
from src.utils.config import LabelsConfig, SplitConfig

CFG = LabelsConfig(
    population_buckets=[1, 2, 3],
    horizon_months=3,
    zero_balance_codes={
        "01": "cure",
        "02": "roll",
        "03": "roll",
        "09": "roll",
        "15": "roll",
        "16": "exclude",
        "96": "exclude",
    },
)
SCHEMA = (
    "loan_sequence_number string, reporting_period date, dq_bucket int, "
    "is_reo boolean, zero_balance_code string"
)
FAR_FUTURE = dt.date(2030, 1, 1)  # max period that never censors the fixtures


def m(i: int) -> dt.date:
    """Month i after 2015-01 as a first-of-month date."""
    return dt.date(2015 + i // 12, i % 12 + 1, 1)


def loan(name, buckets, zbc_at=None, zbc=None, skip=(), reo_from=None):
    """Rows for one loan from month 0, one bucket per month; optional terminal code."""
    rows = []
    for i, b in enumerate(buckets):
        if i in skip:
            continue
        is_reo = reo_from is not None and i >= reo_from
        code = zbc if i == zbc_at else None
        rows.append((name, m(i), 4 if is_reo else b, is_reo, code))
    return rows


def label_at(spark, rows, month, max_period=FAR_FUTURE):
    """Return (rolls_deeper, label_reason) for the row at ``month`` of the single loan."""
    df = spark.createDataFrame(rows, SCHEMA)
    out = L.build_labels(df, CFG, max_period).filter(F.col("reporting_period") == m(month))
    got = out.collect()
    assert len(got) == 1, f"expected one population row at month {month}, got {len(got)}"
    return got[0][L.LABEL], got[0][L.REASON]


# --- test_label_logic ---------------------------------------------------------


def test_roll_to_deeper_bucket(spark):
    assert label_at(spark, loan("A", [0, 1, 1, 2, 2]), 1) == (1, L.DEEPER_BUCKET)


def test_cure_back_to_current_is_not_a_roll(spark):
    assert label_at(spark, loan("A", [0, 1, 0, 0, 0]), 1) == (0, L.NO_ROLL)


def test_hold_in_same_bucket_is_not_a_roll(spark):
    assert label_at(spark, loan("A", [0, 2, 2, 2, 2]), 1) == (0, L.NO_ROLL)


def test_roll_only_at_t_plus_3_counts(spark):
    assert label_at(spark, loan("A", [0, 1, 1, 1, 2]), 1) == (1, L.DEEPER_BUCKET)


def test_roll_at_t_plus_4_is_outside_window(spark):
    assert label_at(spark, loan("A", [0, 1, 1, 1, 1, 2]), 1) == (0, L.NO_ROLL)


def test_month_t_itself_is_not_in_window(spark):
    # bucket 3 at T and 3 after: only something deeper than T counts
    assert label_at(spark, loan("A", [3, 3, 3, 3]), 0) == (0, L.NO_ROLL)


def test_reo_after_bucket_3_is_a_roll(spark):
    assert label_at(spark, loan("A", [3, 3, 0], reo_from=2), 0) == (1, L.DEEPER_BUCKET)


@pytest.mark.parametrize("code", ["02", "03", "09", "15"])
def test_roll_zero_balance_codes(spark, code):
    # balance zeroed at T+2 while still bucket 1: the code alone makes it a roll
    rows = loan("A", [1, 1, 1], zbc_at=2, zbc=code)
    assert label_at(spark, rows, 0) == (1, L.ROLL_ZBC)


def test_prepayment_is_an_observed_cure_even_with_short_window(spark):
    rows = loan("A", [1, 0], zbc_at=1, zbc="01")  # loan ends at T+1
    assert label_at(spark, rows, 0) == (0, L.CURE_ZBC)


@pytest.mark.parametrize("code", ["16", "96"])
def test_excluded_zero_balance_codes_drop_the_row(spark, code):
    rows = loan("A", [1, 0], zbc_at=1, zbc=code)
    assert label_at(spark, rows, 0) == (None, L.EXCLUDED_ZBC)


def test_roll_before_excluded_code_is_still_a_roll(spark):
    rows = loan("A", [1, 2, 0], zbc_at=2, zbc="16")
    assert label_at(spark, rows, 0) == (1, L.DEEPER_BUCKET)


def test_population_excludes_current_120_plus_reo_and_terminating_rows(spark):
    rows = (
        loan("CUR", [0, 0, 0, 0])
        + loan("D120", [4, 4, 4, 4])
        + loan("REO", [3, 3], reo_from=0)
        + loan("END", [1], zbc_at=0, zbc="01")
        + loan("IN", [2, 2, 2, 2])
    )
    df = spark.createDataFrame(rows, SCHEMA)
    out = L.build_labels(df, CFG, FAR_FUTURE).select("loan_sequence_number").distinct()
    assert {r[0] for r in out.collect()} == {"IN"}


# --- test_censoring -----------------------------------------------------------


def test_final_months_are_dropped_not_labelled(spark):
    rows = loan("A", [1, 1, 2, 3, 3, 3])
    df = spark.createDataFrame(rows, SCHEMA)
    out = {
        r["reporting_period"]: (r[L.LABEL], r[L.REASON])
        for r in L.build_labels(df, CFG, max_period=m(5)).collect()
    }
    # T=m(2) has its full window m(3)..m(5); m(3)..m(5) do not, even where a roll is visible
    assert out[m(2)] == (1, L.DEEPER_BUCKET)
    for i in (3, 4, 5):
        assert out[m(i)] == (None, L.END_OF_DATA)


def test_record_ending_without_terminal_code_is_dropped(spark):
    rows = loan("A", [1, 1])  # history stops at T+1, no zero-balance code
    assert label_at(spark, rows, 0) == (None, L.INCOMPLETE)


def test_missing_month_is_not_bridged_by_row_offsets(spark):
    # T+1 missing: rows T+2, T+3 present -> only 2 of 3 months observed, no roll seen
    rows = loan("A", [1, 1, 1, 1, 2], skip={1})
    assert label_at(spark, rows, 0) == (None, L.INCOMPLETE)


def test_month_t_plus_4_after_a_gap_is_not_pulled_into_window(spark):
    # A ROWS window of 3 would reach the bucket-2 row at T+4 across the gap
    rows = loan("A", [1, 1, 1, 1, 2], skip={2})
    assert label_at(spark, rows, 0) == (None, L.INCOMPLETE)


def test_roll_seen_across_a_gap_is_still_a_roll(spark):
    rows = loan("A", [1, 1, 2, 2], skip={1})
    assert label_at(spark, rows, 0) == (1, L.DEEPER_BUCKET)


def test_last_labellable_period():
    assert L.last_labellable_period(dt.date(2026, 3, 1), 3) == dt.date(2025, 12, 1)
    assert L.last_labellable_period(dt.date(2026, 1, 1), 3) == dt.date(2025, 10, 1)


# --- test_split_disjoint ------------------------------------------------------

SPLITS = [
    SplitConfig(name="train", start="2015-01", end="2015-06"),
    SplitConfig(name="validation", start="2015-07", end="2015-12"),
    SplitConfig(name="test", start="2016-01", end="2016-06"),
]


def test_split_disjoint(spark):
    rows = [
        ("STRADDLE", m(5), 1, 1, "x"),  # train
        ("STRADDLE", m(6), 1, 1, "x"),  # validation -> dropped
        ("STRADDLE", m(13), 1, 0, "x"),  # test -> dropped
        ("VAL_THEN_TEST", m(8), 1, 0, "x"),
        ("VAL_THEN_TEST", m(12), 1, 0, "x"),  # test -> dropped
        ("TEST_ONLY", m(14), 2, 1, "x"),
        ("OUTSIDE", m(30), 2, 1, "x"),  # after every split
    ]
    df = spark.createDataFrame(
        rows,
        f"loan_sequence_number string, reporting_period date, dq_bucket int, "
        f"{L.LABEL} int, {L.REASON} string",
    )
    out = L.assign_disjoint_splits(df, SPLITS).collect()
    got = sorted((r["loan_sequence_number"], r["reporting_period"], r["split"]) for r in out)
    assert got == [
        ("STRADDLE", m(5), "train"),
        ("TEST_ONLY", m(14), "test"),
        ("VAL_THEN_TEST", m(8), "validation"),
    ]
    per_loan = {}
    for name, _, split in got:
        per_loan.setdefault(name, set()).add(split)
    assert all(len(s) == 1 for s in per_loan.values())
