"""Stage 1 transforms on hand-written rows with known answers."""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import functions as F

from src.pipeline import checks
from src.pipeline.ingest import (
    CORRUPT_COL,
    build_panel,
    cast_columns,
    clean_strings,
    delinquency_columns,
    prepare_performance,
    read_pipe_files,
)
from src.utils.config import DelinquencyConfig

DQ = DelinquencyConfig(unknown_code="XX", reo_code="RA", bucket_cap=4, reo_bucket=4)


def _status_frame(spark, codes):
    return spark.createDataFrame([(c,) for c in codes], "delinquency_status string")


# --- test_dq_mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "months", "bucket", "is_reo"),
    [
        ("00", 0, 0, False),
        ("01", 1, 1, False),
        ("02", 2, 2, False),
        ("03", 3, 3, False),
        ("04", 4, 4, False),
        ("12", 12, 4, False),
        ("99", 99, 4, False),  # documented cap
        ("100", 100, 4, False),  # 3-char field: still numeric, still 4+
        ("RA", None, 4, True),
        ("XX", None, None, None),  # unknown is null, never coerced to current
        (None, None, None, None),
        ("Q1", None, None, None),
    ],
)
def test_dq_mapping(spark, raw, months, bucket, is_reo):
    df = _status_frame(spark, [raw]).withColumns(
        delinquency_columns(F.col("delinquency_status"), DQ)
    )
    row = df.first()
    assert (row["dq_months"], row["dq_bucket"], row["is_reo"]) == (months, bucket, is_reo)


def test_prepare_performance_drops_only_unknown_rows(spark):
    df = spark.createDataFrame(
        [
            ("L1", dt.date(2015, 1, 1), "00"),
            ("L1", dt.date(2015, 2, 1), "XX"),
            ("L1", dt.date(2015, 3, 1), "01"),
            ("L1", dt.date(2015, 4, 1), "RA"),
        ],
        "loan_sequence_number string, monthly_reporting_period date, delinquency_status string",
    )
    out = prepare_performance(df, DQ).orderBy("reporting_period").collect()
    assert [r["delinquency_status"] for r in out] == ["00", "01", "RA"]
    assert [r["dq_bucket"] for r in out] == [0, 1, 4]
    assert [r["reporting_year"] for r in out] == [2015, 2015, 2015]


# --- cleaning and typing ------------------------------------------------------


def test_clean_strings_nulls_blanks_and_column_specific_sentinels(spark):
    df = spark.createDataFrame(
        [(" 9999 ", "9999", "  ", "720")],
        "credit_score string, current_upb string, channel string, other string",
    )
    sentinels = {"credit_score": ["9999"], "channel": ["9"]}
    row = clean_strings(df, df.columns, sentinels).first()
    assert row["credit_score"] is None  # sentinel, after trimming
    assert row["current_upb"] == "9999"  # same text, but no sentinel for this column
    assert row["channel"] is None  # blank
    assert row["other"] == "720"


def test_cast_columns_and_failure_counts(spark):
    fields = [
        ("score", "int", "s"),
        ("upb", "double", "u"),
        ("period", "yyyymm", "p"),
        ("code", "string", "c"),
    ]
    df = spark.createDataFrame(
        [
            ("720", "1000.50", "201503", "01"),
            ("abc", "2.5", "2015-03", "02"),
            (None, None, None, None),
        ],
        "score string, upb string, period string, code string",
    )
    df = df.withColumn(CORRUPT_COL, F.lit(None).cast("string"))
    typed = cast_columns(df, fields).first()
    assert typed["score"] == 720
    assert typed["upb"] == pytest.approx(1000.5)
    assert typed["period"] == dt.date(2015, 3, 1)
    assert typed["code"] == "01"  # categorical codes keep leading zeros

    prof = checks.raw_profile(df, fields)
    assert prof["rows"] == 3
    assert prof["cast_fail__score"] == 1
    assert prof["cast_fail__period"] == 1
    assert prof["cast_fail__upb"] == 0  # nulls are not failures


def test_read_pipe_files_flags_wrong_field_count(spark, tmp_path):
    path = tmp_path / "sample_orig_2015.txt"
    path.write_text('L1|720|"quoted"\nL2|650\nL3|700|x|extra\n')
    fields = [("id", "string", "i"), ("score", "string", "s"), ("note", "string", "n")]
    rows = {r["id"]: r for r in read_pipe_files(spark, [str(path)], fields, "|").collect()}
    assert rows["L1"]["note"] == '"quoted"'  # no quote handling
    assert rows["L1"][CORRUPT_COL] is None
    assert rows["L2"][CORRUPT_COL] is not None
    assert rows["L3"][CORRUPT_COL] is not None


# --- join -----------------------------------------------------------------------


def test_build_panel_surfaces_missing_origination(spark):
    perf = spark.createDataFrame(
        [("L1", 1), ("L2", 2)], "loan_sequence_number string, dq_bucket int"
    )
    orig = spark.createDataFrame([("L1", 2015)], "loan_sequence_number string, vintage int")
    out = {
        r["loan_sequence_number"]: r["vintage"]
        for r in build_panel(
            perf, orig, ["loan_sequence_number", "dq_bucket"], ["loan_sequence_number", "vintage"]
        ).collect()
    }
    assert out == {"L1": 2015, "L2": None}  # left join: unmatched rows kept, not dropped


def test_build_panel_rejects_column_retained_from_both_sides(spark):
    df = spark.createDataFrame([("L1", 1)], "loan_sequence_number string, x int")
    with pytest.raises(ValueError, match="both files"):
        build_panel(df, df, ["loan_sequence_number", "x"], ["loan_sequence_number", "x"])


# --- checks ---------------------------------------------------------------------


def test_duplicate_and_gap_checks(spark):
    df = spark.createDataFrame(
        [
            ("L1", dt.date(2015, 1, 1)),
            ("L1", dt.date(2015, 2, 1)),
            ("L1", dt.date(2015, 2, 1)),  # duplicate key
            ("L2", dt.date(2015, 1, 1)),
            ("L2", dt.date(2015, 3, 1)),  # gap
        ],
        "loan_sequence_number string, reporting_period date",
    )
    assert checks.duplicate_count(df, ["loan_sequence_number", "reporting_period"]) == 1
    assert checks.gap_summary(df.dropDuplicates()) == {"loans": 2, "loans_with_gaps": 1}
