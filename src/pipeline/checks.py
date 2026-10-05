"""Stage 1 acceptance checks. Each returns numbers; the stage script decides pass/fail."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.ingest import CORRUPT_COL, Field, cast_failure_exprs


def raw_profile(df: DataFrame, fields: Sequence[Field], extra: Sequence | None = None) -> dict:
    """Return row count, corrupt-row count, per-field cast failures and any ``extra`` aggregates.

    ``df`` is the cleaned string frame, before casting. One pass over the data.
    """
    aggs = [
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(CORRUPT_COL).isNotNull(), 1).otherwise(0)).alias("corrupt_rows"),
        *cast_failure_exprs(fields),
        *(extra or []),
    ]
    row = df.agg(*aggs).first()
    assert row is not None
    return {k: (v or 0) for k, v in row.asDict().items()}


def duplicate_count(df: DataFrame, keys: Sequence[str]) -> int:
    """Return how many key combinations appear more than once."""
    return df.groupBy(*keys).count().filter(F.col("count") > 1).count()


def dq_distribution(panel: DataFrame) -> pd.DataFrame:
    """Return row count and share of the panel per dq_bucket (REO shown separately)."""
    label = F.when(F.col("is_reo"), F.lit("REO")).otherwise(F.col("dq_bucket").cast("string"))
    pdf = panel.groupBy(label.alias("dq_bucket")).count().toPandas()
    pdf["share"] = pdf["count"] / pdf["count"].sum()
    order = {b: i for i, b in enumerate(["0", "1", "2", "3", "4", "REO"])}
    pdf["_o"] = pdf["dq_bucket"].map(order)
    return pdf.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def gap_summary(panel: DataFrame) -> dict[str, int]:
    """Return loan count and how many loans have a missing month inside their history."""
    per_loan = panel.groupBy("loan_sequence_number").agg(
        F.count(F.lit(1)).alias("n"),
        (
            F.months_between(F.max("reporting_period"), F.min("reporting_period")).cast("int") + 1
        ).alias("span"),
    )
    row = per_loan.agg(
        F.count(F.lit(1)).alias("loans"),
        F.sum(F.when(F.col("span") != F.col("n"), 1).otherwise(0)).alias("loans_with_gaps"),
    ).first()
    assert row is not None
    return {k: int(v or 0) for k, v in row.asDict().items()}


def spot_check_loans(panel: DataFrame, n: int, seed: int) -> list[pd.DataFrame]:
    """Return the full monthly sequence of ``n`` random loans that were ever delinquent.

    Sampled from ever-delinquent loans because a uniformly random loan is almost
    always current every month, which tells you nothing about the mapping.
    """
    ever_dq = (
        panel.groupBy("loan_sequence_number")
        .agg(F.max("dq_bucket").alias("worst"))
        .filter(F.col("worst") > 0)
        .select("loan_sequence_number")
    )
    ids = [r[0] for r in ever_dq.orderBy(F.rand(seed)).limit(n).collect()]
    cols = [
        "loan_sequence_number",
        "reporting_period",
        "delinquency_status",
        "dq_bucket",
        "current_upb",
        "zero_balance_code",
    ]
    rows = panel.filter(F.col("loan_sequence_number").isin(ids)).select(*cols).toPandas()
    return [
        g.sort_values("reporting_period").reset_index(drop=True)
        for _, g in rows.groupby("loan_sequence_number")
    ]
