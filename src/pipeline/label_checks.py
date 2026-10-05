"""Stage 2 acceptance checks. Each returns numbers or frames; the stage script decides."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.pipeline.labels import LABEL, REASON


def reason_counts(labelled: DataFrame) -> pd.DataFrame:
    """Return population rows per label_reason, with the label each reason implies."""
    pdf = labelled.groupBy(REASON, LABEL).count().orderBy(F.col("count").desc()).toPandas()
    pdf["share"] = pdf["count"] / pdf["count"].sum()
    return pdf


def split_overlaps(df: DataFrame, split_names: Sequence[str]) -> dict[tuple[str, str], int]:
    """Return the number of loans shared by each pair of splits."""
    loans = df.select("loan_sequence_number", "split").distinct()
    out: dict[tuple[str, str], int] = {}
    for a, b in combinations(split_names, 2):
        la = loans.filter(F.col("split") == a).select("loan_sequence_number")
        lb = loans.filter(F.col("split") == b).select("loan_sequence_number")
        out[(a, b)] = la.join(lb, "loan_sequence_number").count()
    return out


def split_summary(df: DataFrame, split_names: Sequence[str]) -> pd.DataFrame:
    """Return rows, loans, base rate and period range per split, in split order."""
    pdf = (
        df.groupBy("split")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("loan_sequence_number").alias("loans"),
            F.avg(LABEL).alias("base_rate"),
            F.min("reporting_period").alias("min_period"),
            F.max("reporting_period").alias("max_period"),
        )
        .toPandas()
    )
    pdf["_o"] = pdf["split"].map({n: i for i, n in enumerate(split_names)})
    return pdf.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def base_rate_by_bucket(df: DataFrame, split_names: Sequence[str]) -> pd.DataFrame:
    """Return base rate of the label per split (rows) and dq_bucket at T (columns)."""
    pdf = df.groupBy("split", "dq_bucket").agg(F.avg(LABEL).alias("rate")).toPandas()
    wide = pdf.pivot_table(index="split", columns="dq_bucket", values="rate")
    return wide.reindex([n for n in split_names if n in wide.index])


def is_strictly_increasing(values: Sequence[float]) -> bool:
    """Return True if every value is larger than the one before it."""
    return all(b > a for a, b in zip(values, values[1:], strict=False))


def trace_rows(
    df: DataFrame, panel: DataFrame, label: int, n: int, seed: int, horizon: int
) -> list[tuple[dict, pd.DataFrame]]:
    """Return ``n`` random labelled rows with label ``label`` and the panel around each.

    Each trace shows months T-2 .. T+horizon so the label can be checked by eye.
    """
    picks = (
        df.filter(F.col(LABEL) == label)
        .orderBy(F.rand(seed))
        .limit(n)
        .select("loan_sequence_number", "reporting_period", "dq_bucket", REASON)
        .collect()
    )
    cols = [
        "reporting_period",
        "delinquency_status",
        "dq_bucket",
        "current_upb",
        "zero_balance_code",
    ]
    out = []
    for p in picks:
        t = F.lit(p["reporting_period"])
        window = panel.filter(
            (F.col("loan_sequence_number") == p["loan_sequence_number"])
            & (F.col("reporting_period") >= F.add_months(t, -2))
            & (F.col("reporting_period") <= F.add_months(t, horizon))
        )
        seq = window.select(*cols).orderBy("reporting_period").toPandas()
        out.append((p.asDict(), seq))
    return out
