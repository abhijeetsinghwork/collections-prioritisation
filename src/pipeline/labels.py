"""Stage 2 transforms: population, forward-looking label, censoring, disjoint splits.

This is the only module allowed to read months after T, and it does so only to
build the label. Windows are defined on a calendar month index, not on row
offsets, because the source panel has missing months.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from src.utils.config import LabelsConfig, SplitConfig

LABEL = "rolls_deeper"
REASON = "label_reason"

# label_reason values
DEEPER_BUCKET = "deeper_bucket"
ROLL_ZBC = "roll_zero_balance"
CURE_ZBC = "cure_zero_balance"
NO_ROLL = "no_roll_full_window"
EXCLUDED_ZBC = "excluded_zero_balance"
INCOMPLETE = "incomplete_window"
END_OF_DATA = "end_of_data"

DROPPED_REASONS = (EXCLUDED_ZBC, INCOMPLETE, END_OF_DATA)


def month_index(period: Column) -> Column:
    """Return a calendar month count (year * 12 + month - 1) for a first-of-month date."""
    return F.year(period) * 12 + F.month(period) - 1


def in_population(buckets: Sequence[int]) -> Column:
    """Return the predicate for loan-months in the collections population at T.

    Delinquent in one of ``buckets``, not REO, and not terminating this month.
    """
    return (
        F.col("dq_bucket").isin(list(buckets))
        & ~F.coalesce(F.col("is_reo"), F.lit(False))
        & F.col("zero_balance_code").isNull()
    )


def with_future_window(panel: DataFrame, cfg: LabelsConfig) -> DataFrame:
    """Return ``panel`` with aggregates over months T+1 .. T+horizon of the same loan.

    Reads only the future window, never month T itself, and only to build the label.
    """
    w = (
        Window.partitionBy("loan_sequence_number")
        .orderBy("month_idx")
        .rangeBetween(1, cfg.horizon_months)
    )
    zbc = F.col("zero_balance_code")

    def any_code(kind: str) -> Column:
        return F.max(zbc.isin(cfg.codes(kind)).cast("int")).over(w)

    return (
        panel.withColumn("month_idx", month_index(F.col("reporting_period")))
        .withColumn("future_max_bucket", F.max("dq_bucket").over(w))
        .withColumn("future_months", F.count(F.lit(1)).over(w))
        .withColumn("future_roll_zbc", any_code("roll"))
        .withColumn("future_cure_zbc", any_code("cure"))
        .withColumn("future_exclude_zbc", any_code("exclude"))
    )


def label_reason(cfg: LabelsConfig, last_labellable: dt.date) -> Column:
    """Return the label_reason for a population row with future-window columns.

    Order matters: a roll seen inside the window is an observed outcome even if
    the window is otherwise incomplete; a non-roll needs evidence for all of it.
    """
    roll_bucket = F.coalesce(F.col("future_max_bucket"), F.lit(-1)) > F.col("dq_bucket")
    flag = lambda c: F.coalesce(F.col(c), F.lit(0)) == 1  # noqa: E731
    return (
        F.when(F.col("reporting_period") > F.lit(last_labellable), F.lit(END_OF_DATA))
        .when(roll_bucket, F.lit(DEEPER_BUCKET))
        .when(flag("future_roll_zbc"), F.lit(ROLL_ZBC))
        .when(flag("future_exclude_zbc"), F.lit(EXCLUDED_ZBC))
        .when(flag("future_cure_zbc"), F.lit(CURE_ZBC))
        .when(F.col("future_months") == cfg.horizon_months, F.lit(NO_ROLL))
        .otherwise(F.lit(INCOMPLETE))
    )


def label_from_reason(reason: Column) -> Column:
    """Return 1 for a roll, 0 for an observed non-roll, null for a dropped row."""
    return (
        F.when(reason.isin(DEEPER_BUCKET, ROLL_ZBC), F.lit(1))
        .when(reason.isin(CURE_ZBC, NO_ROLL), F.lit(0))
        .cast("int")
    )


def last_labellable_period(max_period: dt.date, horizon_months: int) -> dt.date:
    """Return the last month T whose full label window lies inside the data."""
    idx = max_period.year * 12 + max_period.month - 1 - horizon_months
    return dt.date(idx // 12, idx % 12 + 1, 1)


def build_labels(panel: DataFrame, cfg: LabelsConfig, max_period: dt.date) -> DataFrame:
    """Return every population row with label_reason and rolls_deeper (null if dropped).

    Only loans that are ever in the population are windowed, to keep the shuffle small.
    """
    pop = in_population(cfg.population_buckets)
    loans = panel.filter(pop).select("loan_sequence_number").distinct()
    windowed = with_future_window(panel.join(loans, "loan_sequence_number", "left_semi"), cfg)
    reason = label_reason(cfg, last_labellable_period(max_period, cfg.horizon_months))
    return (
        windowed.filter(pop)
        .withColumn(REASON, reason)
        .withColumn(LABEL, label_from_reason(F.col(REASON)))
    )


def split_column(splits: Sequence[SplitConfig]) -> Column:
    """Return the split name for each row's reporting_period, or null if outside all splits."""
    period = F.col("reporting_period")
    expr: Column | None = None
    for s in splits:
        cond = period.between(F.lit(s.start_date), F.lit(s.end_date))
        expr = F.when(cond, F.lit(s.name)) if expr is None else expr.when(cond, F.lit(s.name))
    assert expr is not None
    return expr


def assign_disjoint_splits(labelled: DataFrame, splits: Sequence[SplitConfig]) -> DataFrame:
    """Return labelled rows with a ``split`` column, keeping each loan in one split only.

    A loan stays in the earliest split it has a labelled row in; its rows in later
    splits are dropped. A temporal cut alone would let a long-lived loan straddle
    the boundary and be memorised in train, then scored in test.
    """
    order = {s.name: i for i, s in enumerate(splits)}
    order_col = F.create_map(*[x for kv in order.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])
    with_split = labelled.withColumn("split", split_column(splits)).filter(
        F.col("split").isNotNull()
    )
    with_order = with_split.withColumn("split_order", order_col[F.col("split")])
    first = with_order.groupBy("loan_sequence_number").agg(
        F.min("split_order").alias("first_split_order")
    )
    return (
        with_order.join(first, "loan_sequence_number")
        .filter(F.col("split_order") == F.col("first_split_order"))
        .drop("split_order", "first_split_order")
    )
