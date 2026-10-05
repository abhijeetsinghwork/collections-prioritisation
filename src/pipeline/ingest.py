"""Stage 1 transforms: raw pipe-delimited SFLLD text -> typed, cleaned panel rows.

Every function here works on a single month's row in isolation (no windows,
no cross-row reads), so nothing in Stage 1 can look across time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from src.utils.config import DelinquencyConfig

CORRUPT_COL = "_corrupt_record"
Field = tuple[str, str, str]  # (column, kind, layout attribute name)


def string_schema(fields: Sequence[Field]) -> StructType:
    """Return an all-string schema in file order, plus a corrupt-record column.

    Reading everything as text first makes column positions explicit (never
    inferred) and lets type-cast failures be counted rather than silently nulled.
    """
    cols = [StructField(name, StringType(), True) for name, _, _ in fields]
    return StructType([*cols, StructField(CORRUPT_COL, StringType(), True)])


def read_pipe_files(
    spark: SparkSession, paths: Sequence[str], fields: Sequence[Field], delimiter: str
) -> DataFrame:
    """Return the raw rows of headerless delimited files as strings, with source file."""
    return (
        spark.read.option("sep", delimiter)
        .option("header", "false")
        .option("quote", "")  # SFLLD fields are never quoted
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COL)
        .schema(string_schema(fields))
        .csv(list(paths))
        .withColumn("_source_file", F.input_file_name())
    )


def clean_strings(
    df: DataFrame, columns: Iterable[str], sentinels: dict[str, list[str]]
) -> DataFrame:
    """Return ``df`` with the given columns trimmed, and blanks and sentinels set to null."""
    out = df
    for name in columns:
        trimmed = F.trim(F.col(name))
        is_missing = trimmed == ""
        if name in sentinels:
            is_missing = is_missing | trimmed.isin(sentinels[name])
        out = out.withColumn(name, F.when(is_missing, F.lit(None)).otherwise(trimmed))
    return out


def cast_expr(col: Column, kind: str) -> Column:
    """Return ``col`` (a cleaned string) cast to the type named by ``kind``."""
    if kind == "string":
        return col
    if kind == "int":
        return col.cast("int")
    if kind == "double":
        return col.cast("double")
    if kind == "yyyymm":
        return F.to_date(col, "yyyyMM")  # first day of the month
    raise ValueError(f"unknown ingest kind: {kind}")


def cast_columns(df: DataFrame, fields: Sequence[Field]) -> DataFrame:
    """Return ``df`` with every schema field cast from string to its declared kind."""
    out = df
    for name, kind, _ in fields:
        out = out.withColumn(name, cast_expr(F.col(name), kind))
    return out


def cast_failure_exprs(fields: Sequence[Field]) -> list[Column]:
    """Return aggregate columns counting non-null strings that fail their cast, per field."""
    return [
        F.sum(
            F.when(F.col(name).isNotNull() & cast_expr(F.col(name), kind).isNull(), 1).otherwise(0)
        ).alias(f"cast_fail__{name}")
        for name, kind, _ in fields
        if kind != "string"
    ]


def delinquency_columns(status: Column, cfg: DelinquencyConfig) -> dict[str, Column]:
    """Return dq_months, dq_bucket and is_reo derived from the raw status string.

    - numeric codes ("00", "01", ... up to the documented cap of 99) give
      dq_months; dq_bucket caps it at ``cfg.bucket_cap`` ("4+").
    - the REO code is terminal: dq_months null, dq_bucket ``cfg.reo_bucket``.
    - the unknown code, blanks and anything unrecognised give all-null
      (callers exclude those rows; they are never coerced to current).
    """
    is_numeric = status.rlike(r"^\d+$")
    is_reo = status == cfg.reo_code
    months = F.when(is_numeric, status.cast("int"))
    bucket = (
        F.when(is_numeric, F.least(months, F.lit(cfg.bucket_cap)))
        .when(is_reo, F.lit(cfg.reo_bucket))
        .cast("int")
    )
    return {
        "dq_months": months,
        "dq_bucket": bucket,
        "is_reo": F.when(is_numeric | is_reo, is_reo),
    }


def with_vintage(df: DataFrame) -> DataFrame:
    """Return ``df`` with the integer origination vintage taken from its file name."""
    year = F.regexp_extract(F.col("_source_file"), r"sample_orig_(\d{4})", 1)
    return df.withColumn("vintage", F.when(year != "", year.cast("int")))


def prepare_performance(perf_typed: DataFrame, cfg: DelinquencyConfig) -> DataFrame:
    """Return performance rows with period columns and delinquency mapping, XX rows removed."""
    dq = delinquency_columns(F.col("delinquency_status"), cfg)
    return (
        perf_typed.filter(
            F.col("delinquency_status").isNull() | (F.col("delinquency_status") != cfg.unknown_code)
        )
        .withColumnRenamed("monthly_reporting_period", "reporting_period")
        .withColumn("reporting_year", F.year("reporting_period"))
        .withColumns(dq)
    )


def build_panel(
    perf: DataFrame,
    orig: DataFrame,
    retain_performance: Sequence[str],
    retain_origination: Sequence[str],
) -> DataFrame:
    """Return performance rows left-joined to (broadcast) origination attributes.

    A left join so that a performance row with no origination record surfaces
    as a null ``vintage`` and is caught by the checks, instead of vanishing.
    """
    overlap = (set(retain_origination) & set(retain_performance)) - {"loan_sequence_number"}
    if overlap:
        raise ValueError(f"columns retained from both files: {sorted(overlap)}")
    return perf.select(*retain_performance).join(
        F.broadcast(orig.select(*retain_origination)), on="loan_sequence_number", how="left"
    )
