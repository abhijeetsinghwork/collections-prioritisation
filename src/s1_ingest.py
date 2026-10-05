"""Stage 1 - ingest raw SFLLD sample files into a partitioned Parquet panel.

Run: python -m src.s1_ingest [--vintages 2015 2016 ...]
Writes data/interim/panel (partitioned by reporting_year) and, only if every
acceptance check passes, a .done marker next to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from src.pipeline import checks
from src.pipeline.ingest import (
    build_panel,
    cast_columns,
    clean_strings,
    prepare_performance,
    read_pipe_files,
    with_vintage,
)
from src.spark import get_spark
from src.utils.config import Config, load_config
from src.utils.schema import ORIGINATION_FIELDS, PERFORMANCE_FIELDS


def done_marker(cfg: Config) -> Path:
    """Return the path of the marker written after a fully checked ingest."""
    return cfg.paths.panel_dir.parent / ".s1_ingest.done"


def input_paths(cfg: Config, vintages: list[int]) -> tuple[list[str], list[str]]:
    """Return origination and performance file paths for ``vintages``, failing if any is missing."""
    orig = [cfg.paths.raw_dir / cfg.paths.orig_glob.format(vintage=v) for v in vintages]
    perf = [cfg.paths.raw_dir / cfg.paths.perf_glob.format(vintage=v) for v in vintages]
    missing = [str(p) for p in orig + perf if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw files: {missing}")
    return [str(p) for p in orig], [str(p) for p in perf]


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def run(cfg: Config, vintages: list[int]) -> bool:
    """Ingest ``vintages``, write the panel, run acceptance checks. Return True if all pass."""
    spark = get_spark(cfg.spark, "s1_ingest")
    orig_paths, perf_paths = input_paths(cfg, vintages)
    ing = cfg.ingest
    dq = ing.delinquency
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    orig_names = [n for n, _, _ in ORIGINATION_FIELDS]
    perf_names = [n for n, _, _ in PERFORMANCE_FIELDS]
    orig_clean = clean_strings(
        read_pipe_files(spark, orig_paths, ORIGINATION_FIELDS, ing.delimiter),
        orig_names,
        ing.sentinels,
    )
    perf_clean = clean_strings(
        read_pipe_files(spark, perf_paths, PERFORMANCE_FIELDS, ing.delimiter),
        perf_names,
        ing.sentinels,
    )

    section(f"Raw profile - vintages {vintages}")
    status = F.col("delinquency_status")
    recognised = status.rlike(r"^\d+$") | status.isin(dq.reo_code, dq.unknown_code)
    orig_prof = checks.raw_profile(
        orig_clean,
        ORIGINATION_FIELDS,
        [F.countDistinct("loan_sequence_number").alias("distinct_loans")],
    )
    perf_prof = checks.raw_profile(
        perf_clean,
        PERFORMANCE_FIELDS,
        [
            F.sum(F.when(status == dq.unknown_code, 1).otherwise(0)).alias("xx_rows"),
            F.sum(F.when(status == dq.reo_code, 1).otherwise(0)).alias("reo_rows"),
            F.sum(F.when(status.isNull() | ~recognised, 1).otherwise(0)).alias("unrecognised"),
        ],
    )
    print(f"origination rows {orig_prof['rows']:,}  distinct loans {orig_prof['distinct_loans']:,}")
    print(
        f"performance rows {perf_prof['rows']:,}  XX rows {perf_prof['xx_rows']:,}  "
        f"RA rows {perf_prof['reo_rows']:,}"
    )
    require(orig_prof["corrupt_rows"] == 0, "origination: no rows with wrong field count")
    require(perf_prof["corrupt_rows"] == 0, "performance: no rows with wrong field count")
    require(orig_prof["rows"] == orig_prof["distinct_loans"], "origination: one row per loan")
    require(perf_prof["unrecognised"] == 0, "performance: every delinquency status recognised")
    for prof, label in ((orig_prof, "origination"), (perf_prof, "performance")):
        bad = {k: v for k, v in prof.items() if k.startswith("cast_fail__") and v}
        require(not bad, f"{label}: every non-null value casts to its type {bad or ''}")

    if failures:  # do not write a panel built on unparseable input
        return False

    orig = with_vintage(cast_columns(orig_clean, ORIGINATION_FIELDS))
    perf = prepare_performance(cast_columns(perf_clean, PERFORMANCE_FIELDS), dq)
    panel = build_panel(perf, orig, ing.retain_performance, ing.retain_origination)

    section("Write")
    out = str(cfg.paths.panel_dir)
    (
        panel.repartition("reporting_year")
        .sortWithinPartitions("loan_sequence_number", "reporting_period")
        .write.mode("overwrite")
        .partitionBy("reporting_year")
        .parquet(out)
    )
    print(f"wrote {out}")

    written = spark.read.parquet(out)

    section("Acceptance checks")
    expected = perf_prof["rows"] - perf_prof["xx_rows"]
    actual = written.count()
    print(
        f"raw performance rows {perf_prof['rows']:,} - XX rows {perf_prof['xx_rows']:,} "
        f"= {expected:,};  panel rows {actual:,}"
    )
    require(actual == expected, "panel rows == raw performance rows - XX rows")

    unmatched = written.filter(F.col("vintage").isNull()).count()
    require(
        unmatched == 0, f"every performance row has an origination record ({unmatched:,} unmatched)"
    )

    dupes = checks.duplicate_count(written, ["loan_sequence_number", "reporting_period"])
    require(dupes == 0, f"(loan_sequence_number, reporting_period) unique ({dupes:,} duplicates)")

    null_bucket = written.filter(F.col("dq_bucket").isNull()).count()
    require(null_bucket == 0, f"no null dq_bucket after XX removal ({null_bucket:,})")

    section("dq_bucket distribution")
    dist = checks.dq_distribution(written)
    print(
        dist.to_string(index=False, formatters={"count": "{:,}".format, "share": "{:.4%}".format})
    )
    current = float(dist.loc[dist["dq_bucket"] == "0", "share"].sum())
    require(
        current > cfg.checks.min_current_share,
        f"current share {current:.4%} > {cfg.checks.min_current_share:.0%}",
    )

    section("Month contiguity")
    gaps = checks.gap_summary(written)
    print(
        f"loans {gaps['loans']:,}; loans with a missing month in their history "
        f"{gaps['loans_with_gaps']:,}  (XX removal creates gaps by design)"
    )

    section(f"Spot check - {cfg.checks.spot_check_loans} random ever-delinquent loans")
    with pd.option_context("display.max_rows", 500, "display.width", 120):
        for seq in checks.spot_check_loans(written, cfg.checks.spot_check_loans, cfg.random_seed):
            print(f"\n{seq['loan_sequence_number'].iloc[0]}  ({len(seq)} months)")
            print(seq.drop(columns="loan_sequence_number").to_string(index=False))

    spark.stop()
    return not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintages", type=int, nargs="+", help="override config vintages")
    args = parser.parse_args(argv)
    cfg = load_config()
    vintages = args.vintages or cfg.ingest.vintages

    marker = done_marker(cfg)
    marker.unlink(missing_ok=True)  # a failed or partial run must never look complete
    ok = run(cfg, vintages)
    section("Result")
    if ok:
        marker.write_text(" ".join(map(str, vintages)) + "\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
        return 0
    print("CHECKS FAILED - no .done marker written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
