"""Stage 2 - population, 3-month roll label, censoring and disjoint temporal splits.

Run: python -m src.s2_labels
Reads the Stage 1 panel, writes data/processed/labels (partitioned by split)
and, only if every acceptance check passes, a .done marker next to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from src.pipeline import label_checks as lc
from src.pipeline.labels import LABEL, REASON, assign_disjoint_splits, build_labels
from src.spark import get_spark
from src.utils.config import Config, load_config

OUTPUT_COLUMNS = [
    "loan_sequence_number",
    "reporting_period",
    "vintage",
    "dq_bucket",
    "current_upb",
    LABEL,
    REASON,
    "split",
]


def done_marker(cfg: Config) -> Path:
    """Return the path of the marker written after a fully checked labels run."""
    return cfg.paths.labels_dir.parent / ".s2_labels.done"


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def run(cfg: Config) -> bool:
    """Build labels and splits, write them, run acceptance checks. Return True if all pass."""
    spark = get_spark(cfg.spark, "s2_labels")
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    panel = spark.read.parquet(str(cfg.paths.panel_dir))
    split_names = [s.name for s in cfg.splits]

    section("Inputs")
    max_row = panel.agg(F.max("reporting_period")).first()
    assert max_row is not None and max_row[0] is not None, "panel is empty"
    max_period = max_row[0]
    print(f"panel max reporting_period {max_period}")
    known = set(cfg.labels.zero_balance_codes)
    seen = {r[0] for r in panel.select("zero_balance_code").distinct().collect() if r[0]}
    require(
        seen <= known,
        f"every zero-balance code is classified (unclassified: {sorted(seen - known)})",
    )
    if failures:
        return False

    labelled = build_labels(panel, cfg.labels, max_period).cache()

    section("Population and censoring")
    reasons = lc.reason_counts(labelled)
    print(
        reasons.to_string(
            index=False, formatters={"count": "{:,}".format, "share": "{:.2%}".format}
        )
    )

    kept = labelled.filter(F.col(LABEL).isNotNull())
    with_split = assign_disjoint_splits(kept, cfg.splits)
    n_kept = kept.count()
    n_outside = kept.filter(
        ~F.col("reporting_period").between(cfg.splits[0].start_date, cfg.splits[-1].end_date)
    ).count()
    print(f"\nlabelled rows {n_kept:,}; outside every split window {n_outside:,}")

    section("Write")
    out = str(cfg.paths.labels_dir)
    (
        with_split.select(*OUTPUT_COLUMNS)
        .repartition("split")
        .sortWithinPartitions("loan_sequence_number", "reporting_period")
        .write.mode("overwrite")
        .partitionBy("split")
        .parquet(out)
    )
    print(f"wrote {out}")
    written = spark.read.parquet(out)
    labelled.unpersist()

    section("Splits")
    summary = lc.split_summary(written, split_names)
    print(
        summary.to_string(
            index=False,
            formatters={
                "rows": "{:,}".format,
                "loans": "{:,}".format,
                "base_rate": "{:.4f}".format,
            },
        )
    )
    n_in_splits = int(summary["rows"].sum())
    n_dropped_overlap = n_kept - n_outside - n_in_splits
    print(f"rows dropped to keep each loan in a single split: {n_dropped_overlap:,}")

    for name in split_names:
        require(name in set(summary["split"]), f"split {name} is present")
    lo, hi = cfg.checks.base_rate_bounds
    for rec in summary.to_dict("records"):
        split, rows, rate = str(rec["split"]), int(rec["rows"]), float(rec["base_rate"])
        require(
            rows >= cfg.checks.min_rows_per_split,
            f"{split}: {rows:,} rows >= {cfg.checks.min_rows_per_split:,}",
        )
        require(lo < rate < hi, f"{split}: base rate {rate:.4f} in ({lo}, {hi})")
    ranges = list(zip(summary["min_period"], summary["max_period"], strict=True))
    require(
        all(prev[1] < nxt[0] for prev, nxt in zip(ranges, ranges[1:], strict=False)),
        "split period ranges are ordered and do not overlap",
    )

    overlaps = lc.split_overlaps(written, split_names)
    shared = {f"{a}&{b}": n for (a, b), n in overlaps.items() if n}
    require(not shared, f"no loan appears in two splits {shared or ''}")
    require(overlaps[("train", "test")] == 0, "train and test loans disjoint")

    section("Base rate by split and dq_bucket at T")
    rates = lc.base_rate_by_bucket(written, split_names)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(rates.to_string())
    for name in split_names:
        if name not in rates.index:
            continue  # absence already failed above
        rising = lc.is_strictly_increasing([float(v) for v in rates.loc[name].to_numpy()])
        if name in cfg.checks.monotonic_splits:
            require(rising, f"{name}: base rate rises with bucket depth")
        else:
            print(f"INFO  {name}: base rate rises with bucket depth = {rising} (not asserted)")

    section("Label traces (months T-2 .. T+horizon)")
    for label in (1, 0):
        for pick, seq in lc.trace_rows(
            written,
            panel,
            label,
            cfg.checks.trace_examples_per_label,
            cfg.random_seed,
            cfg.labels.horizon_months,
        ):
            print(
                f"\n{pick['loan_sequence_number']}  T={pick['reporting_period']}  "
                f"bucket@T={pick['dq_bucket']}  {LABEL}={label}  ({pick[REASON]})"
            )
            print(seq.to_string(index=False))

    spark.stop()
    return not failures


def main() -> int:
    cfg = load_config()
    marker = done_marker(cfg)
    marker.unlink(missing_ok=True)  # a failed or partial run must never look complete
    ok = run(cfg)
    section("Result")
    if ok:
        marker.write_text("ok\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
        return 0
    print("CHECKS FAILED - no .done marker written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
