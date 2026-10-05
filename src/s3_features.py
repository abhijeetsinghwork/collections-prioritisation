"""Stage 3 - features (families A-D), leakage guard, null-rate and single-feature AUC audit.

Run: python -m src.s3_features [--audit-only]
Writes data/processed/features (partitioned by split) and the audit table, and,
only if every check passes, a .done marker. A candidate feature without an
explicit keep/drop decision in config.yaml fails the stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.pipeline import feature_audit
from src.pipeline.features import KEY, build_loan_month_features, feature_registry
from src.pipeline.labels import LABEL
from src.pipeline.macro import national_macro_table, state_macro_table
from src.spark import get_spark
from src.utils import leakage
from src.utils.config import Config, load_config

ID_COLUMNS = [*KEY, "split", "vintage", LABEL, "exposure"]


def done_marker(cfg: Config) -> Path:
    """Return the path of the marker written after a fully checked features run."""
    return cfg.paths.features_dir.parent / ".s3_features.done"


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def macro_frames(
    spark: SparkSession, cfg: Config, states: list[str]
) -> tuple[DataFrame, DataFrame]:
    """Return the lagged state and national macro tables as small Spark frames."""
    m = cfg.features.macro
    state = state_macro_table(cfg.paths.macro_dir, states, m)
    national = national_macro_table(cfg.paths.macro_dir, m)
    for df in (state, national):
        df["reporting_period"] = pd.to_datetime(df["reporting_period"]).dt.date
    return spark.createDataFrame(state), spark.createDataFrame(national)


def build(cfg: Config) -> None:
    """Compute every candidate feature for every labelled row and write the feature table."""
    spark = get_spark(cfg.spark, "s3_features")
    registry = feature_registry(cfg.features)
    panel = spark.read.parquet(str(cfg.paths.panel_dir))
    labels = spark.read.parquet(str(cfg.paths.labels_dir)).select(
        *KEY, "split", LABEL, F.col("current_upb").alias("exposure")
    )

    loans = labels.select("loan_sequence_number").distinct()
    history = panel.join(loans, "loan_sequence_number", "left_semi")
    loan_months = build_loan_month_features(history, cfg.features)

    states = [r[0] for r in panel.select("property_state").distinct().collect() if r[0]]
    state_macro, national_macro = macro_frames(spark, cfg, states)

    non_macro = [n for n, m in registry.items() if m.family != "D_macro"]
    table = (
        labels.join(loan_months.select(*KEY, "vintage", *non_macro), KEY, "inner")
        .join(F.broadcast(state_macro), ["property_state", "reporting_period"], "left")
        .join(F.broadcast(national_macro), ["reporting_period"], "left")
        .select(*ID_COLUMNS, *registry)
    )
    out = str(cfg.paths.features_dir)
    (
        table.repartition("split")
        .sortWithinPartitions(*KEY)
        .write.mode("overwrite")
        .partitionBy("split")
        .parquet(out)
    )
    print(f"wrote {out}")
    spark.stop()


def read_split(cfg: Config, split: str) -> pd.DataFrame:
    """Return one split of the feature table as pandas."""
    dataset = ds.dataset(cfg.paths.features_dir, format="parquet", partitioning="hive")
    return dataset.to_table(filter=ds.field("split") == split).to_pandas()


def audit(cfg: Config) -> bool:
    """Run the leakage guard and audit on the written table. Return True if all checks pass."""
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    registry = feature_registry(cfg.features)
    candidates = list(registry)
    train, valid = read_split(cfg, "train"), read_split(cfg, "validation")
    labels_rows = ds.dataset(
        cfg.paths.labels_dir, format="parquet", partitioning="hive"
    ).count_rows()
    feature_rows = ds.dataset(
        cfg.paths.features_dir, format="parquet", partitioning="hive"
    ).count_rows()

    section("Coverage")
    print(
        f"labelled rows {labels_rows:,}; feature rows {feature_rows:,}; "
        f"candidates {len(candidates)}"
    )
    require(feature_rows == labels_rows, "every labelled row has a feature row")
    by_family = pd.Series({n: m.family for n, m in registry.items()}).value_counts().sort_index()
    print(by_family.to_string())

    section("Leakage guard: blocklist and declared windows")
    try:
        leakage.check_registry(registry)
        leakage.check_features_registered(
            [c for c in train.columns if c not in ID_COLUMNS and c != "split"], registry
        )
        leakage.check_columns(candidates)
        require(True, "no feature name or source matches the blocklist")
        require(
            True, "every feature's window ends at T or earlier (trajectory: T-1 unless declared)"
        )
    except leakage.LeakageError as err:
        require(False, f"leakage guard: {err}")
    reads_t = sorted(
        n for n, m in registry.items() if m.family == "C_trajectory" and m.window_end == 0
    )
    print(f"trajectory features that read month T by definition: {reads_t}")

    section("Null rate and single-feature AUC (fit on train, scored on validation)")
    table = feature_audit.audit_table(train, valid, candidates, registry, LABEL, cfg.features.audit)
    cfg.paths.feature_audit_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.feature_audit_table, index=False, float_format="%.4f")
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(table.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nwrote {cfg.paths.feature_audit_table}")

    section("Explicit keep/drop decisions")
    decisions = cfg.features.decisions
    missing = [c for c in candidates if c not in decisions]
    unknown = sorted(set(decisions) - set(candidates))
    require(not unknown, f"decisions name only real candidates {unknown or ''}")
    require(not missing, f"every candidate has a keep/drop decision ({len(missing)} missing)")
    if missing:
        print("\nadd to config.yaml features.decisions:")
        for c in missing:
            print(f'    {c}: {{keep: true, reason: "..."}}')
    else:
        kept = cfg.features.kept()
        print(
            f"kept {len(kept)} of {len(candidates)}; dropped: "
            f"{[c for c in candidates if not decisions[c].keep]}"
        )
    return not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="reuse the written feature table")
    args = parser.parse_args(argv)
    cfg = load_config()
    marker = done_marker(cfg)
    marker.unlink(missing_ok=True)
    if not args.audit_only:
        build(cfg)
    ok = audit(cfg)
    section("Result")
    if ok:
        marker.write_text("ok\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
        return 0
    print("CHECKS FAILED - no .done marker written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
