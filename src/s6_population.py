"""Stage 6 - the monitoring population: every delinquent account-month in the drift windows.

Run:
  python -m src.s6_population

Stage 2 keeps each loan in the first split it appears in, which is right for
training and testing but makes every later window start with first-time
delinquents only (see methodology, Stage 6). A deployed model scores every
delinquent account, so drift is monitored on that population instead: the
same labels (Stage 2) and features (Stage 3) computed with the same functions,
without the one-split-per-loan rule. ``first_split`` records the split each
loan first appeared in, so the two populations can be compared.

Writes only under ``paths.monitor_features_dir``; Stages 2-4 outputs are untouched.
"""

from __future__ import annotations

import sys

from pyspark.sql import functions as F

from src.pipeline.features import KEY, build_loan_month_features, feature_registry
from src.pipeline.labels import LABEL, build_labels, split_column
from src.s3_features import ID_COLUMNS, macro_frames
from src.s4_models import section
from src.spark import get_spark
from src.utils.config import Config, load_config


def build(cfg: Config) -> int:
    """Write the monitoring feature table and return its row count."""
    spark = get_spark(cfg.spark, "s6_population")
    registry = feature_registry(cfg.features)
    panel = spark.read.parquet(str(cfg.paths.panel_dir))
    max_row = panel.agg(F.max("reporting_period")).first()
    assert max_row is not None and max_row[0] is not None, "panel is empty"

    labelled = (
        build_labels(panel, cfg.labels, max_row[0])
        .filter(F.col(LABEL).isNotNull())
        .withColumn("split", split_column(cfg.splits))
        .filter(F.col("split").isNotNull())
    )
    order = {s.name: i for i, s in enumerate(cfg.splits)}
    order_col = F.create_map(*[x for kv in order.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])
    names = F.create_map(*[x for kv in order.items() for x in (F.lit(kv[1]), F.lit(kv[0]))])
    first = (
        labelled.groupBy("loan_sequence_number")
        .agg(F.min(order_col[F.col("split")]).alias("first_order"))
        .select("loan_sequence_number", names[F.col("first_order")].alias("first_split"))
    )
    monitored = (
        labelled.filter(F.col("split").isin(cfg.drift.monitored_splits))
        .join(first, "loan_sequence_number")
        .select(*KEY, "split", "first_split", LABEL, F.col("current_upb").alias("exposure"))
    )

    loans = monitored.select("loan_sequence_number").distinct()
    history = panel.join(loans, "loan_sequence_number", "left_semi")
    loan_months = build_loan_month_features(history, cfg.features)
    states = [r[0] for r in panel.select("property_state").distinct().collect() if r[0]]
    state_macro, national_macro = macro_frames(spark, cfg, states)
    non_macro = [n for n, m in registry.items() if m.family != "D_macro"]
    table = (
        monitored.join(loan_months.select(*KEY, "vintage", *non_macro), KEY, "inner")
        .join(F.broadcast(state_macro), ["property_state", "reporting_period"], "left")
        .join(F.broadcast(national_macro), ["reporting_period"], "left")
        .select(*ID_COLUMNS, "first_split", *registry)
    )
    out = str(cfg.paths.monitor_features_dir)
    (
        table.repartition("split")
        .sortWithinPartitions(*KEY)
        .write.mode("overwrite")
        .partitionBy("split")
        .parquet(out)
    )
    written = spark.read.parquet(out)
    n_monitored, n_written = monitored.count(), written.count()
    section("Monitoring population")
    summary = (
        written.groupBy("split", "first_split").count().orderBy("split", "first_split").toPandas()
    )
    print(summary.to_string(index=False))
    print(f"labelled monitoring rows {n_monitored:,}; feature rows {n_written:,}; wrote {out}")
    spark.stop()
    if n_written != n_monitored:
        print("FAIL  every monitoring row has a feature row")
        return -1
    return n_written


def main() -> int:
    cfg = load_config()
    marker = cfg.paths.monitor_features_dir.parent / ".s6_population.done"
    marker.unlink(missing_ok=True)
    n = build(cfg)
    section("Result")
    if n <= 0:
        print("CHECKS FAILED - no .done marker written")
        return 1
    marker.write_text("ok\n")
    print(f"wrote {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
