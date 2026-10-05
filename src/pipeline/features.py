"""Stage 3 feature families A-D, computed in Spark, each strictly backward-looking.

Every feature is declared in ``feature_registry`` with its sources and the
newest month it reads relative to T. Trajectory windows are RANGE windows on a
calendar month index ending at T-1, so missing months cannot pull in month T
or anything later. ``tests/test_features.py`` perturbs months T and later and
asserts each feature only moves when its declared window allows it.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window, WindowSpec
from pyspark.sql import functions as F

from src.pipeline.labels import month_index
from src.utils.config import FeaturesConfig
from src.utils.leakage import FeatureMeta

KEY = ["loan_sequence_number", "reporting_period"]

STATIC_FEATURES = [
    "credit_score",
    "orig_ltv",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_interest_rate",
    "orig_loan_term",
    "mi_pct",
    "loan_purpose",
    "occupancy_status",
    "channel",
    "property_type",
    "num_units",
    "num_borrowers",
    "first_time_homebuyer",
    "property_state",
    "msa",
]

# Month-T values passed through as-is.
CURRENT_PASSTHROUGH = [
    "dq_bucket",
    "loan_age",
    "current_upb",
    "current_interest_rate",
    "remaining_months_to_maturity",
    "current_deferred_upb",
    "eltv",
    "modification_flag",
    "step_modification_flag",
    "payment_deferral_flag",
    "borrower_assistance_status",
    "delinquency_due_to_disaster",
]


def feature_registry(cfg: FeaturesConfig) -> dict[str, FeatureMeta]:
    """Return every candidate feature with its family, sources and window end (relative to T)."""
    reg: dict[str, FeatureMeta] = {}

    def add(name: str, family: str, sources: tuple[str, ...], end: int | None) -> None:
        reg[name] = FeatureMeta(family, sources, end)

    for c in STATIC_FEATURES:
        add(c, "A_static", (c,), None)
    for c in CURRENT_PASSTHROUGH:
        add(c, "B_current", (c,), 0)
    add("upb_to_orig_upb", "B_current", ("current_upb", "orig_upb"), 0)
    add("rate_change_since_orig", "B_current", ("current_interest_rate", "orig_interest_rate"), 0)

    dq = ("dq_bucket",)
    for n in cfg.trajectory_windows:
        add(f"months_delinquent_last_{n}", "C_trajectory", dq, -1)
        add(f"max_bucket_last_{n}", "C_trajectory", dq, -1)
        add(f"mean_bucket_last_{n}", "C_trajectory", dq, -1)
    add(f"months_observed_last_{max(cfg.trajectory_windows)}", "C_trajectory", dq, -1)
    for n in cfg.event_windows:
        add(f"n_cure_events_last_{n}", "C_trajectory", dq, -1)
        add(f"n_roll_events_last_{n}", "C_trajectory", dq, -1)
    for k in cfg.bucket_lags:
        add(f"bucket_lag_{k}", "C_trajectory", dq, -k)
    add("months_since_last_current", "C_trajectory", dq, -1)
    add("months_since_first_delinquency", "C_trajectory", dq, -1)
    add("longest_delinquent_run_before_t", "C_trajectory", dq, -1)
    add("times_entered_delinquency_before_t", "C_trajectory", dq, -1)
    add("max_bucket_before_t", "C_trajectory", dq, -1)
    add("ever_modified_before_t", "C_trajectory", ("modification_flag",), -1)
    add("ever_forbearance_before_t", "C_trajectory", ("borrower_assistance_status",), -1)
    add("ever_payment_deferral_before_t", "C_trajectory", ("payment_deferral_flag",), -1)
    add("ever_disaster_before_t", "C_trajectory", ("delinquency_due_to_disaster",), -1)
    # These compare month T with an earlier month, as the spec defines them.
    add(f"delinquency_velocity_{cfg.velocity_lag}m", "C_trajectory", dq, 0)
    for n in cfg.upb_change_windows:
        add(f"upb_change_pct_{n}", "C_trajectory", ("current_upb",), 0)
    add("current_run_length", "C_trajectory", dq, 0)

    m = cfg.macro
    for name in [*m.state_series, *m.national_series]:
        lag = m.publication_lag_months[name]
        src = (m.state_series.get(name) or m.national_series[name],)
        add(name, "D_macro", src, -lag)
        for n in m.change_windows:
            add(f"{name}_chg_{n}m", "D_macro", src, -lag)
    return reg


def _w() -> WindowSpec:
    return Window.partitionBy("loan_sequence_number").orderBy("month_idx")


def _range(lo: int, hi: int) -> WindowSpec:
    return _w().rangeBetween(lo, hi)


def _past() -> WindowSpec:
    """Every earlier month of the loan, ending at T-1."""
    return _w().rangeBetween(Window.unboundedPreceding, -1)


def _value_at(col: str, k: int) -> Column:
    """Return ``col`` at exactly month T-k (null if that month is missing)."""
    return F.max(col).over(_range(-k, -k))


def add_history_columns(panel: DataFrame) -> DataFrame:
    """Return the panel with month index and per-month event flags used by trajectory features.

    Each flag at month t uses months t and t-1 only.
    """
    w = _w()
    prev_idx = F.lag("month_idx").over(w)
    prev_bucket = F.when(prev_idx == F.col("month_idx") - 1, F.lag("dq_bucket").over(w))
    b = F.col("dq_bucket")
    return (
        panel.withColumn("month_idx", month_index(F.col("reporting_period")))
        .withColumn("prev_bucket", prev_bucket)
        .withColumn("is_dq", (b > 0).cast("int"))
        .withColumn("cure_event", ((F.col("prev_bucket") > 0) & (b == 0)).cast("int"))
        .withColumn("roll_event", (b > F.col("prev_bucket")).cast("int"))
        .withColumn(
            "entry_event", ((F.coalesce(F.col("prev_bucket"), F.lit(0)) == 0) & (b > 0)).cast("int")
        )
        .withColumn(
            "run_start",
            F.max(F.when(F.col("entry_event") == 1, F.col("month_idx"))).over(
                w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            ),
        )
        .withColumn("run_len", F.when(b > 0, F.col("month_idx") - F.col("run_start") + 1))
        .withColumn("is_modified", F.col("modification_flag").isNotNull().cast("int"))
        .withColumn("is_forbearance", (F.col("borrower_assistance_status") == "F").cast("int"))
        .withColumn("is_deferral", F.col("payment_deferral_flag").isNotNull().cast("int"))
        .withColumn("is_disaster", (F.col("delinquency_due_to_disaster") == "Y").cast("int"))
    )


def trajectory_columns(cfg: FeaturesConfig) -> dict[str, Column]:
    """Return family C feature expressions over a frame from ``add_history_columns``."""
    cols: dict[str, Column] = {}
    for n in cfg.trajectory_windows:
        w = _range(-n, -1)
        cols[f"months_delinquent_last_{n}"] = F.sum("is_dq").over(w)
        cols[f"max_bucket_last_{n}"] = F.max("dq_bucket").over(w)
        cols[f"mean_bucket_last_{n}"] = F.avg("dq_bucket").over(w)
    n_max = max(cfg.trajectory_windows)
    cols[f"months_observed_last_{n_max}"] = F.count(F.lit(1)).over(_range(-n_max, -1))
    for n in cfg.event_windows:
        cols[f"n_cure_events_last_{n}"] = F.sum("cure_event").over(_range(-n, -1))
        cols[f"n_roll_events_last_{n}"] = F.sum("roll_event").over(_range(-n, -1))
    for k in cfg.bucket_lags:
        cols[f"bucket_lag_{k}"] = _value_at("dq_bucket", k)

    past = _past()
    idx = F.col("month_idx")
    cols["months_since_last_current"] = idx - F.max(F.when(F.col("dq_bucket") == 0, idx)).over(past)
    cols["months_since_first_delinquency"] = idx - F.min(F.when(F.col("dq_bucket") > 0, idx)).over(
        past
    )
    cols["longest_delinquent_run_before_t"] = F.max("run_len").over(past)
    cols["times_entered_delinquency_before_t"] = F.sum("entry_event").over(past)
    cols["max_bucket_before_t"] = F.max("dq_bucket").over(past)
    cols["ever_modified_before_t"] = F.max("is_modified").over(past)
    cols["ever_forbearance_before_t"] = F.max("is_forbearance").over(past)
    cols["ever_payment_deferral_before_t"] = F.max("is_deferral").over(past)
    cols["ever_disaster_before_t"] = F.max("is_disaster").over(past)

    cols[f"delinquency_velocity_{cfg.velocity_lag}m"] = F.col("dq_bucket") - _value_at(
        "dq_bucket", cfg.velocity_lag
    )
    for n in cfg.upb_change_windows:
        before = _value_at("current_upb", n)
        cols[f"upb_change_pct_{n}"] = F.when(
            before > 0, (F.col("current_upb") / before - 1.0) * 100.0
        )
    cols["current_run_length"] = F.col("run_len")
    return cols


def current_state_columns() -> dict[str, Column]:
    """Return the derived family B features (pass-through columns need no expression)."""
    return {
        "upb_to_orig_upb": F.when(F.col("orig_upb") > 0, F.col("current_upb") / F.col("orig_upb")),
        "rate_change_since_orig": F.col("current_interest_rate") - F.col("orig_interest_rate"),
    }


def build_loan_month_features(panel: DataFrame, cfg: FeaturesConfig) -> DataFrame:
    """Return families A-C for every panel row (macro is joined separately)."""
    hist = add_history_columns(panel)
    return hist.withColumns(trajectory_columns(cfg)).withColumns(current_state_columns())
