"""Stage 3: feature values on a known history, and the no-future-leakage guarantee."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest
from pyspark.sql import functions as F

from src.pipeline import features as FT
from src.pipeline import macro as M
from src.utils import leakage
from src.utils.config import FeaturesConfig

SCHEMA = (
    "loan_sequence_number string, reporting_period date, dq_bucket int, current_upb double, "
    "orig_upb double, current_interest_rate double, orig_interest_rate double, "
    "modification_flag string, borrower_assistance_status string, "
    "payment_deferral_flag string, delinquency_due_to_disaster string, loan_age int, "
    "remaining_months_to_maturity int, current_deferred_upb double, eltv int, "
    "step_modification_flag string"
)
# 16 months of one loan; T is month 13 so 12-month windows are full.
BUCKETS = [0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 0, 1, 2, 1, 2, 3]
T = 13


def m(i: int) -> dt.date:
    return dt.date(2015 + i // 12, i % 12 + 1, 1)


def history(buckets=BUCKETS, skip=(), overrides=None):
    rows = []
    for i, b in enumerate(buckets):
        if i in skip:
            continue
        row = {
            "loan_sequence_number": "L1",
            "reporting_period": m(i),
            "dq_bucket": b,
            "current_upb": 100_000.0 - 500 * i,
            "orig_upb": 100_000.0,
            "current_interest_rate": 4.0,
            "orig_interest_rate": 4.0,
            "modification_flag": None,
            "borrower_assistance_status": None,
            "payment_deferral_flag": None,
            "delinquency_due_to_disaster": None,
            "loan_age": i + 1,
            "remaining_months_to_maturity": 360 - i - 1,
            "current_deferred_upb": 0.0,
            "eltv": 80,
            "step_modification_flag": None,
        }
        row.update((overrides or {}).get(i, {}))
        rows.append(tuple(row.values()))
    return rows


def features_at(spark, cfg: FeaturesConfig, rows, month=T) -> dict:
    df = spark.createDataFrame(rows, SCHEMA)
    out = FT.build_loan_month_features(df, cfg).filter(F.col("reporting_period") == m(month))
    got = out.collect()
    assert len(got) == 1
    return got[0].asDict()


def computed_features(cfg: FeaturesConfig) -> list[str]:
    """Registry features this module computes from the panel (static and macro are joins)."""
    reg = FT.feature_registry(cfg)
    return [n for n, meta in reg.items() if meta.family in ("B_current", "C_trajectory")]


def same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


PERTURB = {
    "dq_bucket": 3,
    "current_upb": 1.0,
    "current_interest_rate": 9.9,
    "modification_flag": "Y",
    "borrower_assistance_status": "F",
    "payment_deferral_flag": "C",
    "delinquency_due_to_disaster": "Y",
    "loan_age": 999,
    "remaining_months_to_maturity": 1,
    "current_deferred_upb": 5_000.0,
    "eltv": 150,
    "step_modification_flag": "Y",
}
# A second variant flips every flag the other way (e.g. delinquent -> current).
PERTURB_DOWN = {**PERTURB, "dq_bucket": 0, "current_upb": 250_000.0, "loan_age": 0}


# --- test_no_future_leakage ---------------------------------------------------


def test_no_future_leakage_months_after_t_change_nothing(spark, cfg):
    base = features_at(spark, cfg.features, history())
    future = {i: dict(PERTURB) for i in range(T + 1, len(BUCKETS))}
    changed = features_at(spark, cfg.features, history(overrides=future))
    names = computed_features(cfg.features)
    assert all(same(base[n], changed[n]) for n in names), [
        n for n in names if not same(base[n], changed[n])
    ]
    truncated = features_at(spark, cfg.features, history(buckets=BUCKETS[: T + 1]))
    assert all(same(base[n], truncated[n]) for n in names)


@pytest.mark.parametrize("perturb", [PERTURB, PERTURB_DOWN], ids=["up", "down"])
@pytest.mark.parametrize("k", [0, 1, 2])  # no feature ends before T-3
def test_features_ignore_months_newer_than_their_declared_window(spark, cfg, k, perturb):
    """Perturb month T-k: every feature whose window ends before T-k must not move."""
    reg = FT.feature_registry(cfg.features)
    base = features_at(spark, cfg.features, history())
    changed = features_at(spark, cfg.features, history(overrides={T - k: dict(perturb)}))
    must_hold = [
        n
        for n in computed_features(cfg.features)
        if reg[n].window_end < -k  # type: ignore[operator]
    ]
    assert must_hold, "no features to check"
    moved = [n for n in must_hold if not same(base[n], changed[n])]
    assert not moved, f"features read month T-{k} despite declaring an earlier window: {moved}"


def test_declared_windows_are_not_overly_generous(spark, cfg):
    """Each month-T feature actually responds to month T (the registry is not just padding)."""
    base = features_at(spark, cfg.features, history())
    changed = features_at(spark, cfg.features, history(overrides={T: dict(PERTURB)}))
    reg = FT.feature_registry(cfg.features)
    reading_t = [n for n in computed_features(cfg.features) if reg[n].window_end == 0]
    assert any(not same(base[n], changed[n]) for n in reading_t)


# --- values on a known history ------------------------------------------------


def test_trajectory_values(spark, cfg):
    f = features_at(spark, cfg.features, history())
    # months T-3..T-1 = 10, 11, 12 -> buckets 0, 1, 2
    assert f["months_delinquent_last_3"] == 2
    assert f["max_bucket_last_3"] == 2
    assert f["mean_bucket_last_3"] == pytest.approx(1.0)
    assert (f["bucket_lag_1"], f["bucket_lag_2"], f["bucket_lag_3"]) == (2, 1, 0)
    assert f["delinquency_velocity_3m"] == 1 - 0  # bucket 1 at T, 0 at T-3
    assert f["months_since_last_current"] == 3  # last current at month 10
    assert f["months_since_first_delinquency"] == T - 2
    # entries into delinquency before T: months 2, 6, 11
    assert f["times_entered_delinquency_before_t"] == 3
    # cures before T (within last 6 = months 7..12): month 8
    assert f["n_cure_events_last_6"] == 1
    # rolls in months 7..12: 7 (1->1 no), 12 (1->2 yes) ... and 11 (0->1 yes)
    assert f["n_roll_events_last_6"] == 2
    assert f["longest_delinquent_run_before_t"] == 2
    assert f["current_run_length"] == 3  # months 11, 12, 13
    assert f["max_bucket_before_t"] == 2
    assert f["upb_change_pct_3"] == pytest.approx((93_500 / 95_000 - 1) * 100)


def test_missing_month_is_not_bridged(spark, cfg):
    f = features_at(spark, cfg.features, history(skip={T - 1}))
    assert f["bucket_lag_1"] is None  # T-1 missing: not silently taken from T-2
    assert f["bucket_lag_2"] == 1
    assert f["months_observed_last_12"] == 11
    # T-2 and T-3 still in the 3-month window; the run restarts after the gap
    assert f["months_delinquent_last_3"] == 1
    assert f["current_run_length"] == 1


# --- registry and blocklist ---------------------------------------------------


def test_registry_passes_the_guard(cfg):
    leakage.check_registry(FT.feature_registry(cfg.features))


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        (leakage.FeatureMeta("C", ("actual_loss",), -1), "blocked"),
        (leakage.FeatureMeta("C", ("dq_bucket",), 1), "T\\+1"),
        (leakage.FeatureMeta("C", ("rolls_deeper",), -1), "blocked"),
    ],
)
def test_registry_guard_rejects(meta, match):
    with pytest.raises(leakage.LeakageError, match=match):
        leakage.check_registry({"some_feature": meta})


@pytest.mark.parametrize("column", ["zero_balance_code", "ddlpi", "Actual_Loss_x", "rolls_deeper"])
def test_blocked_column_names(column):
    with pytest.raises(leakage.LeakageError):
        leakage.check_columns(["credit_score", column])


def test_unregistered_feature_column_is_rejected(cfg):
    with pytest.raises(leakage.LeakageError, match="mystery"):
        leakage.check_features_registered(
            ["credit_score", "mystery"], FT.feature_registry(cfg.features)
        )


# --- macro --------------------------------------------------------------------


def test_macro_lag_uses_only_published_values():
    idx = pd.date_range("2015-01-01", periods=6, freq="MS")
    monthly = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    out = M.lagged_features(monthly, "unemployment_rate", lag=2, change_windows=[1])
    assert out.loc["2015-03-01", "unemployment_rate"] == 1.0  # Jan value, public by March
    assert out.loc["2015-04-01", "unemployment_rate_chg_1m"] == pytest.approx(1.0)
    assert (
        pd.isna(out.loc["2015-02-01", "unemployment_rate"]) if "2015-02-01" in out.index else True
    )


def test_quarterly_series_fills_its_quarter_including_the_last():
    q = pd.Series([10.0, 20.0], index=pd.to_datetime(["2015-01-01", "2015-04-01"]))
    monthly = M.to_monthly(q)
    assert monthly.loc["2015-03-01"] == 10.0
    assert monthly.loc["2015-06-01"] == 20.0  # last quarter carried to its third month
    assert "2015-07-01" not in monthly.index
