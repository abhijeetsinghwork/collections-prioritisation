"""Stage 4: metric helpers with hand-checkable answers, and model-matrix safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.pipeline import evaluation as ev
from src.pipeline import modeling as md
from src.pipeline.labels import LABEL
from src.utils.config import LightGBMConfig


def test_reliability_table_by_hand():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.95, 0.85])
    t = ev.reliability_table(y, p, n_bins=10)
    assert t.loc[0, "n"] == 1 and t.loc[0, "actual"] == 0.0
    assert t.loc[1, "n"] == 1 and t.loc[1, "actual"] == 1.0
    assert t.loc[8, "n"] == 1 and t.loc[9, "n"] == 1
    assert t["n"].sum() == 4
    assert t.loc[5, "n"] == 0 and pd.isna(t.loc[5, "mean_pred"])


def test_expected_calibration_error_by_hand():
    # one bin each: |0.2-0| = 0.2 and |0.9-1| = 0.1, equal weights -> 0.15
    y = np.array([0, 1])
    p = np.array([0.2, 0.9])
    assert ev.expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.15)


def test_decile_table_orders_highest_risk_first():
    p = np.linspace(0.01, 1.0, 100)
    y = (p > 0.5).astype(int)
    t = ev.decile_table(y, p, n_bins=10)
    assert list(t["decile"]) == list(range(1, 11))
    assert t.loc[0, "actual"] == 1.0 and t.loc[9, "actual"] == 0.0
    assert (t["n"] == 10).all()


def test_per_bucket_handles_single_class_bucket():
    y = np.array([0, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.2, 0.3])
    b = np.array([1, 1, 2, 2])
    out = ev.per_bucket(y, s, b, [1, 2])
    assert out[1]["auc"] == 1.0
    assert "auc" not in out[2]  # all zeros: AUC undefined, not reported as a number


def test_isotonic_fixes_a_systematically_overconfident_score():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.6, 20_000)
    y = rng.binomial(1, true_p)
    overconfident = np.clip(true_p * 1.5, 0, 1)
    cal = IsotonicRegression(out_of_bounds="clip").fit(overconfident[:10_000], y[:10_000])
    after = cal.predict(overconfident[10_000:])
    yt = y[10_000:]
    assert ev.brier(yt, after) < ev.brier(yt, overconfident[10_000:])


# --- model matrix --------------------------------------------------------------


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "state": rng.choice(["CA", "TX"], size=n).astype(object),
            "dq_bucket": rng.choice([1, 2, 3], size=n),
            LABEL: rng.integers(0, 2, size=n),
        }
    )


def test_matrix_spec_rejects_outcome_columns():
    spec = md.MatrixSpec(features=["x", LABEL])
    with pytest.raises(ValueError, match="outcome"):
        spec.matrix(_frame())


def test_matrix_spec_fixes_categories_from_train():
    train = _frame()
    spec = md.MatrixSpec.from_train(train, ["x", "state"])
    other = train.copy()
    other.loc[0, "state"] = "NY"  # unseen in train
    X = spec.matrix(other)
    assert list(X["state"].cat.categories) == ["CA", "TX"]
    assert pd.isna(X.loc[0, "state"])


def test_segmented_prediction_routes_by_bucket():
    train, valid = _frame(600, 1), _frame(300, 2)
    spec = md.MatrixSpec.from_train(train, ["x", "state", "dq_bucket"])
    cfg = LightGBMConfig(
        params={"objective": "binary", "metric": "auc", "verbosity": -1, "min_data_in_leaf": 5},
        num_boost_round=5,
        early_stopping_rounds=2,
    )
    boosters = md.fit_segmented(train, valid, spec, cfg, seed=0, buckets=[1, 2, 3])
    p = md.predict_segmented(boosters, spec, valid)
    for b in (1, 2, 3):
        rows = valid[valid["dq_bucket"] == b]
        np.testing.assert_allclose(
            p[valid["dq_bucket"].to_numpy() == b],
            md.predict_lightgbm(boosters[b], spec, rows),
        )
    with pytest.raises(ValueError, match="no segment model"):
        md.predict_segmented({1: boosters[1]}, spec, valid)


# --- calibration transfer check ------------------------------------------------


def _transfer_frame(raw, y, months):
    periods = pd.date_range("2016-01-01", periods=months, freq="MS")
    per = len(raw) // months
    return pd.DataFrame(
        {"reporting_period": np.repeat(periods, per)[: len(raw)], LABEL: y[: per * months]}
    )


def test_transfer_check_keeps_isotonic_when_miscalibration_is_stable():
    from src.s4_models import calibration_transfer

    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.05, 0.6, 24_000)
    y = rng.binomial(1, true_p)
    raw = np.clip(true_p * 1.5, 0, 1)  # same overconfidence in both years
    table, method = calibration_transfer(_transfer_frame(raw, y, 24), raw, 12, 10)
    assert method == "isotonic"
    assert set(table["method"]) == {"raw", "isotonic"}
    assert table["eval_rows"].iloc[0] == 12_000


def test_transfer_check_falls_back_to_raw_when_the_map_does_not_carry_forward():
    from src.s4_models import calibration_transfer

    rng = np.random.default_rng(2)
    raw = rng.uniform(0.05, 0.6, 24_000)
    true_p = raw.copy()
    true_p[:12_000] = np.clip(raw[:12_000] * 1.4, 0, 1)  # year 1 rolls more than predicted
    y = rng.binomial(1, true_p)  # year 2: raw is right
    _, method = calibration_transfer(_transfer_frame(raw, y, 24), raw, 12, 10)
    assert method == "raw"
