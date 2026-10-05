"""Stage 6: PSI by hand, bin edge cases, monthly metrics and the departure rule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline import drift as dr


def test_psi_by_hand():
    # (0.5-0.25)ln(0.5/0.25) + (0.5-0.75)ln(0.5/0.75) = 0.25 ln 2 + 0.25 ln 1.5
    expected = 0.25 * math.log(2) + 0.25 * math.log(1.5)
    assert dr.psi(np.array([0.25, 0.75]), np.array([0.5, 0.5]), 1e-4) == pytest.approx(expected)


def test_psi_identical_is_zero_and_empty_bin_is_finite():
    assert dr.psi(np.array([0.2, 0.8]), np.array([0.2, 0.8]), 1e-4) == 0.0
    value = dr.psi(np.array([0.0, 1.0]), np.array([0.5, 0.5]), 1e-4)
    assert np.isfinite(value) and value > 0


def test_numeric_bins_are_reference_quantiles_with_null_bin():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    bins = dr.Bins.fit(ref, n_bins=2)
    np.testing.assert_allclose(bins.edges, [2.5])
    assert bins.labels() == ["b0", "b1", dr.NULL_BIN]
    np.testing.assert_allclose(bins.shares(ref), [0.4, 0.4, 0.2])
    # values beyond the reference range fall in the end bins, never lost
    np.testing.assert_allclose(bins.shares(pd.Series([-100.0, 100.0])), [0.5, 0.5, 0.0])


def test_tied_quantiles_collapse_to_unique_edges():
    bins = dr.Bins.fit(pd.Series([0.0] * 90 + [1.0] * 10), n_bins=10)
    assert len(bins.edges) == len(np.unique(bins.edges))
    assert bins.shares(pd.Series([0.0, 1.0])).sum() == pytest.approx(1.0)


def test_categorical_bins_send_unseen_levels_to_other():
    ref = pd.Series(["CA", "TX", "CA", None], dtype=object)
    bins = dr.Bins.fit(ref, n_bins=10)
    assert bins.labels() == ["CA", "TX", dr.OTHER_BIN, dr.NULL_BIN]
    shares = bins.shares(pd.Series(["CA", "NY", None, "TX"], dtype=object))
    np.testing.assert_allclose(shares, [0.25, 0.25, 0.25, 0.25])


def test_psi_by_period_and_split_half():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=20_000)})
    mon = pd.DataFrame(
        {
            "period": ["a"] * 5_000 + ["b"] * 5_000,
            "x": np.r_[rng.normal(size=5_000), rng.normal(loc=1.0, size=5_000)],
        }
    )
    bins = dr.fit_bins(ref, ["x"], 10)
    table = dr.psi_by_period(ref, mon, bins, "period", 1e-4)
    assert table.loc["a", "x"] < 0.01 < 0.25 < table.loc["b", "x"]
    assert dr.split_half_psi(ref, bins, 1e-4, seed=1)["x"] < 0.01


def test_monthly_metrics_by_hand():
    frame = pd.DataFrame(
        {
            "period": ["m1"] * 4 + ["m2"] * 2,
            "y": [1, 0, 1, 0, 0, 0],
            "p": [0.9, 0.1, 0.8, 0.2, 0.3, 0.5],
            "raw": [0.9, 0.1, 0.8, 0.2, 0.3, 0.5],
        }
    )
    m = dr.monthly_metrics(frame, "y", "p", "raw", "period", 10)
    assert m.loc["m1", "auc"] == 1.0
    assert m.loc["m1", "gap"] == pytest.approx(0.5 - 0.5)
    assert m.loc["m2", "gap"] == pytest.approx(0.4)  # over-predicts a month with no rolls
    assert np.isnan(m.loc["m2", "auc"])  # one class: AUC undefined, not a number


def test_first_departure_needs_a_sustained_run_in_the_bad_direction():
    band = dr.Band(0.0, 1.0)
    s = pd.Series([0.5, 2.0, 0.5, -5.0, -5.0, 2.0, 2.0, 2.0], index=list("abcdefgh"))
    assert dr.first_departure(s, band, "up", consecutive=2) == "f"  # b alone is a blip
    assert dr.first_departure(s, band, "down", consecutive=2) == "d"
    assert dr.first_departure(s, band, "up", consecutive=1) == "b"
    assert dr.first_departure(s.iloc[:3], band, "up", consecutive=2) is None
    assert dr.first_departure(pd.Series([np.nan, np.nan]), band, "up", 1) is None


def test_baseline_band():
    band = dr.baseline_band(pd.Series([1.0, 2.0, 3.0]), k=2.0)
    assert band.lo == pytest.approx(0.0) and band.hi == pytest.approx(4.0)
