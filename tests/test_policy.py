"""Stage 5: ranking never sees the outcome; capture math checked by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import policy as pol
from src.pipeline.labels import LABEL
from src.utils.leakage import LeakageError

BY_BALANCE = pol.Policy("by_balance", "column", pol.EXPOSURE)


def _month(period: str, exposure: list[float], label: list[int], **extra) -> pd.DataFrame:
    return pd.DataFrame(
        {pol.PERIOD: period, pol.EXPOSURE: exposure, LABEL: label, **extra},
    )


def _synthetic(n_months=6, n=2_000, seed=0) -> pd.DataFrame:
    """Months where risk is visible in p and dq_bucket, balances independent of risk."""
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(n_months):
        p = rng.uniform(0.05, 0.9, n)
        frames.append(
            pd.DataFrame(
                {
                    pol.PERIOD: f"2018-{m + 1:02d}-01",
                    pol.EXPOSURE: rng.lognormal(11, 0.6, n),
                    "dq_bucket": np.digitize(p, [0.35, 0.65]) + 1,
                    "p": p,
                    "p_raw": p,
                    LABEL: rng.binomial(1, p),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- the label never reaches the ranking ----------------------------------------


def test_policy_ranking_has_no_label():
    month = _month("2018-01-01", [1.0, 2.0], [0, 1])
    with pytest.raises(LeakageError, match="rolls_deeper"):
        pol.rank(month, BY_BALANCE, np.random.default_rng(0))
    with pytest.raises(LeakageError, match="money_at_risk"):
        features = month.drop(columns=LABEL).assign(money_at_risk=[0.0, 2.0])
        pol.rank(features, BY_BALANCE, np.random.default_rng(0))
    order = pol.rank(month.drop(columns=LABEL), BY_BALANCE, np.random.default_rng(0))
    assert list(order) == [1, 0]


def test_simulate_ranks_without_the_label(monkeypatch):
    seen: list[set[str]] = []
    real_rank = pol.rank

    def spy(features, policy, rng):
        seen.append(set(features.columns))
        return real_rank(features, policy, rng)

    monkeypatch.setattr(pol, "rank", spy)
    frame = _month("2018-01-01", [1.0, 2.0, 3.0], [1, 0, 1])
    pol.simulate(frame, [BY_BALANCE], [0.5, 1.0], seeds=[0, 1])
    assert seen and all(LABEL not in cols for cols in seen)


# --- capture math ------------------------------------------------------------------


def test_contact_count_rounds_down_without_float_error():
    assert pol.contact_count(10, 0.3) == 3  # 0.3 * 10 == 2.9999999999999996
    assert pol.contact_count(411, 0.05) == 20
    assert pol.contact_count(7, 1.0) == 7
    with pytest.raises(ValueError):
        pol.contact_count(10, 1.5)


def test_capture_math():
    # Month 1: money at risk [100, 0, 300, 0, 500], total 900.
    #   by_balance contacts 500, 400 at C=0.4 (k=2) -> 500 captured.
    #   oracle contacts 500, 300 -> 800.
    # Month 2: money at risk [0, 10], total 10. At C=0.4 (k=0) nothing captured;
    #   at C=0.5, k=floor(2.5)=2 in month 1 and k=1 in month 2.
    frame = pd.concat(
        [
            _month("2018-01-01", [100, 200, 300, 400, 500], [1, 0, 1, 0, 1]),
            _month("2018-02-01", [1000, 10], [0, 1]),
        ],
        ignore_index=True,
    )
    sim = pol.simulate(frame, [BY_BALANCE], [0.4, 0.5, 1.0], seeds=[0])
    curve = pol.capture_curve(sim)
    assert curve.loc[0.4, "by_balance"] == pytest.approx(500 / 910)
    assert curve.loc[0.4, "oracle"] == pytest.approx(800 / 910)
    # month 1 k=2 -> 500; month 2 k=1 contacts the 1000 balance that cured -> 0
    assert curve.loc[0.5, "by_balance"] == pytest.approx(500 / 910)
    assert curve.loc[0.5, "oracle"] == pytest.approx(810 / 910)
    assert (curve.loc[1.0] == 1.0).all()

    monthly = pol.monthly_capture(sim, 0.4)
    assert monthly.loc["2018-01-01", "by_balance"] == pytest.approx(500 / 900)
    assert monthly.loc["2018-02-01", "by_balance"] == 0.0


def test_balance_share_by_hand():
    frame = pd.concat(
        [
            _month("2018-01-01", [100, 200, 300, 400], [0, 0, 0, 0]),
            _month("2018-02-01", [50, 950], [1, 1]),
        ],
        ignore_index=True,
    )
    share = pol.balance_share(frame, [0.5])
    # month 1 top 2: 700 of 1000; month 2 top 1: 950 of 1000
    assert share.loc[0.5] == pytest.approx(1650 / 2000)


def test_gap_closed():
    curve = pd.DataFrame({"by_balance": [0.3], "model": [0.5], "oracle": [0.7]}, index=[0.2])
    assert pol.gap_closed(curve, "model", "by_balance").loc[0.2] == pytest.approx(0.5)


# --- ranking behaviour ----------------------------------------------------------------


def test_by_dpd_keeps_bucket_order_and_breaks_ties_randomly():
    features = pd.DataFrame({"dq_bucket": [1, 3, 2, 3, 1, 3, 2, 1], pol.EXPOSURE: 1.0})
    by_dpd = pol.Policy("by_dpd", "column", "dq_bucket")
    orders = {tuple(pol.rank(features, by_dpd, np.random.default_rng(s))) for s in range(20)}
    for order in orders:
        buckets = features["dq_bucket"].to_numpy()[list(order)]
        assert list(buckets) == sorted(buckets, reverse=True)
    assert len(orders) > 1  # ties were not resolved by row order


def test_expected_value_priority_is_prob_times_exposure():
    features = pd.DataFrame({"p": [0.9, 0.2, 0.5], pol.EXPOSURE: [10.0, 100.0, 30.0]})
    ev = pol.Policy("by_expected_value", "expected_value", "p")
    np.testing.assert_allclose(ev.priority(features), [9.0, 20.0, 15.0])
    assert list(pol.rank(features, ev, np.random.default_rng(0))) == [1, 2, 0]


def test_null_priorities_rejected():
    features = pd.DataFrame({"p": [0.1, np.nan], pol.EXPOSURE: [1.0, 1.0]})
    with pytest.raises(ValueError, match="null"):
        pol.Policy("by_prob", "column", "p").priority(features)


# --- structural validity checks ---------------------------------------------------------


CAPS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def _checks(curve, frame):
    return pol.validity_checks(
        curve, pol.balance_share(frame, CAPS), 0.02, 0.05, balance_capacities=[0.1, 0.2, 0.3]
    )


def test_sound_policies_pass_every_check():
    frame = _synthetic()
    sim = pol.simulate(frame, pol.standard_policies("p", "p_raw"), CAPS, seeds=range(20))
    curve = pol.capture_curve(sim)
    failed = [c.message for c in _checks(curve, frame) if not c.ok]
    assert failed == []
    assert abs(curve.loc[0.2, "random"] - 0.2) < 0.02


def test_reversed_sort_fails_the_random_floor():
    frame = _synthetic()
    frame["neg_p"] = -frame["p"]
    policies = [pol.Policy("random", "random"), pol.Policy("by_balance", "column", "exposure")]
    policies.append(pol.Policy("reversed", "column", "neg_p"))
    curve = pol.capture_curve(pol.simulate(frame, policies, CAPS, seeds=range(5)))
    failed = [c.message for c in _checks(curve, frame) if not c.ok]
    assert any(m.startswith("4. reversed") for m in failed)


def test_leaked_outcome_fails_the_oracle_ceiling():
    frame = _synthetic()
    # A "feature" that is secretly the outcome: the ranking guard cannot see it by
    # name, so the structural check has to catch it.
    frame["leak"] = frame[LABEL] * frame[pol.EXPOSURE]
    policies = [pol.Policy("random", "random"), pol.Policy("by_balance", "column", "exposure")]
    policies.append(pol.Policy("leaky", "column", "leak"))
    curve = pol.capture_curve(pol.simulate(frame, policies, CAPS, seeds=range(5)))
    failed = [c.message for c in _checks(curve, frame) if not c.ok]
    assert any(m.startswith("3. leaky") for m in failed)


def test_policy_capacities_include_full_capacity_and_table_points(cfg):
    caps = cfg.policy.capacities()
    assert caps[-1] == 1.0
    assert set(cfg.policy.table_capacities) <= set(caps)
    assert cfg.policy.headline_capacity in caps
    assert caps == sorted(caps)


def test_largest_accounts_roll_rate_by_hand():
    frame = _month("2018-01-01", [10, 20, 30, 40], [1, 1, 0, 1])
    out = pol.largest_accounts_roll_rate(frame, 0.5)  # largest two: 40 (rolled), 30 (cured)
    assert out == {"largest": 0.5, "rest": 1.0}
