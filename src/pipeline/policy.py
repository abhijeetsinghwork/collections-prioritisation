"""Contact policies under a capacity constraint, and the capture metric.

Ranking and scoring are kept apart. ``rank`` sees only what is known at the
start of month m and raises if it is handed an outcome column. The outcome
enters only through ``money_at_risk``, which scores a ranking already made.
The oracle is the one ordering built from the outcome, and it lives in its
own function so that it cannot be confused with a policy.

Pure numpy/pandas; no data loading here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.pipeline.labels import LABEL
from src.utils.leakage import OUTCOME_COLUMNS, LeakageError

MONEY_AT_RISK = "money_at_risk"
EXPOSURE = "exposure"
PERIOD = "reporting_period"
ORACLE = "oracle"
RANDOM = "random"
FORBIDDEN_IN_RANKING = OUTCOME_COLUMNS | {MONEY_AT_RISK}
# Capacity fractions are compared after rounding, so 0.2 * n is not lost to float error.
_CAPACITY_EPS = 1e-9


@dataclass(frozen=True)
class Policy:
    """A ranking rule: highest priority first, ties broken randomly.

    ``kind`` is ``random`` (every account tied), ``column`` (priority is the
    named column) or ``expected_value`` (priority is the column x exposure).
    """

    name: str
    kind: Literal["random", "column", "expected_value"]
    column: str | None = None

    def priority(self, features: pd.DataFrame) -> np.ndarray:
        """Return one priority value per row; larger means contact first."""
        if self.kind == "random":
            return np.zeros(len(features))
        assert self.column is not None, f"policy {self.name} needs a column"
        values = features[self.column].to_numpy(dtype=float)
        if self.kind == "expected_value":
            values = values * features[EXPOSURE].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"policy {self.name}: null priorities")
        return values


def standard_policies(prob_column: str, raw_column: str) -> list[Policy]:
    """Return the spec's policies plus the raw-probability expected value (a sensitivity)."""
    return [
        Policy(RANDOM, "random"),
        Policy("by_dpd", "column", "dq_bucket"),
        Policy("by_balance", "column", EXPOSURE),
        Policy("by_prob", "column", prob_column),
        Policy("by_expected_value", "expected_value", prob_column),
        Policy("by_expected_value_raw", "expected_value", raw_column),
    ]


def assert_no_outcome(features: pd.DataFrame) -> None:
    """Raise if ``features`` carries the label or anything derived from it."""
    leaked = sorted(FORBIDDEN_IN_RANKING & set(features.columns))
    if leaked:
        raise LeakageError(f"ranking received outcome columns: {leaked}")


def rank(features: pd.DataFrame, policy: Policy, rng: np.random.Generator) -> np.ndarray:
    """Return row positions in contact order (first = contact first).

    ``features`` must hold only what is known at the start of the month: the
    function raises if an outcome column is present.
    """
    assert_no_outcome(features)
    tiebreak = rng.permutation(len(features))
    # lexsort sorts by the last key first: priority descending, then the random key.
    return np.lexsort((tiebreak, -policy.priority(features)))


def oracle_order(money_at_risk: np.ndarray) -> np.ndarray:
    """Return row positions by realised money at risk, descending: the ceiling, not a policy."""
    return np.argsort(-money_at_risk, kind="stable")


def money_at_risk(exposure: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Return exposure where the account rolled deeper, else 0. Scoring only."""
    return np.where(label == 1, exposure, 0.0)


def contact_count(n: int, capacity: float) -> int:
    """Return how many of ``n`` accounts fit in ``capacity`` (a fraction; rounded down)."""
    if not 0 <= capacity <= 1:
        raise ValueError(f"capacity must be a fraction in [0, 1], got {capacity}")
    return int(np.floor(capacity * n + _CAPACITY_EPS))


def captured(money_in_order: np.ndarray, capacities: Sequence[float]) -> np.ndarray:
    """Return money at risk captured by the first ``capacity`` share of an ordering."""
    cum = np.concatenate([[0.0], np.cumsum(money_in_order)])
    return cum[[contact_count(len(money_in_order), c) for c in capacities]]


def simulate(
    frame: pd.DataFrame,
    policies: Sequence[Policy],
    capacities: Sequence[float],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Return captured and total money at risk per policy, month and capacity.

    Every policy ranks the same accounts in each month, from a frame with the
    label removed. Captured amounts are averaged over ``seeds`` (the random
    policy and every policy's tie-breaks). Columns: policy, period, capacity,
    n, captured, total_at_risk.
    """
    rows = []
    for period, month in frame.groupby(PERIOD, sort=True):
        month = month.reset_index(drop=True)
        money = money_at_risk(month[EXPOSURE].to_numpy(dtype=float), month[LABEL].to_numpy())
        features = month.drop(columns=[LABEL])
        total = float(money.sum())
        results = {ORACLE: captured(money[oracle_order(money)], capacities)}
        for policy in policies:
            draws = [
                captured(money[rank(features, policy, np.random.default_rng(s))], capacities)
                for s in seeds
            ]
            results[policy.name] = np.mean(draws, axis=0)
        for name, amounts in results.items():
            for c, amount in zip(capacities, amounts, strict=True):
                rows.append(
                    {
                        "policy": name,
                        "period": period,
                        "capacity": c,
                        "n": len(month),
                        "captured": float(amount),
                        "total_at_risk": total,
                    }
                )
    return pd.DataFrame(rows)


def capture_curve(sim: pd.DataFrame) -> pd.DataFrame:
    """Return capture rate (index capacity, one column per policy), months weighted by risk.

    Weighting each month's capture rate by its total money at risk is the same
    as pooling: sum captured over months / sum total at risk over months.
    """
    pooled = sim.pivot_table(
        index="capacity", columns="policy", values=["captured", "total_at_risk"], aggfunc="sum"
    )
    return pd.DataFrame(pooled["captured"] / pooled["total_at_risk"])


def monthly_capture(sim: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Return each month's capture rate at one capacity (index period, column per policy)."""
    at = sim[np.isclose(sim["capacity"], capacity)].assign(
        rate=lambda d: d["captured"] / d["total_at_risk"]
    )
    return at.pivot_table(index="period", columns="policy", values="rate", aggfunc="first")


def balance_share(frame: pd.DataFrame, capacities: Sequence[float]) -> pd.Series:
    """Return the share of balance held by the largest ``capacity`` share of accounts.

    Computed directly from exposure, outside the policy machinery, pooled over
    months the same way as ``capture_curve``. Uses no outcome.
    """
    top = np.zeros(len(capacities))
    total = 0.0
    for _, month in frame.groupby(PERIOD, sort=True):
        exposure = np.sort(month[EXPOSURE].to_numpy(dtype=float))[::-1]
        top += captured(exposure, capacities)
        total += float(exposure.sum())
    return pd.Series(top / total, index=pd.Index(list(capacities), name="capacity"))


def largest_accounts_roll_rate(frame: pd.DataFrame, capacity: float) -> dict[str, float]:
    """Return the roll rate of each month's largest ``capacity`` share by balance vs the rest.

    Diagnostic for check 6, reported after the fact: explains why by_balance
    differs from the balance share. Reads the outcome, so never used to rank.
    """
    flags = []
    for _, month in frame.groupby(PERIOD, sort=True):
        k = contact_count(len(month), capacity)
        top = np.zeros(len(month), dtype=bool)
        top[np.argsort(-month[EXPOSURE].to_numpy(dtype=float), kind="stable")[:k]] = True
        flags.append(pd.DataFrame({"top": top, LABEL: month[LABEL].to_numpy()}))
    both = pd.concat(flags, ignore_index=True)
    return {
        "largest": float(both.loc[both["top"], LABEL].mean()),
        "rest": float(both.loc[~both["top"], LABEL].mean()),
    }


def gap_closed(curve: pd.DataFrame, policy: str, floor: str, ceiling: str = ORACLE) -> pd.Series:
    """Return the share of the floor-to-ceiling gap that ``policy`` closes, per capacity."""
    return (curve[policy] - curve[floor]) / (curve[ceiling] - curve[floor])


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


def validity_checks(
    curve: pd.DataFrame,
    balance: pd.Series,
    random_tolerance: float,
    balance_tolerance: float,
    balance_capacities: Sequence[float],
) -> list[CheckResult]:
    """Return the spec's structural checks on a capture curve. Any failure is a bug.

    ``curve`` must include capacity 1.0. Checks: random near the capacity;
    oracle highest; every policy strictly below oracle below full capacity;
    every policy at or above random; monotone and reaching 1 at full capacity;
    by_balance close to the independently computed balance share.
    """
    eps = 1e-9
    caps = curve.index.to_numpy(dtype=float)
    partial = caps < 1.0
    policies = [p for p in curve.columns if p != ORACLE]
    out: list[CheckResult] = []

    dev = (curve[RANDOM] - caps).abs()
    out.append(
        CheckResult(
            bool((dev <= random_tolerance).all()),
            f"1. random capture within {random_tolerance} of the capacity "
            f"(max deviation {dev.max():.4f} at C={dev.idxmax():.2f})",
        )
    )
    above = {p: float((curve[p] - curve[ORACLE]).max()) for p in policies}
    out.append(
        CheckResult(
            all(v <= eps for v in above.values()),
            "2. oracle is the highest policy at every capacity "
            f"(largest policy - oracle {max(above.values()):+.4f})",
        )
    )
    for p in policies:
        gap = (curve.loc[partial, ORACLE] - curve.loc[partial, p]).min()
        out.append(
            CheckResult(
                bool(gap > eps),
                f"3. {p} strictly below oracle below full capacity (smallest gap {gap:.4f})",
            )
        )
    for p in policies:
        if p == RANDOM:
            continue
        margin = (curve[p] - curve[RANDOM]).min()
        out.append(
            CheckResult(
                bool(margin >= -eps),
                f"4. {p} at or above random at every capacity (smallest margin {margin:+.4f})",
            )
        )
    for p in curve.columns:
        steps = np.diff(curve[p].to_numpy())
        full = float(curve[p].loc[1.0])
        out.append(
            CheckResult(
                bool((steps >= -eps).all() and abs(full - 1.0) <= eps),
                f"5. {p} non-decreasing in capacity and 1.0 at full capacity ({full:.6f})",
            )
        )
    for c in balance_capacities:
        diff = abs(float(curve["by_balance"].loc[c]) - float(balance.loc[c]))
        out.append(
            CheckResult(
                diff <= balance_tolerance,
                f"6. by_balance ({curve['by_balance'].loc[c]:.4f}) tracks the balance share of "
                f"the largest {c:.0%} ({balance.loc[c]:.4f}) within {balance_tolerance} "
                f"at C={c:.2f}",
            )
        )
    return out
