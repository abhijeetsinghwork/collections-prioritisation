"""Drift monitoring: population stability (PSI) and monthly model metrics.

Bins are fixed on the reference (training) distribution and reused for every
monitored month, so a month's PSI measures how far it has moved from what the
model was trained on. Pure numpy/pandas; no data loading here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.pipeline import evaluation as ev
from src.pipeline.feature_audit import is_categorical

NULL_BIN = "__null__"
OTHER_BIN = "__other__"


@dataclass(frozen=True)
class Bins:
    """Reference bins for one feature: numeric edges, or categorical levels."""

    edges: np.ndarray | None = None  # interior cut points (numeric)
    levels: tuple[str, ...] | None = None  # categories seen in the reference

    @classmethod
    def fit(cls, reference: pd.Series, n_bins: int) -> Bins:
        """Return bins from the reference: quantile cut points, or the observed levels."""
        if is_categorical(reference):
            return cls(levels=tuple(sorted(reference.dropna().astype(str).unique())))
        values = reference.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return cls(edges=np.array([]))
        qs = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
        return cls(edges=np.unique(qs))

    def labels(self) -> list[str]:
        """Return every bin label, including the null bin (and the unseen-category bin)."""
        if self.levels is not None:
            return [*self.levels, OTHER_BIN, NULL_BIN]
        assert self.edges is not None
        return [*[f"b{i}" for i in range(len(self.edges) + 1)], NULL_BIN]

    def assign(self, values: pd.Series) -> pd.Series:
        """Return the bin label of every value."""
        out = pd.Series(NULL_BIN, index=values.index, dtype=object)
        present = values.notna()
        if self.levels is not None:
            v = values[present].astype(str)
            out[present] = v.where(v.isin(self.levels), OTHER_BIN)
            return out
        assert self.edges is not None
        idx = np.searchsorted(self.edges, values[present].to_numpy(dtype=float), side="right")
        out[present] = pd.Series([f"b{i}" for i in idx], index=values.index[present])
        return out

    def shares(self, values: pd.Series) -> np.ndarray:
        """Return the share of ``values`` in each bin, in ``labels()`` order."""
        counts = self.assign(values).value_counts()
        return counts.reindex(self.labels(), fill_value=0).to_numpy(dtype=float) / len(values)


def psi(expected: np.ndarray, actual: np.ndarray, floor: float) -> float:
    """Return the population stability index of ``actual`` against ``expected`` shares.

    Shares below ``floor`` are raised to it so an empty bin does not make PSI infinite.
    """
    e = np.maximum(np.asarray(expected, dtype=float), floor)
    a = np.maximum(np.asarray(actual, dtype=float), floor)
    return float(np.sum((a - e) * np.log(a / e)))


def fit_bins(reference: pd.DataFrame, features: Sequence[str], n_bins: int) -> dict[str, Bins]:
    """Return reference bins for each feature."""
    return {f: Bins.fit(reference[f], n_bins) for f in features}


def psi_by_period(
    reference: pd.DataFrame,
    monitored: pd.DataFrame,
    bins: dict[str, Bins],
    period: str,
    floor: float,
) -> pd.DataFrame:
    """Return PSI of every binned column per monitored period (index period, column per feature)."""
    expected = {f: b.shares(reference[f]) for f, b in bins.items()}
    rows = {}
    for p, month in monitored.groupby(period, sort=True):
        rows[p] = {f: psi(expected[f], b.shares(month[f]), floor) for f, b in bins.items()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = period
    return out


def split_half_psi(
    reference: pd.DataFrame, bins: dict[str, Bins], floor: float, seed: int
) -> pd.Series:
    """Return per-feature PSI between two random halves of the reference: should be ~0."""
    mask = np.random.default_rng(seed).random(len(reference)) < 0.5
    a, b = reference[mask], reference[~mask]
    return pd.Series({f: psi(bn.shares(a[f]), bn.shares(b[f]), floor) for f, bn in bins.items()})


def monthly_metrics(
    frame: pd.DataFrame, label: str, prob: str, raw: str, period: str, n_bins: int
) -> pd.DataFrame:
    """Return per-period n, roll rate, mean predicted, calibration gap, ECE and AUC.

    ``gap`` is mean predicted minus actual roll rate (calibration in the large):
    positive means the model over-predicts. AUC is on the raw score (calibration
    maps are monotone, so it is the same for the frozen probability).
    """
    rows = []
    for p, month in frame.groupby(period, sort=True):
        y = month[label].to_numpy()
        pr, rw = month[prob].to_numpy(), month[raw].to_numpy()
        row = {
            period: p,
            "n": len(month),
            "roll_rate": float(y.mean()),
            "mean_pred": float(pr.mean()),
            "gap": float(pr.mean() - y.mean()),
            "gap_raw": float(rw.mean() - y.mean()),
            "ece": ev.expected_calibration_error(y, pr, n_bins),
            "auc": float("nan"),
        }
        if len(np.unique(y)) == 2:
            row["auc"] = ev.discrimination(y, rw)["auc"]
        rows.append(row)
    return pd.DataFrame(rows).set_index(period)


@dataclass(frozen=True)
class Band:
    lo: float
    hi: float


def baseline_band(baseline: pd.Series, k: float) -> Band:
    """Return mean +/- k standard deviations of a series over the baseline periods."""
    mu, sd = float(baseline.mean()), float(baseline.std(ddof=1))
    return Band(mu - k * sd, mu + k * sd)


def first_departure(
    series: pd.Series,
    band: Band,
    direction: Literal["up", "down"],
    consecutive: int,
) -> object | None:
    """Return the first index that starts ``consecutive`` periods outside ``band``.

    Only departures in ``direction`` count (``up``: above ``band.hi``; ``down``:
    below ``band.lo``). Missing values never count as a departure. None if no run.
    """
    values = series.to_numpy(dtype=float)
    out = values > band.hi if direction == "up" else values < band.lo
    out &= ~np.isnan(values)
    for i in range(len(out) - consecutive + 1):
        if out[i : i + consecutive].all():
            return series.index[i]
    return None
