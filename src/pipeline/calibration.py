"""Probability calibrators and the forward-in-time check used to choose between them.

Every calibrator maps raw model scores to probabilities with ``predict``. The
choice is made on validation only: fit on its earlier months, score its last
months, and keep a method only if it beats raw scores on both Brier and ECE.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.pipeline import evaluation as ev

EPS = 1e-6
RAW = "raw"


class Calibrator(Protocol):
    def predict(self, raw: np.ndarray) -> np.ndarray: ...


def _logit(p: np.ndarray) -> np.ndarray:
    return np.asarray(logit(np.clip(p, EPS, 1 - EPS)))


class Identity:
    """Raw scores, clipped to [0, 1]."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Identity:
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(raw, 0.0, 1.0)


class Isotonic:
    """Monotone step function fit to the outcomes (flexible; can overfit a short window)."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Isotonic:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(raw))


class Platt:
    """Logistic regression on the score's log-odds: slope and intercept (2 parameters)."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Platt:
        self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw)[:, None], y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(_logit(raw)[:, None])[:, 1])


class InterceptShift:
    """Add one constant to the score's log-odds (corrects the overall level only)."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> InterceptShift:
        z = _logit(raw)

        def log_loss(c: float) -> float:
            p = np.clip(expit(z + c), EPS, 1 - EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        self.shift = float(minimize_scalar(log_loss, bounds=(-5, 5), method="bounded").x)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(expit(_logit(raw) + self.shift))


METHODS: dict[str, type] = {
    RAW: Identity,
    "isotonic": Isotonic,
    "platt": Platt,
    "intercept_shift": InterceptShift,
}


def fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray) -> Calibrator:
    """Return a fitted calibrator of the named method."""
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method}; known: {sorted(METHODS)}")
    calibrator: Calibrator = METHODS[method]().fit(raw, y)
    return calibrator


def transfer_check(
    periods: pd.Series,
    y: np.ndarray,
    raw: np.ndarray,
    methods: Sequence[str],
    holdout_months: int,
    n_bins: int,
) -> tuple[pd.DataFrame, str]:
    """Fit each method on all but the last ``holdout_months``, score those months.

    Returns the table and the chosen method: the lowest held-out ECE among
    methods that beat raw on both Brier and ECE; raw if none does.
    """
    periods = pd.to_datetime(periods).reset_index(drop=True)
    cutoff = periods.max() - pd.DateOffset(months=holdout_months)
    early, late = (periods <= cutoff).to_numpy(), (periods > cutoff).to_numpy()
    rows = []
    for method in [RAW, *[m for m in methods if m != RAW]]:
        p = fit_calibrator(method, raw[early], y[early]).predict(raw[late])
        rows.append(
            {
                "method": method,
                "fit_rows": 0 if method == RAW else int(early.sum()),
                "eval_rows": int(late.sum()),
                "eval_from": str(periods[late].min().date()),
                "eval_to": str(periods[late].max().date()),
                "brier": ev.brier(y[late], p),
                "ece": ev.expected_calibration_error(y[late], p, n_bins),
            }
        )
    table = pd.DataFrame(rows)
    raw_row = table.iloc[0]
    table["beats_raw"] = (table["brier"] < raw_row["brier"]) & (table["ece"] < raw_row["ece"])
    eligible = table[table["beats_raw"]]
    chosen = RAW if eligible.empty else str(eligible.sort_values("ece").iloc[0]["method"])
    table["chosen"] = table["method"] == chosen
    return table, chosen
