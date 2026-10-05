"""Model metrics: discrimination, calibration tables, per-bucket breakdowns.

Pure numpy/pandas on predictions; no model or data-loading code here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def discrimination(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Return AUC and PR-AUC of ``score`` against binary ``y``."""
    return {
        "auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Return the Brier score of probabilities ``p``."""
    return float(brier_score_loss(y, p))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Return rows per equal-width probability bin: count, mean predicted, actual rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"))
    out = out.reindex(range(n_bins))
    out.insert(0, "bin_lo", edges[:-1])
    out.insert(1, "bin_hi", edges[1:])
    out["n"] = out["n"].fillna(0).astype(int)
    return out.reset_index(drop=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Return the count-weighted mean |predicted - actual| over equal-width bins."""
    t = reliability_table(y, p, n_bins).dropna()
    return float((t["n"] * (t["mean_pred"] - t["actual"]).abs()).sum() / t["n"].sum())


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Return predicted vs actual roll rate per predicted-score quantile (1 = highest risk)."""
    ranks = pd.Series(p).rank(method="first", ascending=False)
    decile = np.ceil(ranks / len(p) * n_bins).astype(int).clip(1, n_bins)
    df = pd.DataFrame({"decile": decile, "y": y, "p": p})
    out = df.groupby("decile").agg(n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"))
    out["gap"] = out["mean_pred"] - out["actual"]
    return out.reset_index()


def per_bucket(
    y: np.ndarray, score: np.ndarray, bucket: np.ndarray, buckets: list[int]
) -> dict[int, dict[str, float]]:
    """Return AUC, PR-AUC, row count and base rate within each dq_bucket at T."""
    out: dict[int, dict[str, float]] = {}
    for b in buckets:
        mask = bucket == b
        yb, sb = y[mask], score[mask]
        row: dict[str, float] = {"n": float(mask.sum()), "base_rate": float(yb.mean())}
        if len(np.unique(yb)) == 2:
            row.update(discrimination(yb, sb))
        out[b] = row
    return out


def flatten_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    """Return ``metrics`` with keys prefixed, for MLflow."""
    return {f"{prefix}_{k}": v for k, v in metrics.items()}
