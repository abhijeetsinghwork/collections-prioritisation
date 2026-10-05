"""Matplotlib figures for MLflow and the README. Each returns a Figure; callers save it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def reliability_plot(tables: dict[str, pd.DataFrame], title: str) -> Figure:
    """Return a reliability diagram with one line per labelled reliability table."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    for label, t in tables.items():
        t = t.dropna(subset=["mean_pred"])
        ax.plot(t["mean_pred"], t["actual"], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("mean predicted P(roll)")
    ax.set_ylabel("actual roll rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def importance_plot(importance: pd.DataFrame, top_n: int, title: str) -> Figure:
    """Return a horizontal bar chart of the top features by gain share."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(top) + 1))
    ax.barh(top["feature"], top["gain_share"], color="#4c72b0")
    ax.set_xlabel("share of total gain")
    ax.set_title(title)
    fig.tight_layout()
    return fig
