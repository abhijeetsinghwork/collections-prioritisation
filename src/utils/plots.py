"""Matplotlib figures for MLflow and the README. Each returns a Figure; callers save it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
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


# Validated categorical slots (light surface), assigned in fixed order; random and
# oracle are reference lines in neutral ink, the raw-probability variant is a
# dashed twin of its calibrated line. Identity is also carried by end labels.
POLICY_STYLE: dict[str, dict[str, object]] = {
    "by_expected_value": {"color": "#2a78d6", "linewidth": 2.6, "label": "P(roll) × balance"},
    "by_dpd": {"color": "#eb6834", "linewidth": 2.0, "label": "days past due"},
    "by_prob": {"color": "#1baf7a", "linewidth": 2.0, "label": "P(roll)"},
    "by_balance": {"color": "#eda100", "linewidth": 2.0, "label": "balance"},
    "by_expected_value_raw": {
        "color": "#2a78d6",
        "linewidth": 1.4,
        "linestyle": (0, (4, 3)),
        "label": "P(roll) × balance, raw P",
    },
    "random": {"color": "#8a8985", "linewidth": 1.4, "label": "random"},
    "oracle": {"color": "#2b2b29", "linewidth": 1.4, "linestyle": (0, (1, 2)), "label": "oracle"},
}
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _recessive_axes(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, length=0)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def capture_curve_plot(curve: pd.DataFrame, title: str, max_capacity: float) -> Figure:
    """Return capture rate vs capacity, one line per policy, labelled at the right edge."""
    shown = curve[curve.index <= max_capacity + 1e-9]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _recessive_axes(ax)
    ends: list[tuple[float, str]] = []
    handles, labels = [], []
    for name, style in POLICY_STYLE.items():
        if name not in shown:
            continue
        kw = {k: v for k, v in style.items() if k != "label"}
        x, y = shown.index.to_numpy() * 100, shown[name].to_numpy() * 100
        handles += ax.plot(x, y, solid_capstyle="round", **kw)  # type: ignore[arg-type]
        labels.append(str(style["label"]))
        ends.append((float(y[-1]), str(style["label"])))
    # End labels in text ink, nudged apart only when they would overlap.
    ends.sort()
    gap = 2.6
    placed: list[float] = []
    for y_end, label in ends:
        y_text = max(y_end, placed[-1] + gap) if placed else y_end
        placed.append(y_text)
        ax.annotate(
            label,
            xy=(shown.index[-1] * 100, y_end),
            xytext=(shown.index[-1] * 100 + 1.2, y_text),
            va="center",
            fontsize=9,
            color=_INK,
            arrowprops={"arrowstyle": "-", "color": _GRID, "linewidth": 0.8},
        )
    ax.set_xlim(shown.index[0] * 100, shown.index[-1] * 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("capacity: share of the delinquent queue contacted (%)", color=_MUTED)
    ax.set_ylabel("share of deteriorating balance captured (%)", color=_MUTED)
    ax.set_title(title, loc="left", color=_INK, fontsize=11)
    ax.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        labelcolor=_INK,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)  # room for the end labels
    return fig


def monthly_capture_plot(monthly: pd.DataFrame, capacity: float, title: str) -> Figure:
    """Return each month's capture rate at one capacity, one line per policy."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _recessive_axes(ax)
    x = pd.to_datetime(monthly.index)
    names = [n for n in POLICY_STYLE if n in monthly and n != "by_expected_value_raw"]
    handles = []
    for name in names:
        kw = {k: v for k, v in POLICY_STYLE[name].items() if k != "label"}
        handles += ax.plot(
            x,
            monthly[name] * 100,
            marker="o",
            markersize=4,
            **kw,  # type: ignore[arg-type]
        )
    ax.axhline(capacity * 100, color=_GRID, linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"capture at C = {capacity:.0%} (%)", color=_MUTED)
    ax.set_title(title, loc="left", color=_INK, fontsize=11)
    ax.legend(
        handles=handles,
        labels=[str(POLICY_STYLE[n]["label"]) for n in names],
        frameon=False,
        fontsize=8.5,
        ncols=len(names),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        labelcolor=_INK,
    )
    fig.tight_layout()
    return fig
