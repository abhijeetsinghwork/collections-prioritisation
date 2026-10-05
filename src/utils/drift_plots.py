"""Drift-study figures (Stage 6). Kept apart from plots.py so that adding them
does not change the content stamp of the stages that use plots.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.utils.plots import _INK, _MUTED, _recessive_axes

_SERIES_1, _SERIES_2 = "#2a78d6", "#eb6834"
_BAND = "#f0efec"
# Sequential blue ramp (light -> dark), from the validated reference palette.
_SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def drift_monitor_plot(
    monitor: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    departures: dict[str, object | None],
    title: str,
) -> Figure:
    """Return three time-aligned panels: PSI, AUC, predicted vs actual roll rate.

    ``monitor`` is indexed by month. ``bands`` gives the baseline band per
    column; ``departures`` the first sustained departure month per column.
    """
    x = pd.to_datetime(monitor.index)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True)
    panels = [
        (
            "PSI vs training",
            [("score_psi", "model score"), ("median_feature_psi", "median feature")],
        ),
        ("AUC (within month)", [("auc", "AUC")]),
        ("roll rate", [("roll_rate", "actual"), ("mean_pred", "predicted (frozen)")]),
    ]
    band_col = {0: "score_psi", 1: "auc", 2: None}
    for i, (ax, (ylabel, lines)) in enumerate(zip(axes, panels, strict=True)):
        _recessive_axes(ax)
        col = band_col[i]
        if col and col in bands:
            lo, hi = bands[col]
            ax.axhspan(lo, hi, color=_BAND, zorder=0, linewidth=0)
        for j, (name, label) in enumerate(lines):
            ax.plot(
                x,
                monitor[name],
                color=[_SERIES_1, _SERIES_2][j],
                linewidth=2,
                marker="o",
                markersize=3.5,
                label=label,
            )
        ax.set_ylabel(ylabel, color=_MUTED)
        if len(lines) > 1:
            ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=_INK)
    for name, month in departures.items():
        if month is None:
            continue
        ax_i = {"score_psi": 0, "median_feature_psi": 0, "auc": 1, "abs_gap": 2, "ece": 2}[name]
        axes[ax_i].axvline(pd.Timestamp(str(month)), color=_MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_title(title, loc="left", color=_INK, fontsize=11)
    axes[-1].set_xlabel("month (shaded: baseline band from 2018-19)", color=_MUTED)
    fig.tight_layout()
    return fig


def psi_heatmap(psi: pd.DataFrame, title: str) -> Figure:
    """Return a features x months heatmap of PSI on a single-hue sequential scale."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(psi.columns) + 1.6))
    data = psi.T.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, interpolation="nearest")
    months = pd.to_datetime(psi.index)
    ticks = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(ticks, [months[i].strftime("%Y-%m") for i in ticks], color=_MUTED, fontsize=8)
    ax.set_yticks(range(len(psi.columns)), list(psi.columns), color=_INK, fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("PSI vs training", color=_MUTED)
    bar.outline.set_visible(False)
    ax.set_title(title, loc="left", color=_INK, fontsize=11)
    fig.tight_layout()
    return fig
