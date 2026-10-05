"""Macro overlay: FRED series as monthly features, lagged to what was public at month T.

Pure pandas on small frames (states x months), joined to the Spark feature
table at the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.utils.config import MacroConfig

# Series whose change is measured in percent; the others in points.
PCT_CHANGE = {"house_price_index"}

# A quarterly value dated at quarter start also stands for the next 2 months.
QUARTER_FILL = 2


def read_series(path: Path) -> pd.Series:
    """Return a FRED CSV as a float series indexed by observation date (missing '.' -> NaN)."""
    df = pd.read_csv(path, parse_dates=["observation_date"], na_values=["."])
    return df.set_index("observation_date").iloc[:, 0].astype(float)


def to_monthly(series: pd.Series) -> pd.Series:
    """Return the series on a first-of-month index.

    Weekly values are averaged within the month; quarterly values are carried
    forward to the months of their quarter. Monthly series pass through.
    """
    monthly = series.resample("MS").mean()
    # Extend so the last quarter of a quarterly series also covers its later months.
    end = monthly.index.max() + pd.DateOffset(months=QUARTER_FILL)
    monthly = monthly.reindex(pd.date_range(monthly.index.min(), end, freq="MS"))
    return monthly.ffill(limit=QUARTER_FILL)


def lagged_features(
    monthly: pd.Series, name: str, lag: int, change_windows: Sequence[int]
) -> pd.DataFrame:
    """Return level and change features for month T using only values from T-lag or earlier.

    ``<name>`` at T is the value observed at T-lag; ``<name>_chg_{n}m`` compares it
    with the value at T-lag-n.
    """
    level = monthly.shift(lag, freq="MS")
    out = pd.DataFrame({name: level})
    for n in change_windows:
        before = level.shift(n, freq="MS").reindex(level.index)
        if name in PCT_CHANGE:
            out[f"{name}_chg_{n}m"] = (level / before - 1.0) * 100.0
        else:
            out[f"{name}_chg_{n}m"] = level - before
    out.index.name = "reporting_period"
    return out


def state_macro_table(macro_dir: Path, states: Sequence[str], cfg: MacroConfig) -> pd.DataFrame:
    """Return one row per (property_state, reporting_period) with lagged state features.

    States with no published series get no rows, so their features are null after the join.
    """
    frames = []
    for state in states:
        parts = []
        for name, template in cfg.state_series.items():
            path = macro_dir / f"{template.format(state=state)}.csv"
            if not path.exists():
                continue
            parts.append(
                lagged_features(
                    to_monthly(read_series(path)),
                    name,
                    cfg.publication_lag_months[name],
                    cfg.change_windows,
                )
            )
        if parts:
            df = pd.concat(parts, axis=1)
            df["property_state"] = state
            frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)


def national_macro_table(macro_dir: Path, cfg: MacroConfig) -> pd.DataFrame:
    """Return one row per reporting_period with lagged national features."""
    parts = [
        lagged_features(
            to_monthly(read_series(macro_dir / f"{sid}.csv")),
            name,
            cfg.publication_lag_months[name],
            cfg.change_windows,
        )
        for name, sid in cfg.national_series.items()
    ]
    return pd.concat(parts, axis=1).reset_index()
