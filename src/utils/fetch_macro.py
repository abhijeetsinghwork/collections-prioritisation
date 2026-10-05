"""Download FRED macro series as CSV into data/raw/macro/ (no API key needed).

Run: python -m src.utils.fetch_macro [--force]
Series that FRED does not publish (e.g. house prices for some territories)
are reported and skipped; the matching features are null for those states.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.utils.config import Config, load_config

TIMEOUT_SECONDS = 60


def series_ids(cfg: Config, states: list[str]) -> list[str]:
    """Return every FRED series id the macro features need."""
    macro = cfg.features.macro
    ids = [t.format(state=s) for t in macro.state_series.values() for s in states]
    return sorted(set(ids) | set(macro.national_series.values()))


def fetch_csv(url: str) -> str | None:
    """Return the CSV text at ``url``, or None when FRED has no such series."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    # FRED answers an unknown id with an HTML error page, not a 404.
    return text if text.startswith("observation_date,") else None


def panel_states(cfg: Config) -> list[str]:
    """Return the property states present in the Stage 1 panel."""
    import pyarrow.dataset as ds

    table = ds.dataset(cfg.paths.panel_dir, format="parquet", partitioning="hive").to_table(
        columns=["property_state"]
    )
    return sorted(s for s in set(table.column("property_state").to_pylist()) if s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    cfg = load_config()
    out_dir: Path = cfg.paths.macro_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for sid in series_ids(cfg, panel_states(cfg)):
        path = out_dir / f"{sid}.csv"
        if path.exists() and not args.force:
            continue
        text = fetch_csv(cfg.features.macro.url.format(series=sid))
        if text is None:
            missing.append(sid)
            continue
        path.write_text(text)
    print(f"macro series in {out_dir}: {len(list(out_dir.glob('*.csv')))}")
    print(f"not published by FRED (features will be null): {missing or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
