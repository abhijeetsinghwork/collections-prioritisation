"""Stage 6 - drift study: how the frozen model degrades through the 2020-21 shock.

Run:
  python -m src.s6_drift

Compares every monitored month (2018-19 baseline, then 2020-21) with the
training distribution: PSI per kept feature and for the model score, plus
within-month AUC and calibration. Nothing is fitted or chosen: the frozen
Stage 4 model is only scored. Each series gets a band from the baseline months,
and "degrades first" is the first 2020-21 month that starts a sustained run
outside that band (rule fixed in config before any drift metric was computed).
"""

from __future__ import annotations

import sys
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.pipeline import drift as dr
from src.pipeline import evaluation as ev
from src.pipeline import modeling as md
from src.pipeline.labels import LABEL
from src.s4_models import (
    POOLED,
    frozen_config_hash,
    git_state,
    load_frozen,
    log_table,
    read_split,
    section,
    setup_mlflow,
)
from src.s5_policy import read_scores
from src.utils import plots
from src.utils.config import Config, load_config

PERIOD = "reporting_period"
KEYS = ["loan_sequence_number", PERIOD]
SCORE = "model_score"
# Series monitored for degradation, and which direction is bad.
SERIES: dict[str, str] = {
    "score_psi": "up",
    "median_feature_psi": "up",
    "auc": "down",
    "abs_gap": "up",
    "ece": "up",
}


def logged_test_auc(cfg: Config, model: str) -> float:
    """Return the most recent logged Stage 4 test AUC for ``model``."""
    log = pd.read_csv(cfg.paths.test_log)
    return float(log[log["model"] == model].iloc[-1]["auc"])


def monitored_frame(cfg: Config, prob: str, raw: str, kept: list[str]) -> pd.DataFrame:
    """Return monitored-split features joined to the frozen scores, one row per account-month."""
    frames = []
    for split in cfg.drift.monitored_splits:
        feats = read_split(cfg, split)[[*KEYS, *kept]]
        scores = read_scores(cfg, split)[[*KEYS, LABEL, prob, raw]]
        feats[PERIOD] = feats[PERIOD].astype(str)
        scores[PERIOD] = scores[PERIOD].astype(str)
        joined = feats.merge(scores, on=KEYS, how="inner", validate="one_to_one")
        if len(joined) != len(scores) or len(joined) != len(feats):
            raise RuntimeError(f"{split}: features and scores do not line up one to one")
        frames.append(joined.assign(split=split))
    return pd.concat(frames, ignore_index=True)


def run(cfg: Config) -> bool:
    """Compute drift and monthly metrics, write tables and figures. True if checks pass."""
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    dc = cfg.drift
    bundle, manifest = load_frozen(cfg)
    if manifest["config_hash"] != frozen_config_hash(cfg):
        print("refusing: model config changed since the freeze; re-run stage 4")
        return False
    chosen = manifest["chosen_model"]
    if chosen != POOLED:
        raise NotImplementedError(f"score PSI is implemented for {POOLED}, not {chosen}")
    prob, raw = f"p_{chosen}", f"score_{chosen}"
    kept = list(manifest["features"])

    section("Inputs")
    reference = read_split(cfg, dc.reference_split)[kept]
    reference[SCORE] = md.predict_lightgbm(bundle[POOLED], bundle["spec"], reference)
    monitored = monitored_frame(cfg, prob, raw, kept).rename(columns={raw: SCORE})
    print(
        f"reference: {dc.reference_split}, {len(reference):,} rows; monitored: "
        f"{', '.join(dc.monitored_splits)}, {len(monitored):,} rows over "
        f"{monitored[PERIOD].nunique()} months; {len(kept)} features + the model score"
    )

    section("Structural checks")
    base = monitored[monitored["split"] == dc.baseline_split]
    pooled_auc = float(ev.discrimination(base[LABEL].to_numpy(), base[SCORE].to_numpy())["auc"])
    logged = logged_test_auc(cfg, chosen)
    require(
        abs(pooled_auc - logged) <= dc.auc_match_tolerance,
        f"scores are the frozen model's: pooled {dc.baseline_split} AUC {pooled_auc:.5f} "
        f"matches the logged Stage 4 test AUC {logged:.5f}",
    )
    bins = dr.fit_bins(reference, [*kept, SCORE], dc.psi_bins)
    halves = dr.split_half_psi(reference, bins, dc.psi_floor, cfg.random_seed)
    require(
        float(halves.max()) <= dc.split_half_max_psi,
        f"PSI between two random halves of {dc.reference_split} is ~0 "
        f"(max {halves.max():.5f}, {halves.idxmax()})",
    )
    expected = []
    for split in dc.monitored_splits:
        s = next(x for x in cfg.splits if x.name == split)
        expected += [str(d.date()) for d in pd.date_range(s.start_date, s.end_date, freq="MS")]
    months = sorted(monitored[PERIOD].unique())
    require(months == expected, f"every monitored month is present ({len(months)} months)")

    section("PSI and monthly metrics")
    psi = dr.psi_by_period(reference, monitored, bins, PERIOD, dc.psi_floor)
    feature_psi = psi[kept]
    require(
        bool(np.isfinite(psi.to_numpy()).all() and (psi.to_numpy() >= 0).all()),
        "every PSI is finite and non-negative",
    )
    metrics = dr.monthly_metrics(
        monitored, LABEL, prob, SCORE, PERIOD, cfg.models.calibration.reliability_bins
    )
    lo_band, hi_band = dc.psi_bands
    monitor = metrics.assign(
        abs_gap=metrics["gap"].abs(),
        score_psi=psi[SCORE],
        median_feature_psi=feature_psi.median(axis=1),
        features_psi_above_hi=(feature_psi > hi_band).sum(axis=1),
        features_psi_above_lo=(feature_psi > lo_band).sum(axis=1),
    )
    split_of = monitored.groupby(PERIOD)["split"].first()
    monitor.insert(0, "split", split_of)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(monitor.to_string())

    section(f"Degradation: first sustained departure from the {dc.baseline_split} band")
    is_base = monitor["split"] == dc.baseline_split
    window = monitor[~is_base]
    bands: dict[str, tuple[float, float]] = {}
    departures: dict[str, object | None] = {}
    rows: list[dict[str, Any]] = []
    for name, direction in SERIES.items():
        band = dr.baseline_band(monitor.loc[is_base, name], dc.band_sd)
        first = dr.first_departure(
            window[name],
            band,
            "up" if direction == "up" else "down",  # narrows str to the Literal
            dc.consecutive_months,
        )
        bands[name], departures[name] = (band.lo, band.hi), first
        rows.append(
            {
                "series": name,
                "bad_direction": direction,
                "band_lo": band.lo,
                "band_hi": band.hi,
                "first_departure": first,
                "value_then": float(window[name].loc[str(first)]) if first else np.nan,
                "worst_in_window": float(
                    window[name].max() if direction == "up" else window[name].min()
                ),
            }
        )
        print(f"{name:20s} band [{band.lo:.4f}, {band.hi:.4f}]  first departure: {first}")
    departure_table = pd.DataFrame(rows)

    section("Features that moved most (2020-21)")
    drift_months = feature_psi.loc[window.index]
    top = drift_months.max().sort_values(ascending=False).head(dc.heatmap_top_n)
    peak = drift_months[top.index].idxmax()
    top_table = pd.DataFrame(
        {"feature": top.index, "max_psi": top.to_numpy(), "peak_month": peak.to_numpy()}
    )
    print(top_table.to_string(index=False, float_format="{:.3f}".format))

    section("Write")
    tables = {
        "drift_monitor_monthly": monitor.reset_index(),
        "drift_psi_monthly": psi.reset_index(),
        "drift_first_departure": departure_table,
        "drift_top_features": top_table,
        "drift_split_half_psi": halves.rename("psi").rename_axis("feature").reset_index(),
    }
    paths = [log_table(name, t, cfg) for name, t in tables.items()]
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "drift_monitor.png": plots.drift_monitor_plot(
            monitor,
            bands,
            departures,
            "Frozen model through the 2020-21 shock: input drift, discrimination, calibration",
        ),
        "drift_psi_heatmap.png": plots.psi_heatmap(
            feature_psi[top.index],
            f"PSI vs training, top {len(top)} features by peak 2020-21 PSI",
        ),
    }
    for name, fig in figures.items():
        fig.savefig(cfg.paths.figures_dir / name, dpi=150)
    print(f"tables -> {cfg.paths.tables_dir}, figures -> {cfg.paths.figures_dir}")

    setup_mlflow(cfg)
    git = git_state()
    with mlflow.start_run(run_name="stage6-drift"):
        mlflow.set_tags(
            {
                **git,
                "phase": "drift",
                "config_hash": manifest["config_hash"],
                "chosen_model": chosen,
            }
        )
        mlflow.log_params(dc.model_dump(mode="json"))
        mlflow.log_artifact("config/config.yaml")
        for r in rows:
            mlflow.log_metric(f"worst_{r['series']}", float(r["worst_in_window"]))
        for p in paths:
            mlflow.log_artifact(str(p))
        for name, fig in figures.items():
            mlflow.log_figure(fig, name)
    return not failures


def main() -> int:
    cfg = load_config()
    marker = cfg.paths.models_dir.parent / ".s6_drift.done"
    marker.unlink(missing_ok=True)
    ok = run(cfg)
    section("Result")
    if ok:
        marker.write_text("ok\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
        return 0
    print("CHECKS FAILED - no .done marker written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
