"""Stage 5 - policy simulation: who to contact when only C% of the queue can be called.

Run:
  python -m src.s5_policy                     # the configured split (test): the deliverable
  python -m src.s5_policy --split validation  # rehearse the mechanics without touching test

Reads the per-row scores the frozen Stage 4 models wrote, so no model is fitted
or chosen here. Each month, every policy ranks the same delinquent accounts
from a frame with the label removed; the label is used only to score the
ranking afterwards. Refuses to run if the scores are older than the frozen
models or the model config changed since the freeze.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import mlflow
import pandas as pd

from src.pipeline import policy as pol
from src.pipeline.labels import LABEL
from src.s4_models import (
    frozen_config_hash,
    git_state,
    load_manifest,
    log_table,
    section,
    setup_mlflow,
)
from src.utils import plots
from src.utils.config import Config, load_config

HEADLINE_FLOORS = ["by_dpd", "by_balance"]


def read_scores(cfg: Config, split: str) -> pd.DataFrame:
    """Return the frozen models' per-row scores for one split, refusing stale ones."""
    manifest_path = cfg.paths.models_dir / "manifest.json"
    path = cfg.paths.scores_dir / f"split={split}" / "scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run stage 4 (and --evaluate-test for test)")
    if path.stat().st_mtime < manifest_path.stat().st_mtime:
        raise RuntimeError(f"{path} is older than the frozen models: re-run stage 4 scoring")
    return pd.read_parquet(path)


def capture_table(curve: pd.DataFrame, capacities: list[float], headline: str) -> pd.DataFrame:
    """Return capture per policy at the table capacities, plus the share of gap closed."""
    rows: list[dict[str, Any]] = []
    for c in capacities:
        row: dict[str, Any] = {"capacity": c, **{str(k): v for k, v in curve.loc[c].items()}}
        for floor in HEADLINE_FLOORS:
            row[f"{headline}_gap_closed_vs_{floor}"] = float(
                pol.gap_closed(curve, headline, floor).loc[c]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(monthly: pd.DataFrame, headline: str) -> pd.DataFrame:
    """Return mean, spread and range of each policy's monthly capture at one capacity."""
    stats = monthly.agg(["mean", "std", "min", "max"]).T
    for floor in HEADLINE_FLOORS:
        stats.loc[headline, f"months_above_{floor}"] = int(
            (monthly[headline] > monthly[floor]).sum()
        )
    stats["months"] = len(monthly)
    return stats.reset_index(names="policy")


def run(cfg: Config, split: str) -> bool:
    """Simulate every policy on ``split``, write tables and figures. True if checks pass."""
    pc = cfg.policy
    manifest = load_manifest(cfg)
    if manifest["config_hash"] != frozen_config_hash(cfg):
        print("refusing: model config changed since the freeze; re-run stage 4")
        return False
    chosen = manifest["chosen_model"]
    prob_col, raw_col = f"p_{chosen}", f"score_{chosen}"
    scores = read_scores(cfg, split)
    frame = scores[[pol.PERIOD, "dq_bucket", pol.EXPOSURE, prob_col, raw_col, LABEL]]

    section(f"Inputs ({split})")
    per_month = frame.groupby(pol.PERIOD).size()
    risk_share = (frame[pol.EXPOSURE] * frame[LABEL]).sum() / frame[pol.EXPOSURE].sum()
    print(
        f"{len(frame):,} account-months over {len(per_month)} months "
        f"({per_month.min():,}-{per_month.max():,} per month); "
        f"model {chosen}, P(roll) = {prob_col} ({manifest['calibration_method']}), "
        f"raw = {raw_col}"
    )
    print(
        f"roll rate {frame[LABEL].mean():.4f}; "
        f"balance that rolled: {risk_share:.4f} of total balance"
    )

    section("Simulate")
    capacities = pc.capacities()
    seeds = [cfg.random_seed + i for i in range(pc.random_seeds)]
    policies = pol.standard_policies(prob_col, raw_col)
    sim = pol.simulate(frame, policies, capacities, seeds)
    curve = pol.capture_curve(sim)
    order = [p.name for p in policies] + [pol.ORACLE]
    curve = curve[order]
    balance = pol.balance_share(frame, capacities)
    print(f"{len(policies)} policies + oracle, {len(capacities)} capacities, {len(seeds)} seeds")

    section("Validity checks (structural; any failure is a bug)")
    checks = pol.validity_checks(
        curve,
        balance,
        pc.random_capture_tolerance,
        pc.balance_share_tolerance,
        pc.table_capacities,
    )
    for check in checks:
        print(("PASS  " if check.ok else "FAIL  ") + check.message)

    section("Capture at the table capacities")
    headline = pc.headline_policy
    table = capture_table(curve, pc.table_capacities, headline)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.set_index("capacity").T.to_string())
    print("\nbalance share of the largest accounts (check 6):")
    print(balance.loc[pc.table_capacities].to_string(float_format="{:.4f}".format))
    largest = pol.largest_accounts_roll_rate(frame, pc.headline_capacity)
    print(
        f"roll rate of the largest {pc.headline_capacity:.0%} by balance "
        f"{largest['largest']:.4f} vs the rest {largest['rest']:.4f}: by_balance sits below "
        "the balance share when large accounts roll less often, and above it when they roll more"
    )

    section(f"Month by month at C = {pc.headline_capacity:.0%}")
    monthly = pol.monthly_capture(sim, pc.headline_capacity)[order]
    stability = stability_table(monthly, headline)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(monthly.to_string())
        print()
        print(stability.to_string(index=False))

    section("Calibrated vs raw probabilities (Stage 4 accepted calibration failure)")
    for c in pc.table_capacities:
        cal_v, raw_v = curve["by_expected_value"].loc[c], curve["by_expected_value_raw"].loc[c]
        print(
            f"C={c:.0%}: frozen ({manifest['calibration_method']}) {cal_v:.4f}, "
            f"raw {raw_v:.4f}, difference {cal_v - raw_v:+.4f}"
        )

    section("Write")
    suffix = f"_{split}"
    checks_df = pd.DataFrame([{"ok": k.ok, "check": k.message} for k in checks])
    tables = {
        f"policy_capture{suffix}": table,
        f"policy_curve{suffix}": curve.reset_index(),
        f"policy_monthly{suffix}": monthly.reset_index(),
        f"policy_stability{suffix}": stability,
        f"policy_checks{suffix}": checks_df,
    }
    paths = [log_table(name, t, cfg) for name, t in tables.items()]
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    window = next(s for s in cfg.splits if s.name == split)
    fig_curve = plots.capture_curve_plot(
        curve,
        "Share of deteriorating balance captured, by contact policy\n"
        f"Delinquent mortgages, {window.start} to {window.end} ({split} window)",
        pc.capacity_grid.stop,
    )
    fig_month = plots.monthly_capture_plot(
        monthly, pc.headline_capacity, f"Capture at C = {pc.headline_capacity:.0%}, by month"
    )
    figures = {f"capture_curve{suffix}.png": fig_curve, f"monthly_capture{suffix}.png": fig_month}
    for name, fig in figures.items():
        fig.savefig(cfg.paths.figures_dir / name, dpi=150)
    print(f"tables -> {cfg.paths.tables_dir}, figures -> {cfg.paths.figures_dir}")

    setup_mlflow(cfg)
    git = git_state()
    with mlflow.start_run(run_name=f"stage5-policy-{split}"):
        mlflow.set_tags(
            {
                **git,
                "phase": "policy",
                "split": split,
                "config_hash": manifest["config_hash"],
                "chosen_model": chosen,
                "calibration_method": manifest["calibration_method"],
            }
        )
        mlflow.log_params(pc.model_dump(mode="json", exclude={"capacity_grid"}))
        mlflow.log_artifact("config/config.yaml")
        for c in pc.table_capacities:
            for name in curve.columns:
                mlflow.log_metric(f"capture_{name}_c{round(c * 100)}", float(curve[name].loc[c]))
        for p in paths:
            mlflow.log_artifact(str(p))
        for name, fig in figures.items():
            mlflow.log_figure(fig, name)
    return all(k.ok for k in checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", help="split to simulate (default: policy.split in config)")
    args = parser.parse_args(argv)
    cfg = load_config()
    split = args.split or cfg.policy.split
    marker = cfg.paths.models_dir.parent / ".s5_policy.done"
    deliverable = split == cfg.policy.split
    if deliverable:
        marker.unlink(missing_ok=True)
    ok = run(cfg, split)
    section("Result")
    if not ok:
        print("CHECKS FAILED - no .done marker written")
        return 1
    if deliverable:
        marker.write_text("ok\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
    else:
        print(f"ALL CHECKS PASSED on {split} (rehearsal: no marker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
