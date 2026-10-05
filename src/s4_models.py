"""Stage 4 - baselines, LightGBM (pooled and segmented), isotonic calibration.

Run:
  python -m src.s4_models                  # fit on train; select, calibrate on validation; freeze
  python -m src.s4_models --evaluate-test  # score the frozen models on test (logged, append-only)

The fit phase never reads the test split. The test phase refuses to run if the
feature list or model config differs from what was frozen, and appends every
evaluation to outputs/tables/test_evaluations.csv so repeated peeking is visible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from src.pipeline import evaluation as ev  # noqa: E402
from src.pipeline import modeling as md  # noqa: E402
from src.pipeline.features import feature_registry  # noqa: E402
from src.pipeline.labels import LABEL  # noqa: E402
from src.utils import leakage, plots  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402

BASELINE = "baseline_dq_bucket"
WOE = "woe_logistic"
POOLED = "lgbm_pooled"
SEGMENTED = "lgbm_segmented"
PROB_MODELS = [WOE, POOLED, SEGMENTED]
SCORE_KEYS = ["loan_sequence_number", "reporting_period", "split", "dq_bucket", "exposure", LABEL]


# --- small helpers ------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def git_state() -> dict[str, str]:
    """Return the current commit SHA and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout

    sha = git("rev-parse", "HEAD").strip()
    dirty = git("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": str(bool(dirty.strip()))}


def frozen_config_hash(cfg: Config) -> str:
    """Return a hash of what defines the frozen models: kept features, model config, seed."""
    payload = {
        "kept": cfg.features.kept(),
        "models": cfg.models.model_dump(mode="json"),
        "seed": cfg.random_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def setup_mlflow(cfg: Config) -> None:
    """Point MLflow at the local SQLite store and create the experiment if needed."""
    tc = cfg.tracking
    tc.mlflow_artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tc.mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(tc.mlflow_experiment) is None:
        mlflow.create_experiment(
            tc.mlflow_experiment, artifact_location=tc.mlflow_artifact_dir.resolve().as_uri()
        )
    mlflow.set_experiment(tc.mlflow_experiment)


def read_split(cfg: Config, split: str) -> pd.DataFrame:
    """Return one split of the feature table as pandas."""
    dataset = ds.dataset(cfg.paths.features_dir, format="parquet", partitioning="hive")
    return dataset.to_table(filter=ds.field("split") == split).to_pandas()


def models_dir(cfg: Config) -> Path:
    return cfg.paths.models_dir


def save_frozen(cfg: Config, bundle: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Write boosters, sklearn objects and the manifest that pins them."""
    out = models_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    bundle[POOLED].save_model(str(out / f"{POOLED}.txt"))
    for b, booster in bundle[SEGMENTED].items():
        booster.save_model(str(out / f"{SEGMENTED}_bucket{b}.txt"))
    joblib.dump(bundle[WOE], out / f"{WOE}.joblib")
    joblib.dump(bundle["calibrators"], out / "calibrators.joblib")
    joblib.dump(bundle["spec"], out / "matrix_spec.joblib")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_frozen(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the frozen bundle and its manifest."""
    out = models_dir(cfg)
    manifest = json.loads((out / "manifest.json").read_text())
    bundle: dict[str, Any] = {
        POOLED: lgb.Booster(model_file=str(out / f"{POOLED}.txt")),
        SEGMENTED: {
            int(b): lgb.Booster(model_file=str(out / f"{SEGMENTED}_bucket{b}.txt"))
            for b in manifest["segment_buckets"]
        },
        WOE: joblib.load(out / f"{WOE}.joblib"),
        "calibrators": joblib.load(out / "calibrators.joblib"),
        "spec": joblib.load(out / "matrix_spec.joblib"),
    }
    # Boosters loaded from file report best_iteration 0 -> use the saved one.
    bundle[POOLED].best_iteration = manifest["best_iteration"][POOLED]
    for b, booster in bundle[SEGMENTED].items():
        booster.best_iteration = manifest["best_iteration"][f"{SEGMENTED}_bucket{b}"]
    return bundle, manifest


def raw_scores(bundle: dict[str, Any], df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return every model's raw score on ``df`` (baseline is the bucket itself)."""
    spec = bundle["spec"]
    return {
        BASELINE: df["dq_bucket"].to_numpy(dtype=float),
        WOE: md.predict_woe_logistic(bundle[WOE], spec, df),
        POOLED: md.predict_lightgbm(bundle[POOLED], spec, df),
        SEGMENTED: md.predict_segmented(bundle[SEGMENTED], spec, df),
    }


def comparison_table(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    n_bins: int,
) -> pd.DataFrame:
    """Return AUC, PR-AUC and (for probabilistic models) Brier and ECE raw vs calibrated."""
    rows = []
    for name, s in scores.items():
        row: dict[str, Any] = {"model": name, **ev.discrimination(y, s)}
        if name in calibrated:
            row.update(
                brier_raw=ev.brier(y, s),
                brier_calibrated=ev.brier(y, calibrated[name]),
                ece_raw=ev.expected_calibration_error(y, s, n_bins),
                ece_calibrated=ev.expected_calibration_error(y, calibrated[name], n_bins),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def per_bucket_table(
    y: np.ndarray, scores: dict[str, np.ndarray], bucket: np.ndarray, buckets: list[int]
) -> pd.DataFrame:
    """Return per-bucket AUC for every model, long format."""
    rows = []
    for name, s in scores.items():
        for b, m in ev.per_bucket(y, s, bucket, buckets).items():
            rows.append({"model": name, "dq_bucket": b, **m})
    return pd.DataFrame(rows)


def guard_final_features(cfg: Config, bundle: dict[str, Any]) -> list[str]:
    """Re-run the leakage guard on the features the frozen boosters actually use.

    Returns a list of problems (empty when clean).
    """
    problems = []
    kept = cfg.features.kept()
    registry = feature_registry(cfg.features)
    try:
        leakage.check_registry({f: registry[f] for f in kept})
        leakage.check_columns(kept)
    except leakage.LeakageError as err:
        problems.append(str(err))
    boosters = [bundle[POOLED], *bundle[SEGMENTED].values()]
    for booster in boosters:
        if booster.feature_name() != kept:
            problems.append("a booster's feature list differs from the kept list")
        try:
            leakage.check_columns(booster.feature_name())
        except leakage.LeakageError as err:
            problems.append(str(err))
    return problems


def log_table(name: str, table: pd.DataFrame, cfg: Config) -> Path:
    """Write an aggregate table to outputs/tables and return its path."""
    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.tables_dir / f"{name}.csv"
    table.to_csv(path, index=False, float_format="%.5f")
    return path


def write_scores(
    cfg: Config, df: pd.DataFrame, raw: dict[str, np.ndarray], cal: dict[str, np.ndarray]
) -> None:
    """Write per-row raw and calibrated scores for one split (loan-level: stays in data/)."""
    out = df[SCORE_KEYS].copy()
    for name, s in raw.items():
        out[f"score_{name}"] = s
    for name, p in cal.items():
        out[f"p_{name}"] = p
    split = str(df["split"].iloc[0])
    path = cfg.paths.scores_dir / f"split={split}"
    path.mkdir(parents=True, exist_ok=True)
    out.drop(columns="split").to_parquet(path / "scores.parquet", index=False)


# --- fit phase ------------------------------------------------------------------


def fit_phase(cfg: Config) -> bool:
    """Fit on train, compare and calibrate on validation, freeze. Return True if checks pass."""
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    mc = cfg.models
    kept = cfg.features.kept()
    train, valid = read_split(cfg, "train"), read_split(cfg, "validation")
    spec = md.MatrixSpec.from_train(train, kept)
    ytr, yva = train[LABEL].to_numpy(), valid[LABEL].to_numpy()

    section("Inputs")
    print(
        f"train {len(train):,} rows (base rate {ytr.mean():.4f}); "
        f"validation {len(valid):,} rows (base rate {yva.mean():.4f}); {len(kept)} features"
    )
    print("no resampling or class weights: the train base rate is not extreme (see config)")

    section("Fit (train only; early stopping on validation)")
    seed = cfg.random_seed
    bundle: dict[str, Any] = {"spec": spec}
    bundle[WOE] = md.fit_woe_logistic(train, spec, mc.woe_logistic, seed)
    print("fitted woe_logistic")
    bundle[POOLED] = md.fit_lightgbm(train, valid, spec, mc.lightgbm, seed)
    print(f"fitted lgbm_pooled: best iteration {bundle[POOLED].best_iteration}")
    bundle[SEGMENTED] = md.fit_segmented(train, valid, spec, mc.lightgbm, seed, mc.segment_buckets)
    print(
        "fitted lgbm_segmented: best iterations "
        f"{ {b: m.best_iteration for b, m in bundle[SEGMENTED].items()} }"
    )

    section("Calibration (isotonic, fit on validation)")
    raw_va = raw_scores(bundle, valid)
    calibrators = {
        name: IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va[name], yva)
        for name in PROB_MODELS
    }
    bundle["calibrators"] = calibrators
    cal_va = {name: calibrators[name].predict(raw_va[name]) for name in PROB_MODELS}

    section("Validation comparison")
    n_bins = mc.calibration.reliability_bins
    comp = comparison_table(yva, raw_va, cal_va, n_bins)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(comp.to_string(index=False))
    print("note: *_calibrated columns are in-sample here (isotonic was fit on validation);")
    print("      the honest calibration check is on test, in the --evaluate-test phase")
    base_auc = float(comp.loc[comp["model"] == BASELINE, "auc"].iloc[0])
    for name in PROB_MODELS:
        auc = float(comp.loc[comp["model"] == name, "auc"].iloc[0])
        require(
            auc > base_auc,
            f"{name} beats {BASELINE} on validation AUC ({auc:.4f} > {base_auc:.4f})",
        )

    section("Segmented vs pooled, per bucket (validation)")
    pb = per_bucket_table(yva, raw_va, valid["dq_bucket"].to_numpy(), mc.segment_buckets)
    wide = pb.pivot_table(index="dq_bucket", columns="model", values="auc")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(wide.to_string())
    for b in mc.segment_buckets:
        row = wide.loc[b].to_dict()
        seg, pooled = float(row[SEGMENTED]), float(row[POOLED])
        winner = SEGMENTED if seg > pooled else POOLED
        print(f"bucket {b}: {winner} wins by {abs(seg - pooled):.4f}")

    auc_of = {r["model"]: r["auc"] for r in comp.to_dict("records")}
    gain = auc_of[SEGMENTED] - auc_of[POOLED]
    chosen = SEGMENTED if gain >= mc.segmented_min_auc_gain else POOLED
    print(
        f"\nsegmented - pooled validation AUC = {gain:+.4f}; required gain "
        f"{mc.segmented_min_auc_gain} -> primary model: {chosen}"
    )

    section(f"Top {mc.importance_top_n} features by gain ({POOLED})")
    imp = md.importance_table(bundle[POOLED])
    top = imp.head(mc.importance_top_n)
    print(top.to_string(index=False, float_format="{:.4f}".format))

    section("Leakage guard on the final feature set")
    problems = guard_final_features(cfg, bundle)
    require(not problems, f"final features pass the blocklist and window checks {problems or ''}")
    require(spec.features == kept, "model matrix uses exactly the kept feature list")

    section("Freeze")
    git = git_state()
    manifest = {
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config_hash": frozen_config_hash(cfg),
        **git,
        "chosen_model": chosen,
        "features": kept,
        "segment_buckets": mc.segment_buckets,
        "best_iteration": {
            POOLED: bundle[POOLED].best_iteration,
            **{f"{SEGMENTED}_bucket{b}": m.best_iteration for b, m in bundle[SEGMENTED].items()},
        },
        "validation_auc": auc_of,
    }
    save_frozen(cfg, bundle, manifest)
    print(
        f"frozen to {models_dir(cfg)} (config hash {manifest['config_hash']}, "
        f"git {git['git_sha'][:8]}{' dirty' if git['git_dirty'] == 'True' else ''})"
    )
    write_scores(cfg, valid, raw_va, cal_va)

    tables = {
        "model_comparison_validation": comp,
        "per_bucket_validation": pb,
        "feature_importance": imp,
        "calibration_deciles_validation": ev.decile_table(
            yva, cal_va[chosen], mc.calibration.decile_bins
        ),
    }
    paths = {name: log_table(name, t, cfg) for name, t in tables.items()}

    section("MLflow")
    setup_mlflow(cfg)
    with mlflow.start_run(run_name="stage4-fit") as parent:
        mlflow.set_tags(
            {**git, "phase": "fit", "config_hash": manifest["config_hash"], "chosen_model": chosen}
        )
        mlflow.log_artifact("config/config.yaml")
        mlflow.log_dict({"features": kept}, "feature_list.json")
        for p in paths.values():
            mlflow.log_artifact(str(p))
        for name in [BASELINE, *PROB_MODELS]:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({**git, "model": name, "phase": "fit"})
                if name in (POOLED, SEGMENTED):
                    mlflow.log_params({k: v for k, v in mc.lightgbm.params.items()})
                    mlflow.log_param("num_boost_round", mc.lightgbm.num_boost_round)
                if name == WOE:
                    mlflow.log_params(mc.woe_logistic.model_dump())
                row = comp[comp["model"] == name].iloc[0].dropna().to_dict()
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in row.items() if k != "model"})
                for r in pb[pb["model"] == name].to_dict("records"):
                    if "auc" in r and pd.notna(r["auc"]):
                        mlflow.log_metric(f"val_auc_bucket{r['dq_bucket']}", float(r["auc"]))
                if name == POOLED:
                    mlflow.log_figure(
                        plots.importance_plot(imp, mc.importance_top_n, "LightGBM pooled: gain"),
                        "feature_importance.png",
                    )
                if name in PROB_MODELS:
                    rel = {
                        "raw": ev.reliability_table(yva, raw_va[name], n_bins),
                        "isotonic": ev.reliability_table(yva, cal_va[name], n_bins),
                    }
                    mlflow.log_figure(
                        plots.reliability_plot(rel, f"{name}: validation"), "calibration.png"
                    )
        print(f"mlflow run {parent.info.run_id}")
    return not failures


# --- test phase -----------------------------------------------------------------


def test_phase(cfg: Config) -> bool:
    """Score frozen models on test once, append to the test log. True if checks pass."""
    failures: list[str] = []

    def require(ok: bool, msg: str) -> None:
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    bundle, manifest = load_frozen(cfg)
    current_hash = frozen_config_hash(cfg)
    if current_hash != manifest["config_hash"]:
        print(
            f"refusing: config changed since freeze ({manifest['config_hash']} -> {current_hash})"
        )
        return False

    mc = cfg.models
    n_bins = mc.calibration.reliability_bins
    chosen = manifest["chosen_model"]
    test = read_split(cfg, "test")
    yte = test[LABEL].to_numpy()

    section(f"Test evaluation of frozen models (frozen {manifest['frozen_at']})")
    raw_te = raw_scores(bundle, test)
    cal_te = {name: bundle["calibrators"][name].predict(raw_te[name]) for name in PROB_MODELS}
    comp = comparison_table(yte, raw_te, cal_te, n_bins)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(comp.to_string(index=False))

    base_auc = float(comp.loc[comp["model"] == BASELINE, "auc"].iloc[0])
    for name in PROB_MODELS:
        auc = float(comp.loc[comp["model"] == name, "auc"].iloc[0])
        require(auc > base_auc, f"{name} beats {BASELINE} on test AUC ({auc:.4f} > {base_auc:.4f})")
    for name, val_auc in manifest["validation_auc"].items():
        test_auc = float(comp.loc[comp["model"] == name, "auc"].iloc[0])
        gap = abs(val_auc - test_auc)
        require(
            gap <= mc.max_val_test_auc_gap,
            f"{name}: validation {val_auc:.4f} vs test {test_auc:.4f} (gap {gap:.4f})",
        )

    section(f"Calibration on test: {chosen}, raw vs isotonic")
    row = comp[comp["model"] == chosen].iloc[0]
    print(f"Brier raw {row['brier_raw']:.4f} -> isotonic {row['brier_calibrated']:.4f}")
    print(f"ECE   raw {row['ece_raw']:.4f} -> isotonic {row['ece_calibrated']:.4f}")
    require(row["brier_calibrated"] < row["brier_raw"], "isotonic improves Brier on test")
    require(row["ece_calibrated"] < row["ece_raw"], "isotonic improves reliability (ECE) on test")
    rel = {
        "raw": ev.reliability_table(yte, raw_te[chosen], n_bins),
        "isotonic": ev.reliability_table(yte, cal_te[chosen], n_bins),
    }
    for label, t in rel.items():
        print(f"\nreliability, {label}:")
        print(t.to_string(index=False, float_format="{:.4f}".format))
    deciles = {
        label: ev.decile_table(yte, p, mc.calibration.decile_bins)
        for label, p in (("raw", raw_te[chosen]), ("isotonic", cal_te[chosen]))
    }
    for label, t in deciles.items():
        print(f"\npredicted vs actual roll rate by decile, {label}:")
        print(t.to_string(index=False, float_format="{:.4f}".format))

    section("Per bucket (test)")
    pb = per_bucket_table(yte, raw_te, test["dq_bucket"].to_numpy(), mc.segment_buckets)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pb.pivot_table(index="dq_bucket", columns="model", values="auc").to_string())

    section("Record")
    git = git_state()
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    log_rows = comp.assign(
        evaluated_at=now,
        frozen_at=manifest["frozen_at"],
        config_hash=manifest["config_hash"],
        model_git_sha=manifest["git_sha"],
        eval_git_sha=git["git_sha"],
        eval_git_dirty=git["git_dirty"],
        chosen=comp["model"] == chosen,
    )
    log_path = cfg.paths.test_log
    previous = pd.read_csv(log_path) if log_path.exists() else None
    if previous is not None:
        print(f"NOTE: test has been evaluated {previous['evaluated_at'].nunique()} time(s) before")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows.to_csv(log_path, mode="a", header=previous is None, index=False, float_format="%.5f")
    print(f"appended to {log_path}")

    log_table("model_comparison_test", comp, cfg)
    log_table("per_bucket_test", pb, cfg)
    for label, t in deciles.items():
        log_table(f"calibration_deciles_test_{label}", t, cfg)
    for label, t in rel.items():
        log_table(f"reliability_test_{label}", t, cfg)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    fig = plots.reliability_plot(rel, f"{chosen}: test, raw vs isotonic")
    fig.savefig(cfg.paths.figures_dir / "calibration_test.png", dpi=150)

    for split in ("test", "drift_study", "late_holdout"):
        df = test if split == "test" else read_split(cfg, split)
        raw = raw_scores(bundle, df)
        cal = {name: bundle["calibrators"][name].predict(raw[name]) for name in PROB_MODELS}
        write_scores(cfg, df, raw, cal)
    print(f"wrote scores for test, drift_study, late_holdout to {cfg.paths.scores_dir}")

    setup_mlflow(cfg)
    with mlflow.start_run(run_name="stage4-test-evaluation"):
        mlflow.set_tags(
            {
                **git,
                "phase": "test",
                "config_hash": manifest["config_hash"],
                "chosen_model": chosen,
                "evaluated_at": now,
            }
        )
        for r in comp.to_dict("records"):
            mlflow.log_metrics(
                {
                    f"test_{r['model']}_{k}": float(v)
                    for k, v in r.items()
                    if k != "model" and pd.notna(v)
                }
            )
        mlflow.log_figure(fig, "calibration_test.png")
    return not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="score the frozen models on the test split (logged)",
    )
    args = parser.parse_args(argv)
    cfg = load_config()
    marker = cfg.paths.models_dir.parent / (
        ".s4_test.done" if args.evaluate_test else ".s4_fit.done"
    )
    marker.unlink(missing_ok=True)
    ok = test_phase(cfg) if args.evaluate_test else fit_phase(cfg)
    section("Result")
    if ok:
        marker.write_text("ok\n")
        print(f"ALL CHECKS PASSED - wrote {marker}")
        return 0
    print("CHECKS FAILED - no .done marker written")
    return 1


if __name__ == "__main__":
    sys.exit(main())
