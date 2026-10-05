"""Stage 4 model wrappers: design matrix, LightGBM (pooled and segmented), WOE logistic.

Every fit sees train rows only; validation is used for early stopping. Nothing
here reads the test split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.pipeline.labels import LABEL
from src.utils.config import LightGBMConfig, WoeLogisticConfig
from src.utils.leakage import OUTCOME_COLUMNS


@dataclass(frozen=True)
class MatrixSpec:
    """The exact feature list and categorical levels a model was trained with."""

    features: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_train(cls, train: pd.DataFrame, features: Sequence[str]) -> MatrixSpec:
        """Return a spec whose categorical levels are fixed from the train split only."""
        cats = {
            f: sorted(train[f].dropna().astype(str).unique().tolist())
            for f in features
            if train[f].dtype == object or isinstance(train[f].dtype, pd.StringDtype)
        }
        return cls(features=list(features), categories=cats)

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model matrix: features only, categoricals with train levels (unseen -> NaN).

        Raises if an outcome column would enter the matrix.
        """
        leaked = OUTCOME_COLUMNS & set(self.features)
        if leaked:
            raise ValueError(f"outcome columns in feature list: {sorted(leaked)}")
        X = df[self.features].copy()
        for f, levels in self.categories.items():
            values = X[f].astype("object")
            X[f] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
        return X


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: MatrixSpec,
    cfg: LightGBMConfig,
    seed: int,
) -> lgb.Booster:
    """Return a LightGBM booster fit on train, early-stopped on validation AUC."""
    params = {**cfg.params, "seed": seed}
    dtrain = lgb.Dataset(spec.matrix(train), label=train[LABEL], free_raw_data=False)
    dvalid = lgb.Dataset(spec.matrix(valid), label=valid[LABEL], reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dvalid],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )


def predict_lightgbm(booster: lgb.Booster, spec: MatrixSpec, df: pd.DataFrame) -> np.ndarray:
    """Return P(roll) from ``booster`` at its best iteration."""
    return np.asarray(booster.predict(spec.matrix(df), num_iteration=booster.best_iteration))


def fit_segmented(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: MatrixSpec,
    cfg: LightGBMConfig,
    seed: int,
    buckets: Sequence[int],
) -> dict[int, lgb.Booster]:
    """Return one booster per entry bucket, each fit and early-stopped on its own bucket's rows."""
    return {
        b: fit_lightgbm(
            train[train["dq_bucket"] == b], valid[valid["dq_bucket"] == b], spec, cfg, seed
        )
        for b in buckets
    }


def predict_segmented(
    boosters: dict[int, lgb.Booster], spec: MatrixSpec, df: pd.DataFrame
) -> np.ndarray:
    """Return P(roll), routing each row to the booster for its dq_bucket."""
    out = np.full(len(df), np.nan)
    buckets = df["dq_bucket"].to_numpy()
    for b, booster in boosters.items():
        mask = buckets == b
        if mask.any():
            out[mask] = predict_lightgbm(booster, spec, df[mask])
    if np.isnan(out).any():
        raise ValueError(
            f"rows with no segment model: buckets {sorted(set(buckets[np.isnan(out)]))}"
        )
    return out


def fit_woe_logistic(
    train: pd.DataFrame, spec: MatrixSpec, cfg: WoeLogisticConfig, seed: int
) -> Pipeline:
    """Return a WOE-binning + logistic regression pipeline fit on train."""
    binning = BinningProcess(
        variable_names=spec.features,
        categorical_variables=list(spec.categories),
        max_n_prebins=cfg.max_n_prebins,
        min_prebin_size=cfg.min_prebin_size,
    )
    model = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, random_state=seed)
    pipe = Pipeline([("woe", binning), ("logit", model)])
    X = train[spec.features].copy()
    for f in spec.categories:
        X[f] = X[f].astype("object")
    return pipe.fit(X, train[LABEL].to_numpy())


def predict_woe_logistic(pipe: Pipeline, spec: MatrixSpec, df: pd.DataFrame) -> np.ndarray:
    """Return P(roll) from the WOE logistic pipeline."""
    X = df[spec.features].copy()
    for f in spec.categories:
        X[f] = X[f].astype("object")
    return np.asarray(pipe.predict_proba(X)[:, 1])


def importance_table(booster: lgb.Booster) -> pd.DataFrame:
    """Return gain and split importance per feature, highest gain first."""
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "splits": booster.feature_importance("split"),
            }
        )
        .assign(gain_share=lambda d: d["gain"] / d["gain"].sum())
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
