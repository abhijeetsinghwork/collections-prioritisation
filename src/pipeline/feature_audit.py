"""Leakage guard, part 2: null rates and single-feature AUC, for review by a person.

Nothing here drops a feature. It produces the table that the explicit
keep/drop decisions in config.yaml are made from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import AuditConfig
from src.utils.leakage import FeatureMeta

MISSING_CATEGORY = "__missing__"


def is_categorical(series: pd.Series) -> bool:
    """Return True for string/object/boolean-coded categorical columns."""
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def target_encode(
    train_x: pd.Series, train_y: pd.Series, apply_x: pd.Series, smoothing: float
) -> np.ndarray:
    """Return smoothed train-fitted target means for ``apply_x`` (unseen -> global mean)."""
    prior = float(train_y.mean())
    tx = train_x.astype("object").fillna(MISSING_CATEGORY)
    stats = train_y.groupby(tx).agg(["sum", "count"])
    enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    ax = apply_x.astype("object").fillna(MISSING_CATEGORY)
    return ax.map(enc).fillna(prior).to_numpy(dtype=float)


def single_feature_auc(
    train: pd.DataFrame, valid: pd.DataFrame, feature: str, label: str, cfg: AuditConfig
) -> tuple[float, float]:
    """Return (train AUC, validation AUC) of a model that sees only ``feature``.

    Categoricals: target-mean encoding fit on train. Numerics: a small decision
    tree fit on train (handles non-monotone shapes and missing values).
    """
    ytr, yva = train[label], valid[label]
    if is_categorical(train[feature]):
        ptr = target_encode(train[feature], ytr, train[feature], cfg.encoding_smoothing)
        pva = target_encode(train[feature], ytr, valid[feature], cfg.encoding_smoothing)
    else:
        xtr = train[[feature]].astype(float).to_numpy()
        xva = valid[[feature]].astype(float).to_numpy()
        if np.all(np.isnan(xtr)) or np.nanstd(xtr) == 0:
            return 0.5, 0.5
        tree = DecisionTreeClassifier(
            max_leaf_nodes=cfg.tree_max_leaf_nodes,
            min_samples_leaf=cfg.tree_min_samples_leaf,
            random_state=0,
        ).fit(xtr, ytr)
        ptr, pva = tree.predict_proba(xtr)[:, 1], tree.predict_proba(xva)[:, 1]
    return float(roc_auc_score(ytr, ptr)), float(roc_auc_score(yva, pva))


def audit_table(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    registry: Mapping[str, FeatureMeta],
    label: str,
    cfg: AuditConfig,
) -> pd.DataFrame:
    """Return one row per feature: family, window end, null rates, train/validation AUC.

    Sorted by validation AUC, highest first: the top of this table is what gets
    reviewed by hand for leakage.
    """
    rows = []
    for f in features:
        auc_tr, auc_va = single_feature_auc(train, valid, f, label, cfg)
        meta = registry[f]
        rows.append(
            {
                "feature": f,
                "family": meta.family,
                "window_end": "static" if meta.window_end is None else f"T{meta.window_end:+d}",
                "kind": "categorical" if is_categorical(train[f]) else "numeric",
                "null_rate_train": float(train[f].isna().mean()),
                "null_rate_valid": float(valid[f].isna().mean()),
                "auc_train": auc_tr,
                "auc_valid": auc_va,
            }
        )
    table = pd.DataFrame(rows).sort_values("auc_valid", ascending=False).reset_index(drop=True)
    table["auc_rank"] = np.arange(1, len(table) + 1)
    return table
