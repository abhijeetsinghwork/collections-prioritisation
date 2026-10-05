"""Leakage guard, part 1: static checks on feature names, sources and windows.

Part 2 (single-feature AUC review) lives in src/pipeline/feature_audit.py.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Post-outcome fields (spec section 7). Any feature whose name or source
# column contains one of these substrings is banned.
BLOCKED = [
    "zero_balance",
    "net_sales_proceeds",
    "mi_recoveries",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance",
    "taxes_and_insurance",
    "actual_loss",
    "delinquent_accrued_interest",
    "defect_settlement",
    "ddlpi",
    "zero_balance_removal_upb",
]

# The label and its derivatives must never be a feature.
OUTCOME_COLUMNS = {"rolls_deeper", "label_reason"}


class LeakageError(ValueError):
    """Raised when a feature could carry information from month T+1 or later."""


@dataclass(frozen=True)
class FeatureMeta:
    """Where a feature comes from and the latest month it may read.

    ``window_end`` is relative to T: None for static origination attributes,
    0 when the feature reads month T itself, -k when its newest input is T-k.
    """

    family: str
    sources: tuple[str, ...]
    window_end: int | None


def blocked_terms(name: str) -> list[str]:
    """Return the blocklist substrings found in ``name``."""
    return [b for b in BLOCKED if b in name.lower()]


def check_columns(columns: Iterable[str]) -> None:
    """Raise if any column name is blocked or is an outcome column."""
    bad = {c: blocked_terms(c) for c in columns if blocked_terms(c)}
    outcome = sorted(set(columns) & OUTCOME_COLUMNS)
    if bad or outcome:
        raise LeakageError(f"blocked feature columns: {bad or ''} outcome columns: {outcome or ''}")


def check_registry(registry: Mapping[str, FeatureMeta]) -> None:
    """Raise unless every feature has unblocked sources and a window ending at or before T."""
    check_columns(registry)
    problems = []
    for name, meta in registry.items():
        for src in meta.sources:
            if blocked_terms(src) or src in OUTCOME_COLUMNS:
                problems.append(f"{name}: source {src} is blocked")
        if meta.window_end is not None and meta.window_end > 0:
            problems.append(f"{name}: window ends at T+{meta.window_end}")
    if problems:
        raise LeakageError("; ".join(problems))


def check_features_registered(columns: Iterable[str], registry: Mapping[str, FeatureMeta]) -> None:
    """Raise if a candidate feature column has no registry entry (and so no audited window)."""
    unknown = sorted(set(columns) - set(registry))
    if unknown:
        raise LeakageError(f"feature columns missing from the registry: {unknown}")
