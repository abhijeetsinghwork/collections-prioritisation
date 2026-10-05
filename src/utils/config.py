"""Load and validate config/config.yaml. Unknown or mistyped keys fail at startup."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    raw_dir: Path
    layout_file: Path
    orig_glob: str
    perf_glob: str
    panel_dir: Path
    labels_dir: Path
    macro_dir: Path
    features_dir: Path
    feature_audit_table: Path
    schema_module: Path


class SparkConfig(_Strict):
    master: str
    driver_memory: str
    shuffle_partitions: int = Field(gt=0)
    session_timezone: str
    supported_java_majors: list[int]
    java_home_candidates: list[Path]


class DelinquencyConfig(_Strict):
    unknown_code: str
    reo_code: str
    bucket_cap: int = Field(gt=0)
    reo_bucket: int = Field(gt=0)


class IngestConfig(_Strict):
    vintages: list[int] = Field(min_length=1)
    delimiter: str
    sentinels: dict[str, list[str]]
    delinquency: DelinquencyConfig
    retain_origination: list[str]
    retain_performance: list[str]

    @model_validator(mode="after")
    def _join_key_retained(self) -> IngestConfig:
        for name, cols in (
            ("retain_origination", self.retain_origination),
            ("retain_performance", self.retain_performance),
        ):
            if "loan_sequence_number" not in cols:
                raise ValueError(f"{name} must include loan_sequence_number")
        return self


class ChecksConfig(_Strict):
    min_current_share: float = Field(ge=0, le=1)
    spot_check_loans: int = Field(ge=0)
    base_rate_bounds: tuple[float, float]
    min_rows_per_split: int = Field(gt=0)
    monotonic_splits: list[str]
    trace_examples_per_label: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> ChecksConfig:
        lo, hi = self.base_rate_bounds
        if not 0 <= lo < hi <= 1:
            raise ValueError(f"base_rate_bounds must satisfy 0 <= lo < hi <= 1, got {lo, hi}")
        return self


class LabelsConfig(_Strict):
    population_buckets: list[int] = Field(min_length=1)
    horizon_months: int = Field(gt=0)
    zero_balance_codes: dict[str, Literal["cure", "roll", "exclude"]]

    def codes(self, kind: str) -> list[str]:
        """Return the zero-balance codes classified as ``kind``."""
        return sorted(c for c, k in self.zero_balance_codes.items() if k == kind)


class SplitConfig(_Strict):
    name: str
    start: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    end: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @property
    def start_date(self) -> dt.date:
        """First day of the first month in the split."""
        return dt.date.fromisoformat(f"{self.start}-01")

    @property
    def end_date(self) -> dt.date:
        """First day of the last month in the split (periods are first-of-month)."""
        return dt.date.fromisoformat(f"{self.end}-01")


class MacroConfig(_Strict):
    url: str
    state_series: dict[str, str]
    national_series: dict[str, str]
    publication_lag_months: dict[str, int]
    change_windows: list[int]

    @model_validator(mode="after")
    def _lag_for_every_series(self) -> MacroConfig:
        names = set(self.state_series) | set(self.national_series)
        missing = names - set(self.publication_lag_months)
        if missing:
            raise ValueError(f"no publication lag configured for: {sorted(missing)}")
        if any(v < 0 for v in self.publication_lag_months.values()):
            raise ValueError("publication lags must be >= 0")
        return self


class AuditConfig(_Strict):
    tree_max_leaf_nodes: int = Field(gt=1)
    tree_min_samples_leaf: int = Field(gt=0)
    encoding_smoothing: float = Field(ge=0)


class FeatureDecision(_Strict):
    keep: bool
    reason: str = Field(min_length=3)


class FeaturesConfig(_Strict):
    trajectory_windows: list[int]
    event_windows: list[int]
    upb_change_windows: list[int]
    velocity_lag: int = Field(gt=0)
    bucket_lags: list[int]
    macro: MacroConfig
    audit: AuditConfig
    decisions: dict[str, FeatureDecision]

    @model_validator(mode="after")
    def _windows_positive(self) -> FeaturesConfig:
        for name in ("trajectory_windows", "event_windows", "upb_change_windows", "bucket_lags"):
            if any(n <= 0 for n in getattr(self, name)):
                raise ValueError(f"{name} must be positive month counts")
        return self

    def kept(self) -> list[str]:
        """Return the features with an explicit keep decision, in config order."""
        return [name for name, d in self.decisions.items() if d.keep]


class Config(_Strict):
    random_seed: int
    paths: PathsConfig
    spark: SparkConfig
    ingest: IngestConfig
    checks: ChecksConfig
    labels: LabelsConfig
    splits: list[SplitConfig] = Field(min_length=1)
    features: FeaturesConfig

    @model_validator(mode="after")
    def _splits_ordered_and_known(self) -> Config:
        names = [s.name for s in self.splits]
        if len(names) != len(set(names)):
            raise ValueError(f"split names must be unique: {names}")
        for s in self.splits:
            if s.start_date > s.end_date:
                raise ValueError(f"split {s.name} starts after it ends")
        for a, b in zip(self.splits, self.splits[1:], strict=False):
            if a.end_date >= b.start_date:
                raise ValueError(f"splits overlap or are out of order: {a.name}, {b.name}")
        unknown = set(self.checks.monotonic_splits) - set(names)
        if unknown:
            raise ValueError(f"checks.monotonic_splits names unknown splits: {sorted(unknown)}")
        return self


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Return the validated project config read from ``path``."""
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
