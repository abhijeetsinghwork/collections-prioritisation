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
    models_dir: Path
    scores_dir: Path
    tables_dir: Path
    figures_dir: Path
    test_log: Path
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


class LightGBMConfig(_Strict):
    params: dict[str, str | int | float | bool]
    num_boost_round: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)


class WoeLogisticConfig(_Strict):
    max_n_prebins: int = Field(gt=1)
    min_prebin_size: float = Field(gt=0, lt=1)
    C: float = Field(gt=0)
    max_iter: int = Field(gt=0)


class CalibrationConfig(_Strict):
    reliability_bins: int = Field(gt=1)
    decile_bins: int = Field(gt=1)
    transfer_holdout_months: int = Field(gt=0)
    methods: list[Literal["isotonic", "platt", "intercept_shift"]] = Field(min_length=1)


class TrackingConfig(_Strict):
    mlflow_tracking_uri: str
    mlflow_artifact_dir: Path
    mlflow_experiment: str


class ModelsConfig(_Strict):
    lightgbm: LightGBMConfig
    woe_logistic: WoeLogisticConfig
    segment_buckets: list[int]
    segmented_min_auc_gain: float = Field(ge=0)
    calibration: CalibrationConfig
    importance_top_n: int = Field(gt=0)
    max_val_test_auc_gap: float = Field(gt=0, lt=1)


class CapacityGrid(_Strict):
    start: float = Field(gt=0, le=1)
    stop: float = Field(gt=0, le=1)
    step: float = Field(gt=0, lt=1)


class PolicyConfig(_Strict):
    split: str
    capacity_grid: CapacityGrid
    table_capacities: list[float] = Field(min_length=1)
    headline_capacity: float = Field(gt=0, lt=1)
    headline_policy: Literal[
        "by_dpd", "by_balance", "by_prob", "by_expected_value", "by_expected_value_raw"
    ]
    random_seeds: int = Field(gt=0)
    random_capture_tolerance: float = Field(gt=0, lt=1)
    balance_share_tolerance: float = Field(gt=0, lt=1)

    def capacities(self) -> list[float]:
        """Return the plotted capacity grid plus the table capacities and 1.0, sorted."""
        g = self.capacity_grid
        n = round((g.stop - g.start) / g.step)
        grid = {round(g.start + i * g.step, 6) for i in range(n + 1)}
        grid |= {round(c, 6) for c in self.table_capacities}
        grid |= {round(self.headline_capacity, 6), 1.0}
        return sorted(grid)

    @model_validator(mode="after")
    def _grid_ordered(self) -> PolicyConfig:
        if self.capacity_grid.start >= self.capacity_grid.stop:
            raise ValueError("policy.capacity_grid.start must be below stop")
        if any(not 0 < c < 1 for c in self.table_capacities):
            raise ValueError("policy.table_capacities must lie strictly between 0 and 1")
        return self


class DriftConfig(_Strict):
    reference_split: str
    baseline_split: str
    monitored_splits: list[str] = Field(min_length=1)
    psi_bins: int = Field(gt=1)
    psi_floor: float = Field(gt=0, lt=0.01)
    psi_bands: tuple[float, float]
    band_sd: float = Field(gt=0)
    consecutive_months: int = Field(gt=0)
    split_half_max_psi: float = Field(gt=0)
    auc_match_tolerance: float = Field(gt=0)
    heatmap_top_n: int = Field(gt=0)

    @model_validator(mode="after")
    def _baseline_monitored(self) -> DriftConfig:
        if self.baseline_split not in self.monitored_splits:
            raise ValueError("drift.baseline_split must be one of drift.monitored_splits")
        if self.reference_split in self.monitored_splits:
            raise ValueError("drift.reference_split cannot also be monitored")
        return self


class AcceptanceConfig(_Strict):
    # check id -> reason. Only checks listed here can be accepted.
    accepted_failures: dict[Literal["calibration_improves_on_test"], str]

    @model_validator(mode="after")
    def _reasons_given(self) -> AcceptanceConfig:
        if any(len(r.strip()) < 20 for r in self.accepted_failures.values()):
            raise ValueError("every accepted failure needs a real reason")
        return self


class Config(_Strict):
    random_seed: int
    paths: PathsConfig
    spark: SparkConfig
    ingest: IngestConfig
    checks: ChecksConfig
    labels: LabelsConfig
    splits: list[SplitConfig] = Field(min_length=1)
    features: FeaturesConfig
    tracking: TrackingConfig
    models: ModelsConfig
    policy: PolicyConfig
    drift: DriftConfig
    acceptance: AcceptanceConfig

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
        if self.policy.split not in names:
            raise ValueError(f"policy.split names an unknown split: {self.policy.split}")
        drift_splits = {self.drift.reference_split, *self.drift.monitored_splits}
        if drift_splits - set(names):
            raise ValueError(f"drift names unknown splits: {sorted(drift_splits - set(names))}")
        return self


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Return the validated project config read from ``path``."""
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
