"""Load and validate config/config.yaml. Unknown or mistyped keys fail at startup."""

from __future__ import annotations

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


class LabelsConfig(_Strict):
    zero_balance_codes: dict[str, Literal["cure", "roll", "exclude"]]


class Config(_Strict):
    random_seed: int
    paths: PathsConfig
    spark: SparkConfig
    ingest: IngestConfig
    checks: ChecksConfig
    labels: LabelsConfig


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Return the validated project config read from ``path``."""
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
