"""Shared fixtures. Tests run on small hand-written frames, never on the real data."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from src.spark import get_spark
from src.utils.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def spark(cfg: Config) -> Iterator[SparkSession]:
    small = cfg.spark.model_copy(
        update={"master": "local[2]", "driver_memory": "1g", "shuffle_partitions": 2}
    )
    session = get_spark(small, "tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
