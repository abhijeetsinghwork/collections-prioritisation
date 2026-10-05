"""The one SparkSession factory used by every stage and by the test suite."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from pyspark.sql import SparkSession

from src.utils.config import SparkConfig


def _java_major(java_home: Path) -> int | None:
    """Return the major version of the JDK at ``java_home``, or None if absent."""
    java = java_home / "bin" / "java"
    if not java.exists():
        return None
    out = subprocess.run([str(java), "-version"], capture_output=True, text=True).stderr
    match = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not match:
        return None
    major = int(match.group(1))
    return int(match.group(2)) if major == 1 and match.group(2) else major  # "1.8" -> 8


def ensure_java(cfg: SparkConfig) -> Path:
    """Point JAVA_HOME at a Spark-supported JDK, or raise with a fix.

    Respects an existing JAVA_HOME; otherwise tries the configured candidates.
    """
    candidates = [Path(os.environ["JAVA_HOME"])] if os.environ.get("JAVA_HOME") else []
    candidates += cfg.java_home_candidates
    for home in candidates:
        if _java_major(home) in cfg.supported_java_majors:
            os.environ["JAVA_HOME"] = str(home)
            return home
    raise RuntimeError(
        f"No supported JDK found (PySpark 3.5 needs Java {cfg.supported_java_majors}). "
        f"Tried: {[str(c) for c in candidates]}. Install with `brew install openjdk@17` "
        "or set JAVA_HOME."
    )


def get_spark(cfg: SparkConfig, app_name: str = "collections") -> SparkSession:
    """Return a local-mode SparkSession configured from ``cfg``."""
    ensure_java(cfg)
    return (
        SparkSession.builder.appName(app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", cfg.session_timezone)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
