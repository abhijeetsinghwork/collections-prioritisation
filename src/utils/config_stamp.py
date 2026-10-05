"""Write a per-stage content stamp, touching it only when the stage's inputs change.

Make uses the stamp's mtime as the stage's only real dependency, so a stage
re-runs when, and only when, something it depends on changes *in content*:

- its own config sections (the validated values, so comments and key order do not count);
- the bytes of its source files (a git checkout or merge rewrites mtimes, not content);
- its external inputs (raw and macro files: name, size and mtime; never in git);
- the stamp of the stage upstream, so a change propagates down the chain.

``src/utils/config.py`` is not hashed: the stamp already holds the validated
config values, so adding a config class for one stage does not re-run the rest.

Run: python -m src.utils.config_stamp <stage> <path>
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import Config, load_config

SPARK = ("src/spark.py", "src/utils/schema.py")


@dataclass(frozen=True)
class Stage:
    sections: tuple[str, ...]
    sources: tuple[str, ...]
    upstream: str | None = None
    inputs: tuple[str, ...] = field(default=())  # globs, relative to the repo root


STAGES: dict[str, Stage] = {
    "ingest": Stage(
        ("ingest",),
        ("src/s1_ingest.py", "src/pipeline/ingest.py", "src/pipeline/checks.py", *SPARK),
        inputs=("data/raw/sample_*.txt",),
    ),
    "labels": Stage(
        ("labels", "splits"),
        ("src/s2_labels.py", "src/pipeline/labels.py", "src/pipeline/label_checks.py", *SPARK),
        upstream="ingest",
    ),
    "features": Stage(
        ("features",),
        (
            "src/s3_features.py",
            "src/pipeline/features.py",
            "src/pipeline/feature_audit.py",
            "src/pipeline/macro.py",
            "src/pipeline/labels.py",
            "src/utils/leakage.py",
            *SPARK,
        ),
        upstream="labels",
        inputs=("data/raw/macro/*.csv",),
    ),
    "models": Stage(
        ("models", "random_seed"),
        (
            "src/s4_models.py",
            "src/pipeline/modeling.py",
            "src/pipeline/evaluation.py",
            "src/pipeline/calibration.py",
            "src/pipeline/features.py",
            "src/utils/leakage.py",
            "src/utils/plots.py",
        ),
        upstream="features",
    ),
    "policy": Stage(
        ("policy", "random_seed"),
        ("src/s5_policy.py", "src/pipeline/policy.py", "src/utils/plots.py"),
        upstream="models",
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp_text(
    stage: str,
    cfg: Config | None = None,
    root: Path = Path("."),
    stages: dict[str, Stage] = STAGES,
) -> str:
    """Return the canonical JSON describing everything ``stage``'s output depends on."""
    cfg = cfg or load_config()
    spec = stages[stage]
    dumped = cfg.model_dump(mode="json")
    inputs = {}
    for pattern in spec.inputs:
        for path in sorted(root.glob(pattern)):
            st = path.stat()
            inputs[str(path.relative_to(root))] = [st.st_size, int(st.st_mtime)]
    payload = {
        "config": {k: dumped[k] for k in spec.sections},
        "sources": {s: _sha((root / s).read_bytes()) for s in spec.sources},
        "inputs": inputs,
        "upstream": (
            _sha(stamp_text(spec.upstream, cfg, root, stages).encode()) if spec.upstream else None
        ),
    }
    return json.dumps(payload, indent=1, sort_keys=True)


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` only if it differs. Return True if written."""
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def main(argv: list[str]) -> int:
    stage, path = argv[0], Path(argv[1])
    write_if_changed(path, stamp_text(stage))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
