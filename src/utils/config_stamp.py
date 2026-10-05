"""Write a per-stage config stamp, touching it only when that stage's config changes.

Make uses the stamp's mtime as a dependency, so editing one stage's settings
(e.g. feature decisions) does not force earlier stages to rebuild.

Run: python -m src.utils.config_stamp <stage> <path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.utils.config import load_config

# Config sections that change each stage's output.
STAGE_SECTIONS = {
    "ingest": ["ingest"],
    "labels": ["labels", "splits"],
    "features": ["features"],
    "models": ["models", "random_seed"],
    "policy": ["policy", "random_seed"],
}


def stamp_text(stage: str) -> str:
    """Return the canonical JSON of the config sections ``stage`` depends on."""
    dumped = load_config().model_dump(mode="json")
    return json.dumps({k: dumped[k] for k in STAGE_SECTIONS[stage]}, indent=1, sort_keys=True)


def main(argv: list[str]) -> int:
    stage, path = argv[0], Path(argv[1])
    text = stamp_text(stage)
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
