"""Pipeline stamps change with content, not with file times, and propagate downstream."""

from __future__ import annotations

import os

import pytest

from src.utils import config_stamp as cs

TOY = {
    "up": cs.Stage(("ingest",), ("up.py",), inputs=("raw/*.txt",)),
    "down": cs.Stage(("policy",), ("down.py",), upstream=("up",)),
}


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.txt").write_text("rows")
    (tmp_path / "up.py").write_text("x = 1\n")
    (tmp_path / "down.py").write_text("y = 2\n")
    return tmp_path


def _stamps(cfg, root):
    return {s: cs.stamp_text(s, cfg, root, TOY) for s in TOY}


def test_touching_a_source_changes_nothing(cfg, repo):
    before = _stamps(cfg, repo)
    os.utime(repo / "up.py", (0, 2_000_000_000))  # what a checkout does to mtimes
    assert _stamps(cfg, repo) == before


def test_editing_upstream_source_propagates_downstream(cfg, repo):
    before = _stamps(cfg, repo)
    (repo / "up.py").write_text("x = 2\n")
    after = _stamps(cfg, repo)
    assert after["up"] != before["up"]
    assert after["down"] != before["down"]


def test_editing_downstream_source_leaves_upstream_alone(cfg, repo):
    before = _stamps(cfg, repo)
    (repo / "down.py").write_text("y = 3\n")
    after = _stamps(cfg, repo)
    assert after["up"] == before["up"]
    assert after["down"] != before["down"]


def test_new_raw_file_changes_the_stamp(cfg, repo):
    before = _stamps(cfg, repo)
    (repo / "raw" / "b.txt").write_text("more rows")
    assert _stamps(cfg, repo)["up"] != before["up"]


def test_other_stages_config_does_not_count(cfg, repo):
    before = _stamps(cfg, repo)
    changed = cfg.model_copy(
        update={"models": cfg.models.model_copy(update={"importance_top_n": 7})}
    )
    assert _stamps(changed, repo) == before
    changed = cfg.model_copy(update={"policy": cfg.policy.model_copy(update={"random_seeds": 3})})
    after = _stamps(changed, repo)
    assert after["up"] == before["up"] and after["down"] != before["down"]


def test_write_if_changed_keeps_mtime_when_content_is_equal(tmp_path):
    path = tmp_path / "s.json"
    assert cs.write_if_changed(path, "a")
    os.utime(path, (0, 1_000))
    assert not cs.write_if_changed(path, "a")
    assert path.stat().st_mtime == 1_000
    assert cs.write_if_changed(path, "b")


def test_every_real_stage_source_exists(cfg):
    for name, stage in cs.STAGES.items():
        for src in stage.sources:
            assert os.path.exists(src), f"{name}: {src}"
        assert set(stage.upstream) <= set(cs.STAGES)
