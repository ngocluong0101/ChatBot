from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbot.config import PROJECT_ROOT, Settings


def test_defaults_resolve_relative_paths_against_project_root():
    s = Settings()
    assert s.papers_dir == PROJECT_ROOT / "papers"
    assert s.index_dir == PROJECT_ROOT / ".index"


def test_absolute_paths_are_kept(tmp_path: Path):
    assert Settings(papers_dir=tmp_path).papers_dir == tmp_path


def test_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("RAG_TOP_K", "3")
    s = Settings()
    assert s.llm_model == "qwen2.5:7b-instruct"
    assert s.top_k == 3


def test_log_level_is_normalized():
    assert Settings(log_level="debug").log_level == "DEBUG"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 0},
        {"top_k": 0},
        {"score_threshold": 1.5},
        {"log_level": "LOUD"},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)
