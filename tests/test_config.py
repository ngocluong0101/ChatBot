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


def test_docs_dir_accepts_new_and_legacy_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert Settings().docs_dir == PROJECT_ROOT / "papers"
    assert Settings(docs_dir=tmp_path).papers_dir == tmp_path  # tên cũ còn đọc được
    assert Settings(papers_dir=tmp_path).docs_dir == tmp_path

    monkeypatch.setenv("RAG_PAPERS_DIR", str(tmp_path / "cu"))
    assert Settings().docs_dir == tmp_path / "cu"
    monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path / "moi"))
    assert Settings().docs_dir == tmp_path / "moi"  # tên mới thắng khi có cả hai


def test_legal_options_defaults_and_validation():
    s = Settings()
    assert s.pdf_loader == "unstructured"
    assert s.sources_file == PROJECT_ROOT / "data" / "SOURCES.md"
    assert Settings(chunking="legal", pdf_loader="pdfplumber").chunking == "legal"
    with pytest.raises(ValidationError):
        Settings(pdf_loader="ocr")


def test_concurrency_and_observability_defaults_keep_old_behaviour():
    s = Settings()
    assert s.max_concurrent_requests == 1 and s.tracing_enabled is False and s.metrics_enabled is True


@pytest.mark.parametrize("kwargs", [{"max_concurrent_requests": 0}, {"queue_timeout_seconds": 0}])
def test_invalid_concurrency_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
