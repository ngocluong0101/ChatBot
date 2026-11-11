from types import SimpleNamespace

import httpx
import pytest
from langchain_core.documents import Document

from ragbot import pipeline as pl
from ragbot.config import Settings
from ragbot.errors import LLMUnavailableError


def doc(source: str, page: int = 1, text: str = "nội dung") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class StubVectorStore:
    def __init__(self, results):
        self._results = results

    def similarity_search_with_score(self, _question, k):
        return self._results[:k]


class StubChain:
    def __init__(self, reply="2 ngày.", error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, inputs):
        self.calls.append(inputs)
        if self.error:
            raise self.error
        return self.reply


def make(results, chain=None, **settings_kwargs):
    chain = chain or StubChain()
    return pl.RagPipeline(Settings(**settings_kwargs), StubVectorStore(results), chain), chain


@pytest.mark.parametrize(
    "text",
    [
        "Tôi không tìm thấy thông tin này trong tài liệu.",
        "Không tìm thấy thông tin này trong tài liệu",
        "KHÔNG TÌM THẤY THÔNG TIN liên quan.",
    ],
)
def test_is_refusal_matches_variants(text):
    assert pl.is_refusal(text)


def test_is_refusal_false_for_real_answer():
    assert not pl.is_refusal("Tối đa 2 ngày mỗi tuần.")


def test_no_hit_above_threshold_skips_llm():
    p, chain = make([(doc("a.pdf"), 0.30)], score_threshold=0.5)
    answer = p.ask("Thủ đô của Pháp?")
    assert answer.refused and answer.text == pl.NOT_FOUND_MESSAGE
    assert answer.sources == ()
    assert chain.calls == [], "retrieval không đạt ngưỡng thì không được gọi LLM"


def test_answer_includes_only_sources_near_top_score():
    results = [(doc("remote.pdf"), 0.68), (doc("hr.pdf"), 0.57), (doc("leave.pdf"), 0.53)]
    p, chain = make(results, score_threshold=0.5, source_score_margin=0.08)

    answer = p.ask("remote mấy ngày?")

    assert [s.file for s in answer.sources] == ["remote.pdf"]
    assert answer.text == "2 ngày."
    # nhưng cả 3 chunk đạt ngưỡng đều được đưa vào context cho LLM
    assert chain.calls[0]["context"].count("[Tài liệu:") == 3


def test_sources_are_deduplicated_and_sorted():
    hits = [(doc("a.pdf", 1), 0.60), (doc("a.pdf", 1), 0.64), (doc("b.pdf", 2), 0.62)]
    sources = pl.select_sources(hits, margin=0.08)
    assert [(s.file, s.page) for s in sources] == [("a.pdf", 1), ("b.pdf", 2)]
    assert sources[0].score == 0.64


def test_llm_refusal_returns_normalized_message_without_sources():
    p, _ = make([(doc("a.pdf"), 0.7)], chain=StubChain("Không tìm thấy thông tin này."))
    answer = p.ask("câu hỏi")
    assert answer.refused and answer.text == pl.NOT_FOUND_MESSAGE and answer.sources == ()


@pytest.mark.parametrize("error", [ConnectionError("down"), httpx.ConnectError("down")])
def test_llm_connection_errors_become_llm_unavailable(error):
    p, _ = make([(doc("a.pdf"), 0.7)], chain=StubChain(error=error))
    with pytest.raises(LLMUnavailableError):
        p.ask("câu hỏi")


def test_render_formats_sources():
    answer = pl.Answer("2 ngày.", sources=(pl.Source("a.pdf", 1, 0.7), pl.Source("b.pdf", None, 0.7)))
    assert answer.render() == "2 ngày.\n\nNguồn:\n- a.pdf (trang 1)\n- b.pdf (trang ?)"
    assert pl.Answer(pl.NOT_FOUND_MESSAGE, refused=True).render() == pl.NOT_FOUND_MESSAGE


def test_check_ollama_reports_unreachable(monkeypatch):
    class Boom:
        def __init__(self, host):
            pass

        def list(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(pl.ollama, "Client", Boom)
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        pl.check_ollama(Settings())


def test_check_ollama_reports_missing_model(monkeypatch):
    class Fake:
        def __init__(self, host):
            pass

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(model="other:1b")])

    monkeypatch.setattr(pl.ollama, "Client", Fake)
    with pytest.raises(LLMUnavailableError, match="ollama pull llama3.2:3b"):
        pl.check_ollama(Settings(llm_model="llama3.2:3b"))


def test_check_ollama_accepts_untagged_name_as_latest(monkeypatch):
    class Fake:
        def __init__(self, host):
            pass

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(model="llama3.2:latest")])

    monkeypatch.setattr(pl.ollama, "Client", Fake)
    pl.check_ollama(Settings(llm_model="llama3.2"))
