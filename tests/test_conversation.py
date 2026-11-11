import httpx
import pytest
from langchain_core.documents import Document

from ragbot import pipeline as pl
from ragbot.config import Settings
from ragbot.errors import LLMUnavailableError


def doc(source: str, text: str = "nội dung") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 1})


class RecordingVectorStore:
    def __init__(self, results):
        self._results = results
        self.questions: list[str] = []

    def similarity_search_with_score(self, question, k):
        self.questions.append(question)
        return self._results[:k]


class StubChain:
    def __init__(self, reply="2 ngày.", error: Exception | None = None):
        self.reply, self.error, self.inputs = reply, error, []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        if self.error:
            raise self.error
        return self.reply

    def stream(self, inputs):
        self.inputs.append(inputs)
        for piece in self.reply.split(" "):
            yield piece + " "
        if self.error:
            raise self.error


def make(rewrite_reply="Câu hỏi độc lập?", rewrite_error=None, results=None, **settings_kwargs):
    settings_kwargs.setdefault("rerank_model", "")
    settings_kwargs.setdefault("retrieval_mode", "dense")
    vs = RecordingVectorStore(results if results is not None else [(doc("a.pdf"), 0.9)])
    chain, rewriter = StubChain(), StubChain(rewrite_reply, rewrite_error)
    pipeline = pl.RagPipeline(Settings(**settings_kwargs), vs, chain, rewrite_chain=rewriter)
    return pipeline, vs, chain, rewriter


HISTORY = [pl.Turn("user", "Remote mấy ngày?"), pl.Turn("assistant", "Tối đa 2 ngày.")]


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mat khau toi thieu bao nhieu ky tu", True),
        ("What is the remote work policy", True),  # tiếng Anh cũng ASCII: để LLM tự giữ nguyên
        ("Mật khẩu tối thiểu bao nhiêu ký tự?", False),
        ("remote", False),  # quá ngắn
        ("12 34 56", False),
    ],
)
def test_looks_unaccented(text, expected):
    assert pl.looks_unaccented(text) is expected


def test_format_history_labels_roles_and_truncates():
    assert pl.format_history([]) == "(trống)"
    text = pl.format_history([pl.Turn("user", "x" * 900), pl.Turn("assistant", "ok")])
    assert text.splitlines()[0] == "Người dùng: " + "x" * pl.MAX_TURN_CHARS
    assert text.splitlines()[1] == "Trợ lý: ok"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Nghỉ ốm cần báo trước mấy giờ?", "Nghỉ ốm cần báo trước mấy giờ?"),
        ('"Nghỉ ốm cần báo trước mấy giờ?"', "Nghỉ ốm cần báo trước mấy giờ?"),
        (
            "Câu hỏi độc lập: Nghỉ ốm cần báo trước mấy giờ?\nGiải thích thêm...",
            "Nghỉ ốm cần báo trước mấy giờ?",
        ),
        ("", "gốc"),
        ("   \n  ", "gốc"),
        ("x" * 500, "gốc"),  # lan man quá mức => không tin
    ],
)
def test_clean_rewrite(raw, expected):
    assert pl.clean_rewrite(raw, "gốc") == expected


# ---------------------------------------------------------------- rewriting in ask()


def test_followup_is_rewritten_and_used_for_retrieval_and_answer():
    pipeline, vs, chain, rewriter = make(rewrite_reply="Nghỉ ốm cần báo trước mấy giờ?")
    answer = pipeline.ask("còn nghỉ ốm thì sao?", HISTORY)

    assert vs.questions == ["Nghỉ ốm cần báo trước mấy giờ?"]
    assert chain.inputs[0]["question"] == "Nghỉ ốm cần báo trước mấy giờ?"
    assert answer.question == "Nghỉ ốm cần báo trước mấy giờ?"
    assert "Người dùng: Remote mấy ngày?" in rewriter.inputs[0]["history"]


def test_standalone_accented_question_without_history_skips_the_rewrite_call():
    pipeline, vs, _, rewriter = make()
    answer = pipeline.ask("Mật khẩu tối thiểu bao nhiêu ký tự?")
    assert rewriter.inputs == [] and vs.questions == ["Mật khẩu tối thiểu bao nhiêu ký tự?"]
    assert answer.question == "Mật khẩu tối thiểu bao nhiêu ký tự?"


def test_unaccented_question_is_rewritten_even_without_history():
    pipeline, vs, _, rewriter = make(rewrite_reply="Mật khẩu tối thiểu bao nhiêu ký tự?")
    pipeline.ask("mat khau toi thieu bao nhieu ky tu")
    assert len(rewriter.inputs) == 1 and vs.questions == ["Mật khẩu tối thiểu bao nhiêu ký tự?"]


def test_rewriting_can_be_disabled():
    pipeline, vs, _, rewriter = make(rewrite_queries=False)
    pipeline.ask("còn nghỉ ốm thì sao?", HISTORY)
    assert rewriter.inputs == [] and vs.questions == ["còn nghỉ ốm thì sao?"]


def test_history_is_limited_to_the_most_recent_turns():
    pipeline, _, _, rewriter = make(max_history_turns=1)
    old = [pl.Turn("user", "câu cũ"), pl.Turn("assistant", "đáp cũ")]
    pipeline.ask("còn nữa?", old + HISTORY)
    sent = rewriter.inputs[0]["history"]
    assert "câu cũ" not in sent and "Remote mấy ngày?" in sent


def test_rewrite_failure_becomes_llm_unavailable():
    pipeline, *_ = make(rewrite_error=httpx.ConnectError("down"))
    with pytest.raises(LLMUnavailableError):
        pipeline.ask("còn nghỉ ốm thì sao?", HISTORY)


def test_garbage_rewrite_falls_back_to_original_question():
    pipeline, vs, *_ = make(rewrite_reply="")
    pipeline.ask("còn nghỉ ốm thì sao?", HISTORY)
    assert vs.questions == ["còn nghỉ ốm thì sao?"]


# ---------------------------------------------------------------- ask_stream()


def test_stream_yields_rewrite_tokens_then_done():
    pipeline, *_ = make(rewrite_reply="Nghỉ ốm cần báo trước mấy giờ?")
    events = list(pipeline.ask_stream("còn nghỉ ốm thì sao?", HISTORY))

    assert [e.kind for e in events] == ["rewrite", "token", "token", "done"]
    assert events[0].text == "Nghỉ ốm cần báo trước mấy giờ?"
    assert events[-1].answer.text == "2 ngày." and events[-1].answer.sources[0].file == "a.pdf"


def test_stream_without_rewrite_has_no_rewrite_event():
    pipeline, *_ = make()
    kinds = [e.kind for e in pipeline.ask_stream("Mật khẩu tối thiểu bao nhiêu ký tự?")]
    assert kinds == ["token", "token", "done"]


def test_stream_refuses_without_calling_llm_when_nothing_passes_threshold():
    pipeline, _, chain, _ = make(results=[(doc("a.pdf"), 0.1)], score_threshold=0.5)
    events = list(pipeline.ask_stream("Thủ đô của Pháp là gì?"))
    assert [e.kind for e in events] == ["done"] and events[0].answer.refused
    assert chain.inputs == []


def test_stream_done_normalizes_a_refusal_that_was_streamed_as_tokens():
    pipeline, _, chain, _ = make()
    chain.reply = "Không có thông tin về vấn đề này."
    events = list(pipeline.ask_stream("Mật khẩu tối thiểu bao nhiêu ký tự?"))
    done = events[-1].answer
    assert done.refused and done.text == pl.NOT_FOUND_MESSAGE and done.sources == ()


def test_stream_llm_error_becomes_llm_unavailable():
    pipeline, _, chain, _ = make()
    chain.error = ConnectionError("down")
    with pytest.raises(LLMUnavailableError):
        list(pipeline.ask_stream("Mật khẩu tối thiểu bao nhiêu ký tự?"))
