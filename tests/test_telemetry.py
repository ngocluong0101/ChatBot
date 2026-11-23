"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ragbot import pipeline as pl
from ragbot.config import Settings
from ragbot.retrieval import Retriever
from ragbot.telemetry import NOOP, Telemetry, UsageRecorder


def traced() -> tuple[Telemetry, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Telemetry(Settings(metrics_enabled=True), tracer=provider.get_tracer("test")), exporter


def test_disabled_telemetry_is_a_noop():
    with NOOP.stage("ask", question="x") as stage:
        stage.set("k", "v")
    assert NOOP.metrics is False and NOOP.render_metrics is not None
    NOOP.observe_request("chat", "ok", 1.0)  # không ném lỗi, không làm gì
    NOOP.observe_tokens({"prompt_tokens": 3})


def test_stage_records_span_with_parent_and_attributes():
    telemetry, exporter = traced()
    with telemetry.stage("ask", **{"rag.question": "q"}) as root, telemetry.stage("retrieve", root) as child:
        child.set("retrieve.candidates", 5)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["retrieve"].parent.span_id == spans["ask"].context.span_id
    assert spans["retrieve"].attributes["retrieve.candidates"] == 5
    assert spans["ask"].attributes["rag.question"] == "q"
    assert spans["ask"].parent is None


def test_stage_marks_errors_but_not_generator_exit():
    telemetry, exporter = traced()
    with pytest.raises(ValueError), telemetry.stage("rewrite"):
        raise ValueError("hỏng")
    with pytest.raises(GeneratorExit), telemetry.stage("llm"):
        raise GeneratorExit
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["rewrite"].attributes["error"] is True
    assert "error" not in spans["llm"].attributes  # client ngắt kết nối không phải lỗi của bước này


def test_metrics_render_counts_stages_requests_and_tokens():
    telemetry = Telemetry(Settings(metrics_enabled=True))
    with telemetry.stage("rerank"):
        pass
    telemetry.observe_request("chat", "ok", 0.5)
    telemetry.observe_request("chat", "busy")
    telemetry.observe_tokens({"prompt_tokens": 100, "completion_tokens": 20})
    with telemetry.in_flight(), telemetry.waiting():
        text = telemetry.render_metrics().decode()
        assert "ragbot_in_flight_requests 1.0" in text and "ragbot_waiting_requests 1.0" in text
    text = telemetry.render_metrics().decode()
    assert 'ragbot_stage_seconds_count{stage="rerank"} 1.0' in text
    assert 'ragbot_requests_total{endpoint="chat",outcome="ok"} 1.0' in text
    assert 'ragbot_requests_total{endpoint="chat",outcome="busy"} 1.0' in text
    assert 'ragbot_llm_tokens_total{kind="prompt"} 100.0' in text
    assert "ragbot_in_flight_requests 0.0" in text


def test_metrics_off_creates_nothing():
    telemetry = Telemetry(Settings(metrics_enabled=False))
    assert telemetry.metrics is False
    telemetry.observe_request("chat", "ok", 1.0)


def test_each_telemetry_has_its_own_registry():
    # Nhiều app trong cùng một tiến trình (test, reload) không được đụng tên metric của nhau.
    one, two = Telemetry(Settings()), Telemetry(Settings())
    one.observe_request("chat", "ok")
    assert 'outcome="ok"' not in two.render_metrics().decode()


def llm_result(**info):
    message = SimpleNamespace(usage_metadata={"input_tokens": 812, "output_tokens": 64})
    generation = SimpleNamespace(message=message, generation_info=info)
    return SimpleNamespace(generations=[[generation]])


def test_usage_recorder_reads_tokens_and_speed_once():
    recorder = UsageRecorder()
    recorder.on_llm_end(llm_result(eval_count=64, eval_duration=2_000_000_000))
    assert recorder.pop() == {"prompt_tokens": 812, "completion_tokens": 64, "tokens_per_second": 32.0}
    assert recorder.pop() is None  # đọc xong là hết, lượt sau không thấy số của lượt trước


def test_usage_recorder_ignores_malformed_response():
    recorder = UsageRecorder()
    recorder.on_llm_end(SimpleNamespace(generations=[]))
    assert recorder.pop() is None


def test_usage_recorder_is_per_thread():
    import threading

    recorder = UsageRecorder()
    recorder.on_llm_end(llm_result())
    seen = []
    thread = threading.Thread(target=lambda: seen.append(recorder.pop()))
    thread.start()
    thread.join()
    assert seen == [None] and recorder.pop() is not None


# ---------------------------------------------------------------- pipeline + retriever


class StubStore:
    def similarity_search_with_score(self, question, k):
        return [(Document(page_content="nội dung", metadata={"source": "a.pdf", "page": 1}), 0.9)][:k]


class StubChain:
    def __init__(self, reply="2 ngày."):
        self.reply = reply

    def invoke(self, inputs):
        return self.reply

    def stream(self, inputs):
        yield from ("2 ", "ngày.")


def make_pipeline(telemetry, recorder=None):
    settings = Settings(rerank_model="", retrieval_mode="dense")
    pipeline = pl.RagPipeline(settings, StubStore(), StubChain(), rewrite_chain=StubChain("Câu hỏi có dấu?"))
    pipeline.telemetry = telemetry
    pipeline.usage = recorder
    return pipeline


def test_ask_emits_a_span_per_stage_with_usage_on_the_llm_span():
    telemetry, exporter = traced()
    recorder = UsageRecorder()
    pipeline = make_pipeline(telemetry, recorder)
    recorder.on_llm_end(llm_result(eval_count=10, eval_duration=1_000_000_000))
    answer = pipeline.ask("cau hoi khong dau")
    assert answer.text == "2 ngày."
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"ask", "rewrite", "retrieve", "retrieve_lock_wait", "dense", "llm"} <= set(spans)
    root = spans["ask"].context.span_id
    assert spans["retrieve"].parent.span_id == root and spans["llm"].parent.span_id == root
    assert spans["dense"].parent.span_id == spans["retrieve"].context.span_id
    assert spans["llm"].attributes["llm.prompt_tokens"] == 812
    assert spans["llm"].attributes["llm.tokens_per_second"] == 10.0
    assert spans["ask"].attributes["rag.refused"] is False


def test_refused_question_has_no_llm_span():
    telemetry, exporter = traced()
    pipeline = make_pipeline(telemetry)
    pipeline._settings = pipeline._settings.model_copy(update={"score_threshold": 0.99})
    assert pipeline.ask("hỏi gì đó").refused
    names = {s.name for s in exporter.get_finished_spans()}
    assert "llm" not in names and "retrieve" in names


def test_stream_records_time_to_first_token():
    telemetry, exporter = traced()
    pipeline = make_pipeline(telemetry)
    events = list(pipeline.ask_stream("Câu hỏi?"))
    assert events[-1].kind == "done"
    llm = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert 0 <= llm.attributes["llm.ttft_seconds"] < 5


def test_telemetry_assignment_reaches_the_retriever():
    telemetry, _ = traced()
    pipeline = make_pipeline(NOOP)
    pipeline.telemetry = telemetry
    assert pipeline._retriever.telemetry is telemetry


def test_retriever_serializes_model_calls_across_threads():
    """Embedding/reranker không an toàn khi gọi đồng thời: không bao giờ có hai `search` chạy cùng lúc."""
    import threading
    import time

    class SlowReranker:
        active = 0
        peak = 0

        def score(self, query, docs):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            time.sleep(0.03)
            type(self).active -= 1
            return [0.9] * len(docs)

    retriever = Retriever(Settings(retrieval_mode="dense", top_k=1), StubStore(), SlowReranker())
    threads = [threading.Thread(target=retriever.search, args=("câu hỏi",)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert SlowReranker.peak == 1
