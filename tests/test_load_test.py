import json

import httpx
import pytest

from evals.perf import load_test as lt


def test_percentile_interpolates_and_handles_empty():
    assert lt.percentile([], 50) is None
    assert lt.percentile([5.0], 99) == 5.0
    assert lt.percentile([1, 2, 3, 4], 50) == 2.5
    assert lt.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_summarize_times_reports_n_and_none_when_empty():
    empty = {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    assert lt.summarize_times([]) == empty
    stats = lt.summarize_times([1.0, 3.0])
    assert stats["n"] == 2 and stats["mean"] == 2.0 and stats["max"] == 3.0


def test_plan_is_deterministic_per_seed_and_user_count():
    questions = [{"question": f"q{i}", "category": "factoid"} for i in range(30)]
    a = lt.plan_requests(questions, 5, 4, seed=7)
    assert a == lt.plan_requests(questions, 5, 4, seed=7)
    assert a != lt.plan_requests(questions, 5, 4, seed=8)
    assert len(a) == 5 and all(len(p) == 4 for p in a)


def test_parse_sse_line():
    assert lt.parse_sse_line("event: token") == ("event", "token")
    assert lt.parse_sse_line('data: {"text": "a"}') == ("data", '{"text": "a"}')
    assert lt.parse_sse_line("") == (None, None)


def sse_body(*events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_one_request_measures_ttft_and_tokens():
    body = sse_body(("token", {"text": "a"}), ("token", {"text": "b"}), ("done", {"refused": False}))
    client = client_for(lambda r: httpx.Response(200, content=body.encode()))
    res = lt.one_request(client, "http://x", "hỏi", {})
    assert res["ok"] and res["tokens"] == 2 and res["refused"] is False
    assert 0 <= res["ttft"] <= res["seconds"]


def test_one_request_refusal_has_no_tokens_and_ttft_equals_total():
    body = sse_body(("done", {"refused": True}))
    client = client_for(lambda r: httpx.Response(200, content=body.encode()))
    res = lt.one_request(client, "http://x", "hỏi", {})
    assert res["ok"] and res["refused"] and res["tokens"] == 0 and res["ttft"] == res["seconds"]


def test_one_request_classifies_failures():
    busy = client_for(lambda r: httpx.Response(503, headers={"retry-after": "120"}))
    res = lt.one_request(busy, "http://x", "hỏi", {})
    assert not res["ok"] and res["error"] == "http_503" and res["retry_after"] == "120"

    def boom(request):
        raise httpx.ReadTimeout("quá lâu")

    assert lt.one_request(client_for(boom), "http://x", "hỏi", {})["error"] == "timeout"

    midstream = sse_body(("token", {"text": "a"}), ("error", {"message": "Ollama tắt"}))
    broken = client_for(lambda r: httpx.Response(200, content=midstream.encode()))
    res = lt.one_request(broken, "http://x", "hỏi", {})
    assert not res["ok"] and res["error"] == "sse_error"


def test_run_level_aggregates_results(monkeypatch):
    calls = []

    def fake(client, url, question, headers):
        calls.append(question)
        return {"ok": True, "ttft": 0.5, "tokens": 10, "refused": False, "error": None, "seconds": 2.0}

    monkeypatch.setattr(lt, "one_request", fake)
    monkeypatch.setattr(lt.GpuSampler, "read", classmethod(lambda cls: (4000.0, 8188.0, 50.0)))
    plan = [[{"question": "a", "category": "x"}] * 3 for _ in range(2)]
    result = lt.run_level("http://x", 2, plan, timeout=5, headers={})
    assert result["requests"] == 6 and result["ok"] == 6 and result["error_rate"] == 0.0
    assert result["latency_seconds"]["p50"] == 2.0 and result["ttft_seconds"]["p50"] == 0.5
    assert result["errors"] == {} and result["refused"] == 0
    assert len(calls) == 6
