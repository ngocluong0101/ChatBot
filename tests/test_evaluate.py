import json
import unicodedata
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragbot import evaluate as ev
from ragbot.errors import LLMUnavailableError
from ragbot.pipeline import Answer, Source

ROOT = Path(__file__).resolve().parent.parent


def example(**overrides) -> ev.Example:
    base = {
        "id": "x1",
        "question": "q?",
        "category": "factoid",
        "answerable": True,
        "gold_sources": ("a.pdf",),
        "evidence": ("tối đa 2 ngày",),
        "must_include": (("2 ngày",),),
    }
    return ev.Example(**{**base, **overrides})


def unanswerable(**overrides) -> ev.Example:
    return example(answerable=False, gold_sources=(), evidence=(), must_include=(), **overrides)


# ---------------------------------------------------------------- matching


def test_normalize_handles_nfd_case_and_whitespace():
    nfd = unicodedata.normalize("NFD", "Phê  duyệt\n\ntrước")
    assert ev.normalize(nfd) == "phê duyệt trước"


def test_contains_term_respects_word_boundaries():
    assert ev.contains_term(ev.normalize("tối đa 7 ngày"), "7 ngày")
    assert not ev.contains_term(ev.normalize("tối đa 17 ngày"), "7 ngày")
    assert not ev.contains_term(ev.normalize("thiết bị"), "IT")
    assert ev.contains_term(ev.normalize("báo cho IT ngay"), "IT")


def test_answer_is_correct_requires_every_group():
    groups = (("09:30", "9h30"), ("14:00",))
    assert ev.answer_is_correct("Từ 9h30 đến 11:30 và 14:00 đến 16:30", groups)
    assert not ev.answer_is_correct("Từ 9h30 đến 11:30", groups)


def test_is_relevant_chunk_needs_gold_source_and_evidence_across_line_breaks():
    ex = example()
    doc = Document(page_content="được đăng ký tối đa\n\n2 ngày làm việc", metadata={"source": "a.pdf"})
    assert ev.is_relevant_chunk(doc, ex)
    other_source = Document(page_content=doc.page_content, metadata={"source": "b.pdf"})
    assert not ev.is_relevant_chunk(other_source, ex)
    other_text = Document(page_content="nội dung khác", metadata={"source": "a.pdf"})
    assert not ev.is_relevant_chunk(other_text, ex)


def test_find_unsupported_examples():
    chunks = [Document(page_content="tối đa 2 ngày", metadata={"source": "a.pdf"})]
    good, bad = example(id="ok"), example(id="bad", evidence=("không tồn tại",))
    assert ev.find_unsupported_examples([good, bad, unanswerable(id="u")], chunks) == ["bad"]


# ---------------------------------------------------------------- dataset


def test_real_dataset_is_well_formed():
    examples = ev.load_dataset()
    assert len({e.id for e in examples}) == len(examples)
    assert any(e.answerable for e in examples) and any(not e.answerable for e in examples)
    assert all(e.question.strip() for e in examples)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path):
    line = json.dumps({"id": "a", "question": "q", "category": "c", "answerable": False})
    path = tmp_path / "d.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        ev.load_dataset(path)


def test_load_dataset_rejects_answerable_without_evidence(tmp_path: Path):
    row = {"id": "a", "question": "q", "category": "c", "answerable": True}
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold_sources"):
        ev.load_dataset(path)


# ---------------------------------------------------------------- stats


def test_percentile_and_ratio():
    assert ev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert ev.percentile([], 95) is None
    assert ev.ratio(1, 0) is None and ev.ratio(1, 4) == 0.25


def test_summarize_retrieval_hit_and_mrr():
    records = [
        {
            "answerable": True,
            "category": "a",
            "first_rank": 1,
            "source_recall": 1.0,
            "context_precision": 1.0,
        },
        {
            "answerable": True,
            "category": "a",
            "first_rank": 2,
            "source_recall": 1.0,
            "context_precision": 0.5,
        },
        {
            "answerable": True,
            "category": "b",
            "first_rank": None,
            "source_recall": 0.5,
            "context_precision": None,
        },
        {"answerable": False, "category": "o", "top1_score": 0.2},
    ]
    s = ev.summarize_retrieval(records, top_k=5)
    assert s["n_answerable"] == 3
    assert s["hit@1"] == pytest.approx(1 / 3)
    assert s["hit@5"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert s["context_precision"] == pytest.approx(0.75)
    assert s["by_category"]["b"]["hit@5"] == 0.0


def test_sweep_thresholds_finds_separating_threshold():
    records = [
        {"answerable": True, "top1_score": 0.7},
        {"answerable": True, "top1_score": 0.6},
        {"answerable": False, "top1_score": 0.3},
        {"answerable": False, "top1_score": 0.4},
    ]
    sweep = ev.sweep_thresholds(records, start=0.3, stop=0.8, step=0.1)
    assert sweep["min_answerable_top1"] == 0.6 and sweep["max_unanswerable_top1"] == 0.4
    assert sweep["best"]["balanced_accuracy"] == 1.0
    assert 0.4 < sweep["best"]["threshold"] <= 0.6


def test_sweep_thresholds_handles_missing_class():
    sweep = ev.sweep_thresholds([{"answerable": True, "top1_score": 0.7}])
    assert sweep["best"] is None


# ---------------------------------------------------------------- generation


class StubPipeline:
    def __init__(self, answers: dict[str, Answer | Exception]):
        self._answers = answers

    def ask(self, question: str) -> Answer:
        result = self._answers[question]
        if isinstance(result, Exception):
            raise result
        return result


def test_evaluate_generation_scores_each_outcome():
    examples = [
        example(id="ok", question="q-ok"),
        example(id="wrong-cite", question="q-cite"),
        example(id="false-refusal", question="q-refuse"),
        unanswerable(id="u-refused", question="q-u1"),
        unanswerable(id="u-halluc", question="q-u2"),
        example(id="boom", question="q-err"),
    ]
    good_source = (Source("a.pdf", 1, 0.7),)
    refusal = "Tôi không tìm thấy thông tin này trong tài liệu."
    pipeline = StubPipeline(
        {
            "q-ok": Answer("Tối đa 2 ngày.", sources=good_source, contexts=("ctx",)),
            "q-cite": Answer("Tối đa 2 ngày.", sources=(Source("z.pdf", 1, 0.7),)),
            "q-refuse": Answer(refusal, refused=True),
            "q-u1": Answer(refusal, refused=True),
            "q-u2": Answer("Bịa ra một câu trả lời", sources=good_source),
            "q-err": LLMUnavailableError("down"),
        }
    )
    judged = []

    def judge(question: str, context: str, answer: str) -> bool:
        judged.append((question, context, answer))
        return True

    records, summary = ev.evaluate_generation(pipeline, examples, judge)

    assert summary["errors"] == 1 and summary["n"] == 6
    assert summary["answerable_accuracy"] == pytest.approx(2 / 3)  # ok + wrong-cite đúng nội dung
    assert summary["false_refusal_rate"] == pytest.approx(1 / 3)
    assert summary["refusal_accuracy"] == 0.5 and summary["hallucination_rate"] == 0.5
    assert summary["citation_accuracy"] == 0.5  # 1/2 câu được trả lời có nguồn đúng
    assert summary["overall_accuracy"] == pytest.approx(3 / 5)
    assert judged == [("q-ok", "ctx", "Tối đa 2 ngày.")]  # chỉ chấm câu có contexts và không từ chối
    assert summary["faithfulness"] == 1.0
    assert {r["id"]: ev._failure_reason(r) for r in records}["boom"].startswith("lỗi")


# ---------------------------------------------------------------- CLI helpers & report


def test_parse_overrides_rejects_unknown_key():
    assert ev.parse_overrides(["top_k=3", "llm_model=x:1b"]) == {"top_k": "3", "llm_model": "x:1b"}
    with pytest.raises(SystemExit):
        ev.parse_overrides(["top_kk=3"])
    with pytest.raises(SystemExit):
        ev.parse_overrides(["top_k"])


def test_committed_baseline_report_renders():
    report = json.loads((ROOT / "evals" / "baseline.json").read_text(encoding="utf-8"))
    text = ev.format_summary(report)
    assert "== Retrieval" in text and "== Generation" in text
