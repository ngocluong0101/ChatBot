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


def test_legacy_dataset_is_well_formed():
    """Bộ 53 câu cũ (từng dùng cho bộ papers/ đã bỏ) giữ lại làm tham chiếu lịch sử; đảm bảo vẫn đọc được."""
    examples = ev.load_dataset(Path(__file__).parent.parent / "evals" / "dataset.jsonl")
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


# ---------------------------------------------------------------- schema mới, split, ngưỡng, CI


def write_dataset(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def row(id_: str, **extra) -> dict:
    base = {
        "id": id_,
        "question": f"câu {id_}?",
        "category": "factoid",
        "answerable": True,
        "gold_sources": ["a.pdf"],
        "evidence": ["tối đa 2 ngày"],
        "must_include": [["2 ngày"]],
    }
    return {**base, **extra}


def test_new_schema_fields_default_so_old_rows_still_load(tmp_path: Path):
    path = write_dataset(
        tmp_path / "d.jsonl", [row("old"), row("new", split="test", difficulty="hard", verified=False)]
    )
    old, new = ev.load_dataset(path)
    assert (old.split, old.difficulty, old.verified) == (None, None, True)
    assert (new.split, new.difficulty, new.verified) == ("test", "hard", False)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"split": "val"}, "split"),
        ({"difficulty": "extreme"}, "difficulty"),
        ({"verified": "yes"}, "verified"),
    ],
)
def test_new_schema_fields_are_validated(tmp_path: Path, extra: dict, message: str):
    path = write_dataset(tmp_path / "d.jsonl", [row("a", **extra)])
    with pytest.raises(ValueError, match=message):
        ev.load_dataset(path)


def test_select_examples_filters_split_and_unverified():
    examples = [
        example(id="d-ok", split="dev"),
        example(id="d-draft", split="dev", verified=False),
        example(id="t-ok", split="test"),
        example(id="legacy"),
    ]
    assert [e.id for e in ev.select_examples(examples, "dev")] == ["d-ok"]
    assert [e.id for e in ev.select_examples(examples, "dev", include_unverified=True)] == ["d-ok", "d-draft"]
    assert [e.id for e in ev.select_examples(examples, "test")] == ["t-ok"]
    assert [e.id for e in ev.select_examples(examples, "all")] == ["d-ok", "t-ok", "legacy"]


def test_default_split_is_dev_only_for_datasets_that_have_splits():
    assert ev.default_split([example(), example(id="b")]) == "all"
    assert ev.default_split([example(split="test"), example(id="b")]) == "dev"


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [1.0] * 30 + [0.0] * 10
    low, high = ev.bootstrap_ci(values)
    assert ev.bootstrap_ci(values) == [low, high]
    assert 0.0 <= low < 0.75 < high <= 1.0
    assert ev.bootstrap_ci([1.0] * 20) == [1.0, 1.0]
    assert ev.bootstrap_ci([]) is None


def test_ci_gets_narrower_with_more_data():
    small = ev.bootstrap_ci([1.0, 0.0] * 5)
    large = ev.bootstrap_ci([1.0, 0.0] * 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_summaries_carry_confidence_intervals():
    rec = {"answerable": True, "category": "a", "source_recall": 1.0, "context_precision": 1.0}
    s = ev.summarize_retrieval([{**rec, "first_rank": 1}, {**rec, "first_rank": None}], top_k=5)
    assert s["hit@1"] == 0.5 and len(s["hit@1_ci95"]) == 2
    g = ev.summarize_generation(
        [
            {
                "answerable": True,
                "category": "a",
                "refused": False,
                "correct": True,
                "citation_ok": True,
                "latency": 1.0,
            },
            {"answerable": False, "category": "o", "refused": True, "correct_refusal": True, "latency": 1.0},
        ]
    )
    assert g["answerable_accuracy_ci95"] == [1.0, 1.0] and g["refusal_accuracy_ci95"] == [1.0, 1.0]


# --- bằng chứng bị cắt ngang giữa hai chunk


LONG = "người lao động làm việc đủ 12 tháng thì được nghỉ hằng năm hưởng nguyên lương"


def chunk(text: str, source: str = "a.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_evidence_split_across_two_chunks_still_counts_for_both_halves():
    ex = example(evidence=(LONG,))
    cut = len(LONG) // 2
    head = chunk("Điều 113. Nghỉ hằng năm. " + LONG[:cut])
    tail = chunk(LONG[cut:] + " theo hợp đồng lao động. Điều 114 quy định khác.")
    assert ev.is_relevant_chunk(head, ex) and ev.is_relevant_chunk(tail, ex)


def test_partial_evidence_in_the_middle_of_a_chunk_does_not_count():
    ex = example(evidence=(LONG,))
    middle = chunk("mở đầu dài dòng " * 5 + LONG[:40] + " rồi chuyển sang nội dung khác hẳn " * 5)
    assert not ev.is_relevant_chunk(middle, ex)


def test_short_evidence_must_match_exactly():
    ex = example(evidence=("tối đa 2 ngày",))
    assert not ev.is_relevant_chunk(chunk("tối đa 2 ng"), ex)


# --- retrieval đi qua bước viết lại câu hỏi


class RetrievalStub:
    def __init__(self, rewrites: dict[str, str], results: dict[str, list], fail: set[str] = frozenset()):
        self.rewrites, self.results, self.fail, self.searched = rewrites, results, fail, []

    def standalone_question(self, question: str) -> str:
        if question in self.fail:
            raise LLMUnavailableError("ollama tắt")
        return self.rewrites.get(question, question)

    def search(self, question: str):
        self.searched.append(question)
        return self.results.get(question, [])


def test_retrieval_eval_searches_with_the_rewritten_question_and_records_errors():
    from ragbot.config import Settings

    good = chunk("được đăng ký tối đa 2 ngày làm việc")
    pipeline = RetrievalStub(
        rewrites={"toi da may ngay": "Tối đa mấy ngày?"},
        results={"Tối đa mấy ngày?": [(good, 0.9)], "q-plain": [(chunk("khác"), 0.1)]},
        fail={"q-down"},
    )
    examples = [
        example(id="unaccented", question="toi da may ngay", split="dev"),
        example(id="plain", question="q-plain"),
        example(id="down", question="q-down"),
    ]
    records, summary = ev.evaluate_retrieval(pipeline, examples, Settings(rerank_model=""))

    assert pipeline.searched == ["Tối đa mấy ngày?", "q-plain"]
    by_id = {r["id"]: r for r in records}
    assert by_id["unaccented"]["rewritten"] == "Tối đa mấy ngày?" and by_id["unaccented"]["first_rank"] == 1
    assert by_id["unaccented"]["split"] == "dev"
    assert "rewritten" not in by_id["plain"] and by_id["plain"]["first_rank"] is None
    assert "error" in by_id["down"]
    assert summary["n_answerable"] == 2 and summary["errors"] == 1  # câu lỗi không bị tính là trượt


# --- chọn ngưỡng chỉ trên dev


def retrieval_records():
    return [
        {
            "id": "d1",
            "answerable": True,
            "category": "a",
            "split": "dev",
            "top1_score": 0.8,
            "first_rank": 1,
            "source_recall": 1.0,
            "context_precision": 1.0,
        },
        {"id": "d2", "answerable": False, "category": "o", "split": "dev", "top1_score": 0.2},
        # test có điểm cực đoan: nếu bị lẫn vào quét ngưỡng thì min/max sẽ đổi
        {
            "id": "t1",
            "answerable": True,
            "category": "a",
            "split": "test",
            "top1_score": 0.01,
            "first_rank": 1,
            "source_recall": 1.0,
            "context_precision": 1.0,
        },
        {"id": "t2", "answerable": False, "category": "o", "split": "test", "top1_score": 0.99},
    ]


def test_report_sweeps_thresholds_on_dev_records_only_and_splits_results():
    from ragbot.config import Settings

    records = retrieval_records()
    report = ev.build_report(
        Settings(rerank_model=""),
        "retrieval",
        [],
        (records, ev.summarize_retrieval(records, 5)),
        None,
        None,
        split="all",
    )
    sweep = report["threshold_sweep"]
    assert (sweep["min_answerable_top1"], sweep["max_unanswerable_top1"]) == (0.8, 0.2)
    assert set(report["by_split"]) == {"dev", "test"}
    assert report["by_split"]["test"]["retrieval"]["n_answerable"] == 1


def test_test_only_report_has_no_threshold_sweep():
    from ragbot.config import Settings

    records = [r for r in retrieval_records() if r["split"] == "test"]
    report = ev.build_report(
        Settings(rerank_model=""),
        "retrieval",
        [],
        (records, ev.summarize_retrieval(records, 5)),
        None,
        None,
        split="test",
    )
    assert report["threshold_sweep"] is None and "by_split" not in report
    assert "không quét ngưỡng" in ev.format_summary(report)


def dev_report(tmp_path: Path, **meta) -> Path:
    from ragbot.config import Settings

    records = [r for r in retrieval_records() if r["split"] == "dev"]
    report = ev.build_report(
        Settings(rerank_model="BAAI/bge-reranker-v2-m3"),
        "retrieval",
        [],
        (records, ev.summarize_retrieval(records, 5)),
        None,
        None,
        split="dev",
    )
    report["meta"].update(meta)
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_threshold_from_dev_report_picks_the_right_setting_key(tmp_path: Path):
    from ragbot.config import Settings

    key, value, info = ev.threshold_from_report(
        dev_report(tmp_path), Settings(rerank_model="BAAI/bge-reranker-v2-m3")
    )
    assert key == "rerank_threshold" and 0.2 < value <= 0.8
    assert info["from"] == "dev.json" and info["dev_unanswerable_rejected"] == 1.0


def test_threshold_from_rejects_non_dev_reports_and_mismatched_reranker(tmp_path: Path):
    from ragbot.config import Settings

    with pytest.raises(SystemExit, match="dev"):
        ev.threshold_from_report(dev_report(tmp_path, split="test"), Settings())
    with pytest.raises(SystemExit, match="reranker"):
        ev.threshold_from_report(dev_report(tmp_path), Settings(rerank_model=""))


def test_cli_refuses_test_split_without_a_threshold_chosen_on_dev(tmp_path: Path):
    path = write_dataset(tmp_path / "d.jsonl", [row("a", split="test")])
    with pytest.raises(SystemExit, match="dev"):
        ev.main(["--dataset", str(path), "--split", "test", "--mode", "retrieval"])


def test_cli_exits_when_no_example_is_selected(tmp_path: Path, capsys):
    path = write_dataset(tmp_path / "d.jsonl", [row("draft", split="dev", verified=False)])
    assert ev.main(["--dataset", str(path), "--split", "dev", "--mode", "retrieval"]) == 2
    assert "Không có câu hỏi nào" in capsys.readouterr().err


def test_select_examples_limit_is_a_reproducible_sample_keeping_order():
    examples = [example(id=f"e{i}", split="dev") for i in range(20)]
    first = [e.id for e in ev.select_examples(examples, "dev", limit=5)]
    assert first == [e.id for e in ev.select_examples(examples, "dev", limit=5)]
    assert len(first) == 5 and first == sorted(first, key=lambda i: int(i[1:]))
    assert len(ev.select_examples(examples, "dev", limit=50)) == 20  # limit lớn hơn số câu: lấy hết
