import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import judge_check as jc
from evals.common import read_jsonl, write_jsonl


def test_perturb_changes_exactly_one_number_and_keeps_zero_padding():
    out = jc.perturb_answer("Tối đa 05 ngày, báo trước 45 ngày.", random.Random(0))
    assert out != "Tối đa 05 ngày, báo trước 45 ngày."
    assert out.count("ngày") == 2 and len(out.split(" ")) == 8
    original = ["Tối", "đa", "05", "ngày,", "báo", "trước", "45", "ngày."]
    diff = [(a, b) for a, b in zip(original, out.split(), strict=True) if a != b]
    assert len(diff) == 1
    padded = jc.perturb_answer("Thử việc 06 tháng.", random.Random(1))
    number = padded.split()[2]
    assert len(number) == 2 and number != "06"  # giữ độ rộng hai chữ số


def test_perturb_doubles_large_numbers_and_skips_answers_without_numbers():
    assert jc.perturb_answer("Mức phạt 200 đồng.", random.Random(0)) == "Mức phạt 400 đồng."
    assert jc.perturb_answer("Không có con số nào.", random.Random(0)) is None


def test_cohen_kappa_known_values():
    assert jc.cohen_kappa([(True, True), (False, False)] * 5) == 1.0
    assert jc.cohen_kappa([(True, False), (False, True)] * 5) == -1.0
    # đồng ý 3/4 nhưng một người luôn nói "đúng": kappa = 0
    assert jc.cohen_kappa([(True, True)] * 3 + [(True, False)]) == pytest.approx(0.0)
    assert jc.cohen_kappa([(True, True)] * 4) is None  # không có biến thiên: kappa không xác định
    assert jc.cohen_kappa([]) is None


def rows():
    return [
        {"id": "jc-01", "judge": True, "human": True},
        {"id": "jc-02", "judge": True, "human": False},  # judge bỏ sót câu sai
        {"id": "jc-03", "judge": False, "human": False},
        {"id": "jc-04", "judge": False, "human": False},
        {"id": "jc-05", "judge": None, "human": True},  # judge không đọc được
        {"id": "jc-06", "judge": True, "human": None},  # chưa chấm tay
    ]


def test_score_rows_counts_agreement_and_how_many_bad_answers_the_judge_catches():
    origins = {"jc-01": "model", "jc-02": "perturbed", "jc-03": "perturbed", "jc-04": "perturbed"}
    result = jc.score_rows(rows(), origins)

    assert (result["rows"], result["human_labeled"], result["judge_unparsable"], result["n"]) == (6, 5, 1, 4)
    assert result["agreement"] == 0.75
    assert result["human_unfaithful"] == 3  # jc-02, jc-03, jc-04 (jc-05 judge không đọc được nên bị loại)
    assert result["catches_unfaithful"] == pytest.approx(2 / 3)
    assert result["accepts_faithful"] == 1.0
    assert result["by_origin"]["model"]["n"] == 1 and result["by_origin"]["perturbed"]["n"] == 3


def test_format_score_warns_when_too_few_labels():
    text = jc.format_score(jc.score_rows(rows(), {}))
    assert "Dưới 30 câu" in text and "Cohen's kappa" in text


class Pipeline:
    def __init__(self):
        self.asked = []

    def ask(self, question: str):
        self.asked.append(question)
        if "từ chối" in question:
            return SimpleNamespace(refused=True, text="không tìm thấy", contexts=())
        return SimpleNamespace(
            refused=False, text=f"Trả lời 12 ngày cho {question}", contexts=("ngữ cảnh 1", "ngữ cảnh 2")
        )


def example(i: int, answerable: bool = True, question: str | None = None):
    return SimpleNamespace(id=f"q{i}", answerable=answerable, question=question or f"câu hỏi {i}?")


def test_build_rows_mixes_real_and_perturbed_answers_and_hides_origin_from_the_rows():
    judged = []

    def judge(question, context, answer):
        judged.append(answer)
        return "12 ngày" in answer  # judge giả: bắt được câu bị đổi số

    examples = [example(i) for i in range(30)] + [
        example(99, answerable=False),
        example(98, question="câu từ chối?"),
    ]
    rows, origins = jc.build_rows(Pipeline(), judge, examples, n_real=5, n_perturbed=3, seed=1)

    assert len(rows) == 8 and sorted(origins.values()).count("perturbed") == 3
    assert [r["id"] for r in rows] == [f"jc-{i:02d}" for i in range(1, 9)]
    assert all(set(r) == {"id", "question", "context", "answer", "judge", "human"} for r in rows)
    assert all(r["human"] is None and "ngữ cảnh 1" in r["context"] for r in rows)
    by_id = {r["id"]: r for r in rows}
    perturbed = [by_id[i] for i, origin in origins.items() if origin == "perturbed"]
    assert all("12 ngày" not in r["answer"] and r["judge"] is False for r in perturbed)
    assert all(r["judge"] is True for i, r in by_id.items() if origins[i] == "model")
    assert len(judged) == 8


def test_build_rows_is_reproducible_and_never_asks_unanswerable_questions():
    pipeline = Pipeline()
    examples = [example(i) for i in range(20)] + [example(99, answerable=False, question="ngoài phạm vi?")]
    first = jc.build_rows(pipeline, lambda *a: True, examples, 4, 2, seed=3)
    second = jc.build_rows(Pipeline(), lambda *a: True, examples, 4, 2, seed=3)
    assert first == second
    assert "ngoài phạm vi?" not in pipeline.asked


def test_label_loop_saves_each_label_and_can_be_resumed(tmp_path: Path):
    path = tmp_path / "jc.jsonl"
    data = [
        {"id": "jc-01", "question": "q1", "context": "c1", "answer": "a1", "judge": True, "human": None},
        {"id": "jc-02", "question": "q2", "context": "c2", "answer": "a2", "judge": False, "human": None},
        {"id": "jc-03", "question": "q3", "context": "c3", "answer": "a3", "judge": True, "human": None},
    ]
    write_jsonl(path, data)
    answers = iter(["x", "f", "u", "q"])
    done = jc.label_loop(data, path, lambda prompt: next(answers), lambda text: None)

    assert not done
    assert [r["human"] for r in read_jsonl(path)] == [True, False, None]

    answers = iter(["f"])
    prompts = []
    done = jc.label_loop(
        read_jsonl(path), path, lambda p: prompts.append(p) or next(answers), lambda text: None
    )
    assert done and len(prompts) == 1  # chỉ còn một câu chưa chấm
    assert [r["human"] for r in read_jsonl(path)] == [True, False, True]


def test_skipped_rows_are_left_unlabeled(tmp_path: Path):
    path = tmp_path / "jc.jsonl"
    data = [{"id": "jc-01", "question": "q", "context": "c", "answer": "a", "judge": True, "human": None}]
    write_jsonl(path, data)
    assert not jc.label_loop(data, path, lambda p: "s", lambda t: None)
    assert read_jsonl(path)[0]["human"] is None
