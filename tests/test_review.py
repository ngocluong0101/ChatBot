import json
from pathlib import Path

from evals import review as rv
from evals.common import read_jsonl, write_jsonl
from ragbot.evaluate import load_dataset


def draft_row(id_: str, category: str = "factoid", **extra) -> dict:
    base = {
        "id": id_,
        "question": f"Câu hỏi số {id_}?",
        "category": category,
        "answerable": True,
        "gold_sources": ["a.pdf"],
        "evidence": ["nghỉ hằng năm 12 ngày"],
        "must_include": [["12 ngày", "mười hai ngày"]],
        "difficulty": "medium",
        "verified": False,
        "split": "dev",
        "_review": {
            "chunks": [
                {
                    "doc_title": "Luật A",
                    "section": "Điều 5. Nghỉ",
                    "page": 3,
                    "text": "Được nghỉ hằng năm 12 ngày.",
                }
            ],
            "chunk_keys": [["a.pdf", "Điều 5. Nghỉ", 3, None, "text"]],
            "warnings": [],
        },
    }
    return {**base, **extra}


def unanswerable_row(id_: str) -> dict:
    return {
        "id": id_,
        "question": f"Câu ngoài phạm vi {id_}?",
        "category": "off_topic",
        "answerable": False,
        "difficulty": "easy",
        "verified": False,
        "split": "test",
        "_review": {"chunks": [], "chunk_keys": [], "warnings": ["kiểm tra lại"]},
    }


class Script:
    """Giả bàn phím: trả lần lượt các câu trả lời đã định và ghi lại các câu hỏi đã hiện ra."""

    def __init__(self, *answers: str):
        self.answers, self.prompts, self.printed = list(answers), [], []

    def input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def print(self, text: str) -> None:
        self.printed.append(text)


def run_review(rows: list[dict], path: Path, *answers: str, search=None) -> tuple[bool, Script]:
    write_jsonl(path, rows)
    script = Script(*answers)
    done = rv.review_loop(rows, path, script.input, script.print, search)
    return done, script


def test_accepting_and_rejecting_are_saved_immediately(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    rows = [draft_row("a"), draft_row("b"), draft_row("c")]
    done, _ = run_review(rows, path, "y", "n", "y")

    saved = {r["id"]: r for r in read_jsonl(path)}
    assert done
    assert (saved["a"]["review"], saved["a"]["verified"]) == ("accepted", True)
    assert (saved["b"]["review"], saved["b"]["verified"]) == ("rejected", False)
    assert rv.progress(read_jsonl(path)) == {"accepted": 2, "rejected": 1, "pending": 0}


def test_quitting_midway_keeps_decisions_and_resume_skips_decided_rows(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    rows = [draft_row("a"), draft_row("b"), draft_row("c")]
    done, _ = run_review(rows, path, "y", "q")
    assert not done and rv.progress(read_jsonl(path)) == {"accepted": 1, "rejected": 0, "pending": 2}

    resumed = read_jsonl(path)
    done, script = run_review(resumed, path, "n", "y")
    assert done and len(script.prompts) == 2  # chỉ còn hai câu chưa quyết định
    assert [r["review"] for r in read_jsonl(path)] == ["accepted", "rejected", "accepted"]


def test_skipped_rows_stay_pending(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    done, _ = run_review([draft_row("a"), draft_row("b")], path, "s", "y")
    assert not done
    assert rv.progress(read_jsonl(path)) == {"accepted": 1, "rejected": 0, "pending": 1}


def test_invalid_key_is_asked_again(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    done, script = run_review([draft_row("a")], path, "x", "", "y")
    assert done and "Phím không hợp lệ." in script.printed
    assert read_jsonl(path)[0]["review"] == "accepted"


def test_edit_changes_question_answer_groups_and_difficulty_then_asks_again(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    done, script = run_review(
        [draft_row("a")],
        path,
        "e",
        "Một năm làm đủ 12 tháng thì nghỉ mấy ngày?",  # câu hỏi mới
        "12 ngày | 12 ngày làm việc ; Luật A",  # đáp án mới
        "hard",
        "y",
    )
    row = read_jsonl(path)[0]
    assert done and row["question"] == "Một năm làm đủ 12 tháng thì nghỉ mấy ngày?"
    assert row["must_include"] == [["12 ngày", "12 ngày làm việc"], ["Luật A"]]
    assert row["difficulty"] == "hard" and row["review"] == "accepted"
    assert any("không có trong đoạn gốc" in line for line in script.printed)  # "Luật A" không nằm trong đoạn


def test_edit_with_enter_keeps_everything(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    run_review([draft_row("a")], path, "e", "", "", "", "n")
    row = read_jsonl(path)[0]
    assert row["question"] == "Câu hỏi số a?" and row["must_include"] == [["12 ngày", "mười hai ngày"]]


def test_search_shows_hits_then_returns_to_the_same_question(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    calls = []

    def search(query: str) -> list[str]:
        calls.append(query)
        return ["  [3.2] a.pdf | Điều 5"]

    done, script = run_review([unanswerable_row("u")], path, "g", "", "y", search=search)
    assert done and calls == ["Câu ngoài phạm vi u?"]  # Enter = dùng chính câu hỏi làm từ khóa
    assert "  [3.2] a.pdf | Điều 5" in script.printed
    assert read_jsonl(path)[0]["review"] == "accepted"


def test_render_highlights_the_answer_and_shows_warnings():
    text = rv.render(draft_row("a", _review={**draft_row("a")["_review"], "warnings": ["cẩn thận"]}), "(1/5)")
    assert "[[12 ngày]]" in text and "Luật A, Điều 5. Nghỉ" in text and "⚠ cẩn thận" in text
    unanswerable = rv.render(unanswerable_row("u"))
    assert "KHÔNG CÓ ĐÁP ÁN" in unanswerable and "⚠ kiểm tra lại" in unanswerable


def test_group_formatting_round_trips():
    groups = [["12 ngày", "mười hai ngày"], ["VPN"]]
    assert rv.parse_groups(rv.format_groups(groups)) == groups
    assert rv.parse_groups(" ; | ") == []


# ---------------------------------------------------------------- finalize


def test_finalize_writes_only_accepted_rows_without_private_fields(tmp_path: Path):
    draft, dataset = tmp_path / "d.jsonl", tmp_path / "ds.jsonl"
    rows = [
        draft_row("a", review="accepted", verified=True),
        draft_row("b", review="rejected"),
        draft_row("c"),
    ]
    rows.append(unanswerable_row("u") | {"review": "accepted", "verified": True})
    write_jsonl(draft, rows)

    summary = rv.finalize([draft], dataset)

    saved = read_jsonl(dataset)
    assert [r["id"] for r in saved] == ["a", "u"] and summary["added"] == 2
    assert all(r["verified"] is True and not set(r) & {"_review", "review"} for r in saved)
    assert "gold_sources" not in saved[1]  # câu không có đáp án không mang trường đáp án
    assert [e.id for e in load_dataset(dataset)] == ["a", "u"]  # ragbot.evaluate đọc được


def test_finalize_is_idempotent_and_never_changes_existing_splits(tmp_path: Path):
    draft, dataset = tmp_path / "d.jsonl", tmp_path / "ds.jsonl"
    first = [draft_row(f"a{i}", review="accepted", verified=True) for i in range(6)]
    write_jsonl(draft, first)
    rv.finalize([draft], dataset)
    before = {r["id"]: r["split"] for r in read_jsonl(dataset)}
    assert sorted(set(before.values())) == ["dev", "test"]

    again = rv.finalize([draft], dataset)
    assert again["added"] == 0 and {r["id"]: r["split"] for r in read_jsonl(dataset)} == before

    more = [draft_row(f"b{i}", review="accepted", verified=True) for i in range(4)]
    write_jsonl(draft, first + more)
    rv.finalize([draft], dataset)
    after = {r["id"]: r["split"] for r in read_jsonl(dataset)}
    assert all(after[k] == v for k, v in before.items())  # split cũ giữ nguyên
    assert abs(sum(v == "dev" for v in after.values()) - 5) <= 1  # câu mới được dồn cho cân bằng


def test_finalize_reports_the_split_table(tmp_path: Path):
    draft, dataset = tmp_path / "d.jsonl", tmp_path / "ds.jsonl"
    write_jsonl(draft, [draft_row(f"a{i}", review="accepted") for i in range(4)])
    table = rv.finalize([draft], dataset)["by_category"]
    assert table == {"factoid": {"dev": 2, "test": 2}}


def test_dataset_rows_survive_json_roundtrip_with_vietnamese_text(tmp_path: Path):
    dataset = tmp_path / "ds.jsonl"
    draft = tmp_path / "d.jsonl"
    write_jsonl(draft, [draft_row("a", review="accepted", question="Người lao động được nghỉ mấy ngày?")])
    rv.finalize([draft], dataset)
    assert "Người lao động" in dataset.read_text(encoding="utf-8")  # không bị escape \uXXXX
    assert json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])["question"].startswith("Người")
