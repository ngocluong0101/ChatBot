import json
import random
import re

import pytest
from langchain_core.documents import Document

from evals import draft_questions as dq
from evals.common import assign_splits
from ragbot.evaluate import is_relevant_chunk, load_dataset, normalize

TOPICS = [
    # (chủ đề, giá trị đáp án, câu hỏi do "LLM" viết)
    ("thử việc", "60 ngày", "Thời gian thử việc tối đa đối với công việc cần trình độ cao đẳng là bao lâu?"),
    ("nghỉ phép", "12 ngày", "Một năm làm việc đủ 12 tháng thì người lao động được nghỉ phép mấy ngày?"),
    (
        "làm thêm giờ",
        "300 giờ",
        "Doanh nghiệp được huy động làm thêm tối đa bao nhiêu giờ mỗi năm ở ngành đặc biệt?",
    ),
    ("tiền lương", "05 ngày", "Tiền lương phải được trả chậm nhất sau bao nhiêu ngày khi chấm dứt hợp đồng?"),
    ("kỷ luật", "06 tháng", "Thời hiệu xử lý kỷ luật lao động thông thường kéo dài trong bao lâu?"),
    (
        "đình công",
        "03 ngày",
        "Người lao động phải báo trước cho người sử dụng lao động mấy ngày trước khi ngừng việc?",
    ),
    ("học nghề", "03 tháng", "Hợp đồng đào tạo nghề dưới một năm có thời hạn thử thách tối đa là bao nhiêu?"),
    ("tai nạn", "30 phút", "Khi xảy ra sự cố, đơn vị phải khai báo trong vòng bao nhiêu phút đầu tiên?"),
]
FILLER = (
    " Quy định này áp dụng đối với mọi người lao động, người sử dụng lao động và các tổ chức đại diện "
    "có liên quan trong quá trình thực hiện hợp đồng lao động, bảo đảm quyền và lợi ích hợp pháp của các bên."
)


def make_chunk(
    source: str, number: int, topic: str, value: str, extra: str = "", kind: str = "text"
) -> Document:
    body = f"1. Về {topic}, mức quy định là {value} theo thỏa thuận hợp pháp.{FILLER}{extra}"
    section = f"Điều {number}. Quy định về {topic}"
    return Document(
        page_content=f"Luật {source}\n{section}\n{body}",
        metadata={
            "source": source,
            "doc_title": f"Luật {source}",
            "section": section,
            "dieu": str(number),
            "dieu_title": f"Quy định về {topic}",
            "page": number,
            "type": kind,
        },
    )


def corpus() -> list[Document]:
    chunks = [make_chunk("a.pdf", i + 1, t, v) for i, (t, v, _) in enumerate(TOPICS)]
    chunks += [
        make_chunk("b.pdf", i + 1, f"{t} nâng cao", f"{9 + i} lần") for i, (t, _, _) in enumerate(TOPICS)
    ]
    return chunks


class StubLLM:
    """Giả LLM: tra câu hỏi theo chủ đề xuất hiện trong prompt; có thể được cấu hình để trả câu hỏng."""

    def __init__(self, mode: str = "good"):
        self.mode, self.prompts = mode, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "ĐOẠN A" in prompt:
            a = re.search(r"mức quy định là (.+?) theo", prompt.split("ĐOẠN B")[0]).group(1)
            b = re.search(r"mức quy định là (.+?) theo", prompt.split("ĐOẠN B")[1]).group(1)
            return json.dumps(
                {
                    "question": f"Kết hợp hai quy định thì cần bao nhiêu thời hạn {len(self.prompts)}?",
                    "answer_a": [a],
                    "answer_b": [b],
                    "difficulty": "hard",
                }
            )
        topic = next(
            ((t, v, q) for t, v, q in TOPICS if f"Về {t}," in prompt and "nâng cao" not in prompt), None
        )
        if topic is None:
            return json.dumps(
                {"question": "Không có chủ đề nào khớp với đoạn văn bản này?", "answer_keys": [["x"]]}
            )
        t, v, q = topic
        if self.mode == "bad-key":
            return json.dumps({"question": q, "answer_keys": [["999 năm"]]})
        if self.mode == "leak":
            return json.dumps({"question": q.rstrip("?") + f", có phải {v} không?", "answer_keys": [[v]]})
        if self.mode == "not-json":
            return "xin lỗi, tôi không biết"
        return json.dumps({"question": q, "answer_keys": [[v]], "difficulty": "easy"})


def run(counts, llm=None, existing=None, seed=1):
    llm = llm or StubLLM()
    return dq.draft(corpus(), llm, counts, seed, existing), llm


# ---------------------------------------------------------------- helpers


def test_body_of_drops_the_repeated_header():
    chunk = make_chunk("a.pdf", 5, "nghỉ phép", "12 ngày")
    assert dq.body_of(chunk).startswith("1. Về nghỉ phép")
    assert "Luật a.pdf" not in dq.body_of(chunk)


def test_evidence_is_verbatim_word_aligned_and_contains_the_key():
    chunk = make_chunk("a.pdf", 5, "nghỉ phép", "12 ngày")
    evidence = dq.evidence_for(chunk, ["12 ngày"], width=30)
    assert "12 ngày" in evidence and len(evidence) < 90
    assert normalize(evidence) in normalize(chunk.page_content)
    assert not evidence.startswith(" ") and evidence[0] != evidence[0].lower() or evidence[0].isalnum()
    assert dq.evidence_for(chunk, ["không có"], width=30) is None


def test_evidence_tries_alternatives_in_order():
    chunk = make_chunk("a.pdf", 5, "nghỉ phép", "12 ngày")
    assert "12 ngày" in dq.evidence_for(chunk, ["mười hai ngày", "12 ngày"])


def test_clean_groups_normalizes_llm_output():
    assert dq.clean_groups([["12 ngày", " mười  hai ngày ", ""], "VPN", [], 5]) == [
        ["12 ngày", "mười hai ngày"],
        ["VPN"],
    ]
    assert dq.clean_groups("không phải danh sách") == []
    assert len(dq.clean_groups([["a"], ["b"], ["c"]])) == 2  # tối đa 2 nhóm


@pytest.mark.parametrize(
    "question, reason",
    [
        ("Ngắn?", "quá ngắn"),
        ("Người lao động được nghỉ phép mấy ngày trong một năm", "dấu ?"),
        ("Theo đoạn văn trên, người lao động nghỉ mấy ngày trong năm?", "tham chiếu"),
        ("Người lao động nghỉ phép 12 ngày mỗi năm đúng không?", "đáp án"),
        (None, "chuỗi"),
    ],
)
def test_question_problem_rejects_bad_questions(question, reason):
    assert reason in dq.question_problem(question, [["12 ngày"]])


def test_question_problem_accepts_a_good_question():
    assert (
        dq.question_problem("Người lao động nghỉ phép mấy ngày trong một năm làm việc?", [["12 ngày"]])
        is None
    )


def test_to_unaccented_strips_marks_and_lowercases():
    out = dq.to_unaccented("Mức lương tối thiểu được Đóng thế nào?", random.Random(0))
    assert out.startswith("muc luong toi thieu duoc dong the nao")
    assert out == out.lower() and out.isascii()


def test_weighted_order_is_a_deterministic_permutation():
    chunks = corpus()
    first = dq.weighted_order(chunks, random.Random(3), prefer_numbers=True)
    assert sorted(map(dq.chunk_key, first)) == sorted(map(dq.chunk_key, chunks))
    assert first == dq.weighted_order(chunks, random.Random(3), prefer_numbers=True)


def test_family_titles_flags_documents_with_overlapping_articles():
    a = [make_chunk("old.pdf", i, t, "1 ngày") for i, (t, _, _) in enumerate(TOPICS)]
    b = [make_chunk("new.pdf", i, t, "2 ngày") for i, (t, _, _) in enumerate(TOPICS)]
    other = [make_chunk("other.pdf", 1, "chủ đề hoàn toàn khác", "3 ngày")]
    assert dq.family_titles(a + b + other) == {"old.pdf", "new.pdf"}


def test_parse_counts_validates_names():
    assert dq.parse_counts("factoid=3,table=1") == {"factoid": 3, "table": 1}
    assert dq.parse_counts(None)["factoid"] == dq.DEFAULT_COUNTS["factoid"]
    with pytest.raises(SystemExit):
        dq.parse_counts("khong-co=3")


# ---------------------------------------------------------------- soạn nháp


def test_factoid_drafts_are_unverified_grounded_and_pass_the_evaluators_own_checks(tmp_path):
    rows, _ = run({"factoid": 4})
    factoid = [r for r in rows if r["category"] == "factoid"]

    assert len(factoid) == 4 and all(r["verified"] is False and r["answerable"] for r in factoid)
    assert len({r["id"] for r in factoid}) == 4 and all(r["id"].startswith("fa-") for r in factoid)
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    examples = load_dataset(path)  # trường phụ _review bị bỏ qua, schema hợp lệ
    chunks_by_source = {c.metadata["source"]: c for c in corpus() if c.metadata["source"] == "a.pdf"}
    assert chunks_by_source  # bảo đảm fixture
    for ex in examples:
        assert any(is_relevant_chunk(c, ex) for c in corpus()), (
            f"bằng chứng của {ex.id} không nằm trong corpus"
        )


def test_questions_whose_answer_is_not_verbatim_in_the_chunk_are_dropped():
    rows, _ = run({"factoid": 3}, StubLLM("bad-key"))
    assert rows == []


def test_questions_that_leak_the_answer_are_dropped():
    rows, _ = run({"factoid": 3}, StubLLM("leak"))
    assert rows == []


def test_garbage_llm_output_is_skipped_not_fatal():
    rows, _ = run({"factoid": 3}, StubLLM("not-json"))
    assert rows == []


def test_duplicate_questions_are_not_accepted_twice():
    chunks = corpus() + [make_chunk("c.pdf", 1, "nghỉ phép", "12 ngày")]  # cùng chủ đề -> cùng câu hỏi
    rows = dq.draft(chunks, StubLLM(), {"factoid": 20}, 1)
    questions = [r["question"] for r in rows]
    assert len(questions) == len(set(questions))
    assert sum("nghỉ phép" in q for q in questions) <= 1


def test_question_with_the_same_answer_in_another_document_is_dropped_as_ambiguous():
    chunks = corpus() + [make_chunk("b.pdf", 99, "thử việc", "60 ngày")]  # văn bản khác chứa cùng đáp án
    rows = dq.draft(chunks, StubLLM(), {"factoid": 20}, 1)
    assert not any("thử việc" in r["question"].lower() for r in rows)


def test_paraphrase_that_copies_the_source_wording_is_rejected():
    chunk = make_chunk("a.pdf", 7, "nghỉ phép", "12 ngày")
    copied = " ".join(dq.body_of(chunk).split()[:12]).rstrip(".") + " là bao nhiêu?"
    assert dq.word_overlap(copied, dq.body_of(chunk)) > dq.MAX_PARAPHRASE_OVERLAP
    assert dq.word_overlap("Người ta được nghỉ bao lâu?", "xyz") < 0.5


def test_no_diacritics_category_is_the_factoid_question_without_accents():
    rows, _ = run({"no_diacritics": 3})
    assert len(rows) == 3
    assert all(r["category"] == "no_diacritics" and r["question"].isascii() for r in rows)
    assert all(r["must_include"] and r["evidence"] for r in rows)


def test_multi_source_questions_list_both_sources_and_evidence():
    rows, _ = run({"multi_source": 3})
    assert rows, "phải soạn được ít nhất một câu ghép"
    for r in rows:
        assert r["category"] == "multi_source" and r["difficulty"] == "hard"
        assert len(r["evidence"]) == 2 and len(r["must_include"]) == 2
        assert set(r["gold_sources"]) <= {"a.pdf", "b.pdf"}
        assert len(r["_review"]["chunks"]) == 2


def test_table_questions_reuse_the_small_set_of_tables():
    table = make_chunk("a.pdf", 3, "phí", "36 | Lệ phí cấp chứng chỉ | Bộ Tài chính", kind="table")
    table.metadata["type"] = "table"

    def llm(prompt: str) -> str:
        n = len(prompt) % 7
        return json.dumps(
            {
                "question": f"Cơ quan nào quy định mức lệ phí cấp chứng chỉ loại {n}{'x' * n}?",
                "answer_keys": [["Bộ Tài chính"]],
            }
        )

    rows = dq.draft([table] + corpus(), llm, {"table": 1}, 1)
    assert [r["category"] for r in rows] == ["table"]


def test_unanswerable_questions_get_warnings_when_the_corpus_may_answer_them():
    chunks = [*corpus(), make_chunk("a.pdf", 50, "thuế thu nhập cá nhân", "10%")]
    rows = dq.draft(chunks, StubLLM(), {"near_domain": 30, "off_topic": 5}, 1)

    near = [r for r in rows if r["category"] == "near_domain"]
    assert len(near) == len(dq.NEAR_DOMAIN_SEEDS) and all(not r["answerable"] for r in near)
    flagged = [r for r in near if r["_review"]["warnings"]]
    assert [r["question"] for r in flagged] and all(
        "thuế thu nhập cá nhân" in r["question"].lower() for r in flagged
    )
    off = [r for r in rows if r["category"] == "off_topic"]
    assert len(off) == 5 and all(r["difficulty"] == "easy" and not r["_review"]["warnings"] for r in off)


def test_every_draft_gets_a_provisional_split_and_stays_unverified():
    rows, _ = run({"factoid": 4, "off_topic": 6})
    assert {r["split"] for r in rows} == {"dev", "test"}
    assert all(r["verified"] is False for r in rows)


def test_draft_resumes_from_existing_rows_without_repeating_ids_or_chunks():
    first, _ = run({"factoid": 2})
    again = dq.draft(corpus(), StubLLM(), {"factoid": 5}, 1, existing=[dict(r) for r in first])

    assert len(again) == 5
    assert [r["id"] for r in again[:2]] == [r["id"] for r in first]
    assert len({r["id"] for r in again}) == 5
    assert len({tuple(map(tuple, r["_review"]["chunk_keys"])) for r in again}) == 5  # mỗi chunk dùng một lần


def test_draft_is_reproducible_for_the_same_seed():
    a, _ = run({"factoid": 4}, seed=7)
    b, _ = run({"factoid": 4}, seed=7)
    assert [r["question"] for r in a] == [r["question"] for r in b]


# ---------------------------------------------------------------- chia dev/test


def rows_for_split(n_a: int, n_b: int):
    return [{"id": f"a-{i}", "category": "a"} for i in range(n_a)] + [
        {"id": f"b-{i}", "category": "b"} for i in range(n_b)
    ]


def test_assign_splits_is_balanced_per_category_and_deterministic():
    rows = rows_for_split(11, 6)
    result = assign_splits(rows)
    assert result == assign_splits(list(reversed(rows)))  # không phụ thuộc thứ tự dòng
    for category in ("a", "b"):
        dev = sum(1 for r in rows if r["category"] == category and result[r["id"]] == "dev")
        total = sum(1 for r in rows if r["category"] == category)
        assert abs(dev - total / 2) <= 1


def test_assign_splits_never_moves_rows_that_already_have_a_split():
    rows = rows_for_split(6, 0)
    rows[0]["split"] = "test"
    rows[1]["split"] = "test"
    result = assign_splits(rows)
    assert result["a-0"] == "test" and result["a-1"] == "test"
    assert sum(v == "dev" for v in result.values()) == 3  # các câu mới được dồn về dev cho cân bằng


def test_assign_splits_changes_with_the_seed():
    rows = rows_for_split(20, 0)
    assert assign_splits(rows, seed=1) != assign_splits(rows, seed=2)


def test_every_accepted_row_is_checkpointed_immediately():
    saved = []
    rows = dq.draft(corpus(), StubLLM(), {"factoid": 3}, 1, checkpoint=lambda r: saved.append(len(r)))
    assert saved[:3] == [1, 2, 3] and len(rows) == 3  # lưu sau từng câu, không đợi hết nhóm


def test_llm_is_limited_in_tokens_and_time(monkeypatch):
    from ragbot.config import Settings

    captured = {}

    class Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, prompt):
            return type("R", (), {"content": "{}"})()

    monkeypatch.setattr("langchain_ollama.ChatOllama", Fake)
    assert dq.make_llm(Settings())("x") == "{}"
    assert captured["num_predict"] == dq.MAX_NEW_TOKENS and captured["client_kwargs"]["timeout"] > 0
    assert captured["format"] == "json"


# ---------------------------------------------------------------- đáp án phải ngắn, không nhắc nhầm văn bản


def test_answer_keys_must_be_short_phrases_not_copied_sentences():
    long_key = "Buộc người sử dụng lao động lập hồ sơ đăng ký đầy đủ cho người lao động"
    assert dq.clean_groups([[long_key, "12 ngày"]]) == [["12 ngày"]]  # cụm dài bị bỏ, cụm ngắn giữ lại
    assert dq.clean_groups([[long_key]]) == []  # không còn cụm ngắn nào: loại cả đáp án
    assert (
        dq.clean_groups([["12 ngày"], [long_key]]) == []
    )  # một nhóm hỏng thì loại cả, không hạ yêu cầu âm thầm
    assert dq.short_enough("Chủ tịch Ủy ban nhân dân") and not dq.short_enough("một hai ba bốn năm sáu bảy")


def test_trailing_punctuation_is_stripped_from_answer_keys():
    assert dq.clean_groups([["ở nơi có nhiều lao động."], ["03 tháng;"]]) == [
        ["ở nơi có nhiều lao động"],
        ["03 tháng"],
    ]


def test_foreign_document_numbers_in_the_question_are_detected():
    own = ["283/2026/NĐ-CP"]
    assert dq.foreign_reference("Theo Nghị định 283/2026/NĐ-CP, mức phạt là bao nhiêu?", own) is None
    assert (
        dq.foreign_reference("Theo Nghị định 12/2022/NĐ-CP, mức phạt là bao nhiêu?", own) == "12/2022/NĐ-CP"
    )
    assert dq.foreign_reference("Theo Luật Việc làm, thời hạn là bao nhiêu?", own) is None
    assert dq.foreign_reference("Theo Nghị định 12/2022/NĐ-CP thì sao?", [None]) == "12/2022/NĐ-CP"


def test_question_citing_another_document_is_dropped():
    chunk = make_chunk("a.pdf", 5, "nghỉ phép", "12 ngày")
    chunk.metadata["so_hieu"] = "01/2020/QH14"

    def llm(prompt: str) -> str:
        return json.dumps(
            {
                "question": "Theo Nghị định 99/2021/NĐ-CP, người lao động nghỉ phép mấy ngày trong năm?",
                "answer_keys": [["12 ngày"]],
            }
        )

    assert dq.draft([chunk, *corpus()], llm, {"factoid": 1}, 1) == []


def test_resume_drops_old_drafts_that_break_the_current_rules_and_redrafts_them():
    old = {
        "id": "fa-001",
        "question": "Câu cũ có đáp án quá dài?",
        "category": "factoid",
        "answerable": True,
        "must_include": [["Buộc người sử dụng lao động lập hồ sơ đăng ký đầy đủ cho người lao động"]],
        "_review": {"chunk_keys": []},
        "split": "dev",
        "verified": False,
    }
    logs = []
    rows = dq.draft(corpus(), StubLLM(), {"factoid": 2}, 1, existing=[old], log=logs.append)

    assert len(rows) == 2 and all(r["id"] != "fa-001" or r is not old for r in rows)
    assert "Câu cũ có đáp án quá dài?" not in [r["question"] for r in rows]
    assert any("Bỏ 1 câu nháp cũ" in line for line in logs)


def test_prompt_asks_for_short_answers_single_fact_and_no_foreign_laws():
    prompt = dq.SINGLE_PROMPT.format(style="", title="Luật A", doc_hint="", chunk="x")
    assert "tối đa 6 từ" in prompt and "MỘT thông tin duy nhất" in prompt and "Luật A" in prompt
    assert "tối đa 6 từ" in dq.MULTI_PROMPT
