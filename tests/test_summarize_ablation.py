import json
from pathlib import Path

import pytest

from evals import run_ablation as ra
from evals import summarize as sm


def report(label: str, split: str = "test", hit1: float = 0.8, **meta) -> dict:
    return {
        "meta": {
            "label": label,
            "split": split,
            "include_unverified": False,
            "active_threshold": 0.31,
            "dataset": {"total": 40},
            "unsupported_examples": [],
            **meta,
        },
        "retrieval": {
            "n_answerable": 30,
            "hit@1": hit1,
            "hit@1_ci95": [hit1 - 0.1, hit1 + 0.1],
            "mrr": 0.85,
            "context_precision": 0.7,
            "by_category": {"no_diacritics": {"n": 4, "hit@1": 0.5, "hit@5": 0.75, "mrr": 0.6}},
        },
        "generation": {
            "answerable_accuracy": 0.7,
            "answerable_accuracy_ci95": [0.55, 0.85],
            "refusal_accuracy": 0.9,
            "refusal_accuracy_ci95": None,
            "by_category": {"no_diacritics": {"n": 4, "accuracy": 0.25}},
        },
    }


def write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_table_shows_ci_threshold_and_source_file(tmp_path: Path):
    a = write(tmp_path, "final-01-test.json", report("1. Dense"))
    text = sm.render([a])
    assert (
        "| 1. Dense | 0.800 [0.70–0.90] | 0.850 | 0.700 | 0.700 [0.55–0.85] | 0.900 | 0.31 | "
        "`final-01-test.json` |" in text
    )


def test_diacritics_table_reports_the_category_per_configuration(tmp_path: Path):
    paths = [
        write(tmp_path, "a.json", report("tắt rewrite")),
        write(tmp_path, "b.json", report("bật rewrite")),
    ]
    text = sm.render(paths)
    assert "Riêng nhóm `no_diacritics`" in text
    assert "| tắt rewrite | 4 | 0.500 | 0.250 |" in text and "| bật rewrite | 4 | 0.500 | 0.250 |" in text


def test_label_falls_back_to_the_file_name(tmp_path: Path):
    data = report("x")
    data["meta"]["label"] = None
    assert sm.render([write(tmp_path, "run-7.json", data)]).splitlines()[2].startswith("| run-7 |")


def test_warnings_for_non_test_draft_and_mismatched_reports(tmp_path: Path):
    paths = [
        write(tmp_path, "a.json", report("A", split="dev")),
        write(
            tmp_path,
            "b.json",
            {
                **report("B", include_unverified=True),
                "meta": {
                    **report("B", include_unverified=True)["meta"],
                    "dataset": {"total": 99},
                    "unsupported_examples": ["q1", "q2"],
                },
            },
        ),
    ]
    text = sm.render(paths)
    assert "không phải split=test" in text and "khác nhau" in text and "CHƯA duyệt" in text
    assert "B: 2 câu không có chunk nào chứa bằng chứng" in text and "B ⚠ NHÁP" in text


def test_main_reports_missing_files(tmp_path: Path, capsys):
    assert sm.main([str(tmp_path / "khong-co.json")]) == 2
    assert "Không thấy" in capsys.readouterr().err


def test_main_writes_markdown_file(tmp_path: Path):
    out = tmp_path / "t.md"
    assert sm.main([str(write(tmp_path, "a.json", report("A"))), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("| Cấu hình")


# ---------------------------------------------------------------- run_ablation


def test_each_configuration_changes_exactly_one_setting_from_the_previous_one():
    settings = [ra.settings_for(c[2]) for c in ra.CONFIGS]
    changed = [c[3] for c in ra.CONFIGS]
    for previous, current, what in zip(settings, settings[1:], changed[1:], strict=False):
        diff = {k for k in current if current[k] != previous.get(k)} - {"index_dir"}
        assert diff == {what}, f"{what}: khác {diff}"
    assert all(s["llm_model"] == "qwen2.5:7b-instruct" and s["docs_dir"] == "data/corpus" for s in settings)


def test_fixed_and_legal_chunking_use_separate_indexes():
    assert ra.settings_for({"chunking": "fixed"})["index_dir"] == ".index-fixed"
    assert ra.settings_for({"chunking": "legal"})["index_dir"] == ".index-legal"


def test_commands_choose_threshold_on_dev_and_apply_it_to_test(tmp_path: Path):
    name, label, overrides, _ = ra.CONFIGS[3]
    dev, test = ra.build_commands(name, label, overrides, tmp_path / "d.jsonl", tmp_path)

    assert (dev.step, test.step) == ("dev", "test")
    assert (
        dev.argv[dev.argv.index("--split") + 1] == "dev"
        and dev.argv[dev.argv.index("--mode") + 1] == "retrieval"
    )
    assert (
        test.argv[test.argv.index("--split") + 1] == "test"
        and test.argv[test.argv.index("--mode") + 1] == "full"
    )
    assert test.argv[test.argv.index("--threshold-from") + 1] == str(dev.out)
    assert "--threshold-from" not in dev.argv
    assert test.out.name == "final-04-rerank-test.json" and dev.out.name == "dev-04-rerank.json"
    assert "--include-unverified" not in dev.argv + test.argv


def test_dry_run_with_unverified_questions_never_writes_final_files(tmp_path: Path):
    name, label, overrides, _ = ra.CONFIGS[0]
    commands = ra.build_commands(
        name, label, overrides, tmp_path / "d.jsonl", tmp_path, include_unverified=True
    )
    assert all("final" not in c.out.name and c.out.name.startswith("draft-") for c in commands)
    assert all("--include-unverified" in c.argv for c in commands)


def test_judge_model_only_applies_to_the_test_run(tmp_path: Path):
    name, label, overrides, _ = ra.CONFIGS[0]
    dev, test = ra.build_commands(name, label, overrides, tmp_path / "d.jsonl", tmp_path, judge_model="m:1b")
    assert "--judge-model" in test.argv and "--judge-model" not in dev.argv


def test_select_configs_by_number_or_full_name():
    assert [c[0] for c in ra.select_configs("03,04")] == ["03-hybrid", "04-rerank"]
    assert [c[0] for c in ra.select_configs("05-rewrite")] == ["05-rewrite"]
    assert len(ra.select_configs(None)) == 5
    with pytest.raises(SystemExit):
        ra.select_configs("09")


def test_dry_run_prints_commands_without_running_anything(capsys, monkeypatch):
    monkeypatch.setattr(ra.subprocess, "run", lambda *a, **k: pytest.fail("không được chạy lệnh thật"))
    assert ra.main(["--dry-run", "--only", "01"]) == 0
    out = capsys.readouterr().out
    assert "ragbot.evaluate" in out and "--threshold-from" in out and "evals.summarize" in out


def test_limit_requires_include_unverified_and_is_forwarded(tmp_path: Path):
    with pytest.raises(SystemExit, match="include-unverified"):
        ra.main(["--limit", "5", "--dry-run"])
    name, label, overrides, _ = ra.CONFIGS[0]
    commands = ra.build_commands(name, label, overrides, tmp_path / "d.jsonl", tmp_path, True, limit=7)
    assert all(c.argv[c.argv.index("--limit") + 1] == "7" for c in commands)


def test_limited_reports_are_flagged_by_the_summary(tmp_path: Path):
    path = write(tmp_path, "a.json", report("A", limit=10))
    assert "--limit" in sm.render([path])


# ---------------------------------------------------------------- điền bảng vào README


README_TEMPLATE = (
    "đầu\n<!-- ablation:start -->\ncũ\n<!-- ablation:end -->\n"
    "giữa\n<!-- dataset:start -->\ncũ\n<!-- dataset:end -->\ncuối\n"
)


def test_inject_replaces_only_the_marked_block():
    out = sm.inject(README_TEMPLATE, "ablation", "BẢNG MỚI")
    assert "<!-- ablation:start -->\nBẢNG MỚI\n<!-- ablation:end -->" in out
    assert out.startswith("đầu\n") and "giữa" in out and out.count("cũ") == 1  # khối dataset không đổi
    assert sm.inject(out, "ablation", "BẢNG MỚI") == out  # chạy lại ra đúng kết quả cũ (idempotent)


def test_inject_keeps_special_characters_in_the_content():
    out = sm.inject(README_TEMPLATE, "dataset", r"| a \1 | `x` |")
    assert r"| a \1 | `x` |" in out


def test_inject_fails_loudly_when_markers_are_missing():
    with pytest.raises(ValueError, match="ablation"):
        sm.inject("không có dấu nào", "ablation", "x")


def write_dataset(path: Path) -> Path:
    rows = [
        {"id": "a", "category": "factoid", "answerable": True, "split": "dev"},
        {"id": "b", "category": "factoid", "answerable": True, "split": "test"},
        {"id": "c", "category": "factoid", "answerable": True, "split": "test"},
        {"id": "d", "category": "off_topic", "answerable": False, "split": "dev"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_dataset_table_counts_per_category_and_split(tmp_path: Path):
    table = sm.dataset_table(write_dataset(tmp_path / "d.jsonl"))
    assert "| factoid | 1 | 2 | 3 |" in table and "| off_topic | 1 | 0 | 1 |" in table
    assert "| **Tổng** | **2** | **2** | **4** |" in table
    assert "1/4 câu (25%) không có đáp án" in table


def test_main_updates_both_readme_blocks(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(README_TEMPLATE, encoding="utf-8")
    report_path = write(tmp_path, "final-x-test.json", report("X"))
    dataset = write_dataset(tmp_path / "d.jsonl")

    assert sm.main([str(report_path), "--readme", str(readme), "--dataset", str(dataset)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "| X | 0.800 [0.70–0.90]" in text and "| **Tổng** |" in text and "cũ" not in text


def test_the_real_readme_has_the_marker_pairs_the_tool_updates():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for name in ("ablation", "dataset"):
        assert f"<!-- {name}:start -->" in readme and f"<!-- {name}:end -->" in readme
