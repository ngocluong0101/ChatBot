"""Giao diện dòng lệnh."""

import logging
import sys

from pydantic import ValidationError

from ragbot.config import get_settings
from ragbot.errors import RagError
from ragbot.pipeline import Turn, build_pipeline

EXIT_COMMANDS = {"exit", "quit"}
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "sentence_transformers", "unstructured", "faiss", "filelock")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Cấu hình không hợp lệ:\n{exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    try:
        pipeline = build_pipeline(settings)
    except RagError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    print("Sẵn sàng. Nhập câu hỏi ('reset' để xóa lịch sử hội thoại, 'exit' hoặc 'quit' để thoát).")
    history: list[Turn] = []
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in EXIT_COMMANDS:
            break
        if not question:
            continue
        if question.lower() == "reset":
            history.clear()
            print("Đã xóa lịch sử hội thoại.")
            continue

        try:
            answer = pipeline.ask(question, history)
            print(answer.render())
            history += [Turn("user", question), Turn("assistant", answer.text)]
            del history[: max(0, len(history) - 2 * settings.max_history_turns)]
        except RagError as exc:
            # Lỗi từng câu hỏi (ví dụ Ollama vừa tắt) không nên làm chết cả phiên chat.
            print(f"Lỗi: {exc}", file=sys.stderr)

    print("Exiting the chatbot. Goodbye!")
    return 0
