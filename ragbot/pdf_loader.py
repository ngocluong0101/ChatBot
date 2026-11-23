"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Đọc PDF bằng pdfplumber: giữ thứ tự dòng, nhận diện bảng có đường kẻ, bỏ đầu/chân trang lặp.

Mỗi trang là một Document. Bảng nằm đúng vị trí trong dòng chảy văn bản, bọc giữa hai dòng đánh dấu
`TABLE_START` / `TABLE_END` để bước chia chunk tách thành chunk riêng.
Bảng không có đường kẻ không được nhận diện.
"""

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TABLE_START = "[[BẢNG]]"
TABLE_END = "[[/BẢNG]]"
TABLE_BLOCK_RE = re.compile(re.escape(TABLE_START) + r"\n(.*?)\n" + re.escape(TABLE_END), re.DOTALL)

# Số dòng đầu/cuối mỗi trang được xét là "đầu trang/chân trang".
EDGE_LINES = 3
# Dòng đầu/chân trang lặp ở ít nhất tỉ lệ này số trang (và tối thiểu 3 trang) thì bị bỏ.
RUNNING_LINE_RATIO = 0.4
PAGE_NUMBER_RE = re.compile(
    r"^(?:[-–—]\s*)?(?:trang\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*[-–—])?$", re.IGNORECASE
)


def table_to_markdown(rows: Sequence[Sequence[str | None]]) -> str | None:
    """Bảng (danh sách hàng) -> markdown. Hàng đầu làm tiêu đề. None nếu bảng rỗng/quá nhỏ."""
    cleaned = [[re.sub(r"\s+", " ", (cell or "").replace("|", "/")).strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if len(cleaned) < 2:
        return None
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    # Bảng thật có từ hai hàng và hai ô có chữ; chữ xếp lệch từng ô là đoạn văn bị nhận nhầm vì nền
    # tô màu/khung.
    if sum(sum(bool(cell) for cell in row) >= 2 for row in cleaned) < 2:
        return None
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    if len(keep) < 2:
        return None
    cleaned = [[row[i] for i in keep] for row in cleaned]
    header, body = cleaned[0], cleaned[1:]
    if re.fullmatch(r"\d+", header[0]):
        # Hàng đầu bắt đầu bằng số thứ tự: bảng không có hàng tiêu đề, nên để tiêu đề trống.
        header, body = [""] * len(header), cleaned
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in keep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _signature(line: str) -> str:
    """Dòng đầu/chân trang chỉ khác nhau ở số trang ("Trang 3/20") nên thay chữ số bằng '#'."""
    return re.sub(r"\d+", "#", line).casefold()


def strip_running_lines(pages: list[list[str]]) -> list[list[str]]:
    """Bỏ số trang đứng một mình và các dòng đầu/chân trang lặp lại ở nhiều trang."""
    if not pages:
        return pages
    counts: Counter[str] = Counter()
    for items in pages:
        edge = {_signature(t) for t in (*items[:EDGE_LINES], *items[-EDGE_LINES:]) if TABLE_START not in t}
        counts.update(edge)
    min_pages = max(3, int(RUNNING_LINE_RATIO * len(pages)))
    running = {sig for sig, n in counts.items() if n >= min_pages}

    cleaned: list[list[str]] = []
    for items in pages:
        last = len(items) - 1
        kept = []
        for i, text in enumerate(items):
            at_edge = i < EDGE_LINES or i > last - EDGE_LINES
            if (
                at_edge
                and TABLE_START not in text
                and (PAGE_NUMBER_RE.match(text) or _signature(text) in running)
            ):
                continue
            kept.append(text)
        cleaned.append(kept)
    return cleaned


def _inside(char: dict[str, Any], boxes: Sequence[tuple[float, float, float, float]]) -> bool:
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


FILL_MIN_SIZE = 3.0  # pt; hình tô màu mỏng hơn mức này là viền bảng kiểu Word, không phải nền


def _without_background_fills(page: Any) -> Any:
    """Bỏ hình chữ nhật tô nền không viền (Công báo tô nền đoạn sửa đổi, pdfplumber coi cạnh của chúng là
    đường kẻ bảng). Viền bảng vẽ bằng hình tô mỏng vẫn giữ."""

    def keep(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "rect" or obj.get("stroke") or not obj.get("fill"):
            return True
        return min(obj["width"], obj["height"]) <= FILL_MIN_SIZE

    return page.filter(keep)


def extract_page_items(page: Any) -> list[str]:
    """Các phần tử của trang từ trên xuống: mỗi dòng chữ là một phần tử, mỗi bảng là một phần tử."""
    tables = _without_background_fills(page).find_tables()
    table_items: list[tuple[float, str]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for table in tables:
        markdown = table_to_markdown(table.extract())
        if markdown is None:
            continue
        boxes.append(table.bbox)
        table_items.append((table.bbox[1], f"{TABLE_START}\n{markdown}\n{TABLE_END}"))

    body = (
        page.filter(lambda obj: obj.get("object_type") != "char" or not _inside(obj, boxes))
        if boxes
        else page
    )
    line_items = [
        (float(line["top"]), _normalize_line(line["text"]))
        for line in body.extract_text_lines(strip=True, return_chars=False)
    ]
    ordered = sorted([*line_items, *table_items], key=lambda item: item[0])
    return [text for _, text in ordered if text]


def load_pdf(path: Path) -> list[Document]:
    """Một Document mỗi trang có nội dung; metadata {source, page} như bộ đọc Unstructured."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [extract_page_items(page) for page in pdf.pages]
    cleaned = strip_running_lines(pages)
    docs: list[Document] = []
    for number, items in enumerate(cleaned, start=1):
        if not items:
            continue
        docs.append(Document(page_content="\n".join(items), metadata={"source": path.name, "page": number}))
    if not docs:
        # PDF scan (ảnh) không có text layer; pdfplumber không OCR.
        logger.warning("Không trích được văn bản từ %s (PDF scan? cần OCR)", path.name)
    return docs


def strip_table_markers(text: str) -> str:
    """Bỏ dòng đánh dấu bảng, giữ nội dung bảng như văn bản thường (dùng cho chunk cố định theo độ dài)."""
    return text.replace(f"{TABLE_START}\n", "").replace(f"\n{TABLE_END}", "")
