"""Thông tin về từng văn bản trong corpus (tên, số hiệu, ngày ban hành/hiệu lực, link nguồn).

Nguồn dữ liệu duy nhất là bảng markdown trong `data/SOURCES.md`: vừa để người đọc kiểm chứng
nguồn gốc, vừa để code lấy tên hiển thị dùng trong trích dẫn ("Bộ luật Lao động 2019, Điều 113").
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ragbot.lexical import strip_accents

logger = logging.getLogger(__name__)

# Số hiệu thường nằm ở đầu văn bản: "Luật số: 45/2019/QH14", "Số: 145/2020/NĐ-CP".
SO_HIEU_CONTEXT_RE = re.compile(
    r"(?:Luật|Bộ luật|Pháp lệnh)?\s*số\s*:?\s*(\d{1,4}/\d{4}/[A-ZĐ][\w\-Đđ]*)", re.IGNORECASE
)


@dataclass(frozen=True)
class DocInfo:
    file: str
    title: str  # tên hiển thị trong trích dẫn, ví dụ "Bộ luật Lao động 2019"
    so_hieu: str | None = None
    doc_type: str | None = None  # "gốc" | "sửa đổi" | ...
    issued: str | None = None
    effective: str | None = None
    url: str | None = None


def title_from_filename(filename: str) -> str:
    """ "45_2019_QH14-bo-luat-lao-dong.pdf" -> "45 2019 QH14 bo luat lao dong" (phương án cuối cùng)."""
    stem = Path(filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip() or filename


def guess_so_hieu(text: str) -> str | None:
    """Tìm số hiệu trong phần đầu văn bản; None nếu không chắc."""
    match = SO_HIEU_CONTEXT_RE.search(text)
    return match.group(1) if match else None


def _header_key(cell: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(cell).casefold()).strip()


COLUMN_ALIASES = {
    "file": "file",
    "ten file": "file",
    "ten van ban": "title",
    "ten": "title",
    "so hieu": "so_hieu",
    "loai": "doc_type",
    "ngay ban hanh": "issued",
    "ngay hieu luc": "effective",
    "link nguon": "url",
    "link": "url",
    "nguon": "url",
}


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_sources(text: str) -> dict[str, DocInfo]:
    """Đọc bảng markdown; trả về {tên file: DocInfo}. Dòng không phải bảng (kể cả comment) bị bỏ qua.

    Cột nhận theo tên tiêu đề (không phân biệt dấu/hoa thường) nên thứ tự cột tùy ý; bắt buộc có
    "File" và "Tên văn bản". Hàng thiếu một trong hai bị bỏ qua.
    """
    columns: dict[str, int] = {}
    result: dict[str, DocInfo] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_row(line)
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # dòng phân cách |---|---|
        if not columns:
            mapped = {
                COLUMN_ALIASES[_header_key(c)]: i
                for i, c in enumerate(cells)
                if _header_key(c) in COLUMN_ALIASES
            }
            if "file" in mapped and "title" in mapped:
                columns = mapped
            continue
        row = {name: (cells[i] if i < len(cells) else "") for name, i in columns.items()}
        if not row["file"] or not row["title"]:
            continue
        result[row["file"]] = DocInfo(
            file=row["file"],
            title=row["title"],
            so_hieu=row.get("so_hieu") or None,
            doc_type=row.get("doc_type") or None,
            issued=row.get("issued") or None,
            effective=row.get("effective") or None,
            url=row.get("url") or None,
        )
    return result


def load_sources(path: Path) -> dict[str, DocInfo]:
    if not path.is_file():
        logger.info("Không có %s: tên văn bản sẽ được suy ra từ tên file", path)
        return {}
    return parse_sources(path.read_text(encoding="utf-8"))


def resolve_info(filename: str, sources: dict[str, DocInfo], first_page_text: str = "") -> DocInfo:
    """DocInfo của một file: ưu tiên bảng SOURCES.md, không có thì suy ra (và cảnh báo)."""
    known = sources.get(filename)
    if known is not None:
        return known
    logger.warning("%s chưa có trong SOURCES.md: dùng tên suy ra từ tên file", filename)
    return DocInfo(file=filename, title=title_from_filename(filename), so_hieu=guess_so_hieu(first_page_text))
