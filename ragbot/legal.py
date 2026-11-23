"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>

Chia văn bản pháp luật Việt Nam thành chunk theo Chương / Mục / Điều / Khoản / Phụ lục.

Quy trình cho MỘT văn bản (gộp mọi trang vì một Điều thường kéo dài qua nhiều trang):
  1. `tokenize_pages`: tách dòng văn bản và khối bảng, nhớ số trang.
  2. `parse_units`: nhận diện tiêu đề Chương/Mục/Điều/Phụ lục, gom nội dung vào từng đơn vị.
  3. `chunk_unit`: mỗi Điều một chunk; Điều quá `chunk_size` thì tách theo Khoản (rồi điểm a), b), rồi
     độ dài). Bảng thành chunk riêng. Mọi chunk lặp lại tên văn bản + Chương + tiêu đề Điều.

Văn bản không có cấu trúc "Điều" trả về None để bên gọi rơi về cách chia cũ.

Giới hạn: bảng không có đường kẻ không được nhận diện; bảng nhiều trang thành nhiều chunk; văn bản sửa đổi
trích nguyên văn Điều cũ trong ngoặc kép có thể bị nhận sai ranh giới Khoản.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from itertools import groupby

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.config import Settings
from ragbot.corpus import DocInfo
from ragbot.pdf_loader import TABLE_BLOCK_RE

logger = logging.getLogger(__name__)

# Văn bản có ít Điều hơn mức này không được coi là văn bản có cấu trúc luật.
MIN_DIEU = 3
MIN_BODY_BUDGET = 200  # chỗ tối thiểu cho nội dung sau khi trừ phần đầu chunk

SENTENCE_END = (".", ":", ";", "!", "?")
EN_DASH = "–"

DIEU_RE = re.compile(r"^Điều\s+(\d{1,3}[a-zđ]?)\s*\.\s+(\S.*)$", re.IGNORECASE)
KHOAN_RE = re.compile(r"^(\d{1,3})\.\s+\S")
DIEM_RE = re.compile(r"^[a-zđ]\)\s+\S")
DASH_RE = re.compile(r"^[-–•]\s+\S")
# Chương I / CHƯƠNG II. TIÊU ĐỀ / Mục 2. Tiêu đề / PHỤ LỤC I
CHUONG_RE = re.compile(r"^(?:CHƯƠNG|Chương)\s+([IVXLCDM]+|\d{1,2})\b\s*(?:[.:\-–—]\s*)?(.*)$")
MUC_RE = re.compile(r"^(?:MỤC|Mục)\s+(\d{1,2}|[IVXLC]+)\b\s*(?:[.:\-–—]\s*)?(.*)$")
PHU_LUC_RE = re.compile(r"^(?:PHỤ LỤC|Phụ lục)\s+([IVXLC]+|\d{1,2}|[A-Z])\b\s*(?:[.:\-–—]\s*)?(.*)$")
# Phần kết của văn bản (nơi nhận, chữ ký, câu "đã được Quốc hội thông qua") không thuộc Điều cuối.
CLOSING_RE = re.compile(
    r"^(?:Nơi nhận\s*:|\(Xem tiếp"
    r"|(?:Luật|Bộ luật|Pháp lệnh|Nghị định|Nghị quyết) này (?:đã )?được (?:Quốc hội|Chính phủ|Ủy ban)"
    r"|TM\.|KT\.|TL\.|TUQ\.|CHỦ TỊCH|THỦ TƯỚNG|BỘ TRƯỞNG|XÁC THỰC VĂN BẢN)"
)


@dataclass
class Tok:
    kind: str  # "text" | "table"
    text: str
    page: int


@dataclass
class Unit:
    """Một đoạn liền mạch của văn bản: phần mở đầu, một Điều, hoặc một Phụ lục."""

    kind: str  # "preamble" | "intro" | "dieu" | "phuluc"
    number: str | None = None
    title: str | None = None
    chuong: str | None = None
    muc: str | None = None
    page: int = 0  # trang của dòng tiêu đề
    tokens: list[Tok] = field(default_factory=list)
    closed: bool = False  # đã gặp phần kết văn bản: bỏ qua phần còn lại cho tới tiêu đề kế tiếp

    @property
    def heading(self) -> str | None:
        if self.kind == "dieu":
            return f"Điều {self.number}. {self.title}"
        if self.kind == "phuluc":
            return f"Phụ lục {self.number}" + (f". {self.title}" if self.title else "")
        return None


@dataclass
class Para:
    text: str
    page: int
    khoan: str | None
    is_table: bool = False


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokenize_pages(pages: list[Document]) -> list[Tok]:
    tokens: list[Tok] = []
    for doc in pages:
        page = doc.metadata.get("page") or 0
        parts = TABLE_BLOCK_RE.split(doc.page_content)  # [văn bản, bảng, văn bản, bảng, ...]
        for index, part in enumerate(parts):
            if index % 2 == 1:
                tokens.append(Tok("table", _nfc(part).strip(), page))
                continue
            for line in part.splitlines():
                line = re.sub(r"\s+", " ", _nfc(line)).strip()
                if line:
                    tokens.append(Tok("text", line, page))
    return tokens


def _is_marker(line: str) -> bool:
    return bool(KHOAN_RE.match(line) or DIEM_RE.match(line) or DASH_RE.match(line))


def _is_mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.7


def _dieu_key(number: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([a-zđ]?)", number.casefold())
    assert match is not None
    return int(match.group(1)), match.group(2)


def _heading_text(prefix: str, number: str, rest: str) -> str:
    return f"{prefix} {number}" + (f". {rest}" if rest else "")


def _valid_sub_heading(rest: str, line: str) -> bool:
    """Chặn dòng văn xuôi tình cờ bắt đầu bằng "Mục 2 của ..." / "Chương trình ..."."""
    return not rest or rest[0].isupper() or rest[0].isdigit() or _is_mostly_upper(line)


def parse_units(tokens: list[Tok]) -> list[Unit]:
    chuong: str | None = None
    muc: str | None = None
    units: list[Unit] = [Unit("preamble")]
    cur = units[0]
    last_key: tuple[int, str] | None = None
    in_appendix = False
    i = 0

    def start(unit: Unit) -> Unit:
        units.append(unit)
        return unit

    def take_title(rest: str, at: int) -> tuple[str, int]:
        """Tiêu đề Chương/Mục/Phụ lục có thể nằm ở dòng kế tiếp và dài vài dòng IN HOA."""
        parts = [rest] if rest else []
        for _ in range(4):
            nxt = tokens[at + 1] if at + 1 < len(tokens) else None
            if nxt is None or nxt.kind != "text" or len(nxt.text) > 150 or not _is_mostly_upper(nxt.text):
                break
            if any(r.match(nxt.text) for r in (DIEU_RE, CHUONG_RE, MUC_RE, PHU_LUC_RE)) or _is_marker(
                nxt.text
            ):
                break
            parts.append(nxt.text)
            at += 1
        return " ".join(parts), at

    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "table":
            if not cur.closed:
                cur.tokens.append(tok)
            i += 1
            continue
        line = tok.text

        if (m := PHU_LUC_RE.match(line)) and _valid_sub_heading(m.group(2), line):
            title, i = take_title(m.group(2).strip(), i)
            # Phụ lục đứng ngoài các Chương nên không mang nhãn Chương/Mục của phần trước.
            cur = start(Unit("phuluc", m.group(1), title or None, page=tok.page))
            in_appendix = True
            i += 1
            continue

        if not in_appendix:
            if (m := CHUONG_RE.match(line)) and _valid_sub_heading(m.group(2), line):
                title, i = take_title(m.group(2).strip(), i)
                chuong, muc = _heading_text("Chương", m.group(1), title), None
                cur = start(Unit("intro", chuong=chuong))
                i += 1
                continue
            if (m := MUC_RE.match(line)) and _valid_sub_heading(m.group(2), line):
                title, i = take_title(m.group(2).strip(), i)
                muc = _heading_text("Mục", m.group(1), title)
                cur = start(Unit("intro", chuong=chuong, muc=muc))
                i += 1
                continue

            if m := DIEU_RE.match(line):
                key = _dieu_key(m.group(1))
                dieu_units = [u for u in units if u.kind == "dieu"]
                accept = last_key is None or key > last_key
                if not accept and key == (1, "") and _looks_like_toc(dieu_units):
                    # Mục lục liệt kê "Điều 1..N" trước nội dung thật: bỏ mục lục, đánh số lại từ đầu.
                    units[:] = [u for u in units if u.kind != "dieu"]
                    accept = True
                if accept:
                    title = m.group(2).strip()
                    for _ in range(4):  # tiêu đề Điều bị xuống dòng (Điều sửa đổi luật khác có thể rất dài)
                        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                        if nxt is None or not _continues_dieu_title(nxt):
                            break
                        title = f"{title} {nxt.text}"
                        i += 1
                    cur = start(Unit("dieu", m.group(1), title, chuong, muc, page=tok.page))
                    last_key = key
                    i += 1
                    continue

        if cur.kind == "dieu" and not cur.closed and CLOSING_RE.match(line):
            cur.closed = True
        if not cur.closed:
            cur.tokens.append(tok)
        i += 1

    return [u for u in units if u.tokens or u.kind in ("dieu", "phuluc")]


def _continues_dieu_title(tok: Tok) -> bool:
    """Dòng sau "Điều N. ..." là phần tiếp của tiêu đề.

    Bắt đầu bằng chữ thường hoặc chữ số, không phải dấu hiệu Khoản/điểm, không kết thúc bằng dấu câu.
    """
    text = tok.text
    return (
        tok.kind == "text"
        and (text[:1].islower() or text[:1].isdigit())
        and not text.endswith(SENTENCE_END)
        and not _is_marker(text)
    )


def _looks_like_toc(dieu_units: list[Unit]) -> bool:
    if len(dieu_units) < 2:
        return False
    body_chars = sum(len(t.text) for u in dieu_units for t in u.tokens)
    return body_chars < 40 * len(dieu_units)


def build_paragraphs(unit: Unit) -> list[Para]:
    """Nối các dòng bị ngắt cứng của PDF thành đoạn; mỗi Khoản/điểm/gạch đầu dòng mở đoạn mới."""
    paras: list[Para] = []
    expected = 1
    khoan: str | None = None
    numbered = unit.kind == "dieu"
    for tok in unit.tokens:
        if tok.kind == "table":
            paras.append(Para(tok.text, tok.page, khoan, is_table=True))
            continue
        line = tok.text
        starts_khoan = numbered and (m := KHOAN_RE.match(line)) is not None and int(m.group(1)) == expected
        if starts_khoan:
            khoan = str(expected)
            expected += 1
            paras.append(Para(line, tok.page, khoan))
            continue
        prev = paras[-1] if paras else None
        new_paragraph = (
            prev is None
            or prev.is_table
            or _is_marker(line)
            or (prev.text.endswith(SENTENCE_END) and not line[:1].islower())
        )
        if new_paragraph:
            paras.append(Para(line, tok.page, khoan))
        else:
            prev.text = f"{prev.text} {line}"
    return paras


def _header(info: DocInfo, unit: Unit) -> str:
    title = info.title
    if info.so_hieu and info.so_hieu not in title:
        title = f"{title} ({info.so_hieu})"
    lines = [title]
    context = "; ".join(x for x in (unit.chuong, unit.muc) if x)
    if context:
        lines.append(context)
    if unit.heading:
        lines.append(unit.heading)
    return "\n".join(lines)


def _split_table(markdown: str, limit: int) -> list[str]:
    lines = markdown.splitlines()
    head, rows = lines[:2], lines[2:]
    pieces: list[list[str]] = [[]]
    size = len("\n".join(head))
    for row in rows:
        if pieces[-1] and size + len(row) + 1 > limit:
            pieces.append([])
            size = len("\n".join(head))
        pieces[-1].append(row)
        size += len(row) + 1
    return ["\n".join([*head, *piece]) for piece in pieces if piece] or [markdown]


def _khoan_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    return labels[0] if labels[0] == labels[-1] else f"{labels[0]}{EN_DASH}{labels[-1]}"


def _group_by_khoan(paras: list[Para]) -> list[tuple[str | None, list[Para]]]:
    groups: list[tuple[str | None, list[Para]]] = []
    for para in paras:
        if groups and groups[-1][0] == para.khoan:
            groups[-1][1].append(para)
        else:
            groups.append((para.khoan, [para]))
    return groups


def chunk_unit(unit: Unit, info: DocInfo, source: str, settings: Settings) -> list[Document]:
    header = _header(info, unit)
    limit = max(MIN_BODY_BUDGET, settings.chunk_size - len(header) - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=limit, chunk_overlap=min(settings.chunk_overlap, 100)
    )
    base: dict[str, object] = {"source": source, "doc_title": info.title}
    if info.so_hieu:
        base["so_hieu"] = info.so_hieu
    if unit.chuong:
        base["chuong"] = unit.chuong
    if unit.muc:
        base["muc"] = unit.muc
    if unit.kind == "dieu":
        base |= {"dieu": unit.number, "dieu_title": unit.title}
    elif unit.kind == "phuluc":
        base["phu_luc"] = unit.number
    section = unit.heading or unit.muc or unit.chuong or "Phần mở đầu"
    text_type = "text" if unit.kind in ("dieu", "phuluc") else "preamble"

    chunks: list[Document] = []

    def emit(body: str, page: int, khoan: str | None, kind: str) -> None:
        meta = {**base, "page": page, "type": kind, "section": section}
        if khoan:
            meta["khoan"] = khoan
        chunks.append(Document(page_content=f"{header}\n{body}" if body else header, metadata=meta))

    # Các nhóm Khoản liên tiếp được gom vào cùng một chunk cho tới khi đầy; bảng luôn tách riêng.
    pending: list[tuple[str | None, str, int]] = []  # (khoản, nội dung, trang)

    def flush() -> None:
        if pending:
            labels = [k for k, _, _ in pending if k]
            emit("\n".join(text for _, text, _ in pending), pending[0][2], _khoan_label(labels), text_type)
            pending.clear()

    for khoan, group in _group_by_khoan(build_paragraphs(unit)):
        for is_table, run_iter in groupby(group, key=lambda p: p.is_table):
            run = list(run_iter)
            if is_table:
                flush()
                for para in run:
                    for piece in _split_table(para.text, limit):
                        emit(piece, para.page, khoan, "table")
                continue
            text = "\n".join(p.text for p in run)
            if len(text) > limit:
                flush()
                for piece, page in _split_long(run, limit, splitter):
                    emit(piece, page, khoan, text_type)
                continue
            if pending and sum(len(t) + 1 for _, t, _ in pending) + len(text) + 1 > limit:
                flush()
            pending.append((khoan, text, run[0].page))
    flush()
    if not chunks and unit.kind in ("dieu", "phuluc"):
        # Điều chỉ có tiêu đề (ví dụ "được bãi bỏ") vẫn phải truy hồi được theo tiêu đề.
        emit("", unit.page, None, text_type)
    return chunks


def _split_long(
    group: list[Para], limit: int, splitter: RecursiveCharacterTextSplitter
) -> list[tuple[str, int]]:
    """Khoản dài hơn một chunk: gom theo điểm a), b)...; điểm vẫn quá dài thì cắt theo độ dài."""
    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    current_page = group[0].page
    size = 0
    for para in group:
        parts = [para.text] if len(para.text) <= limit else splitter.split_text(para.text)
        for part in parts:
            if current and size + len(part) + 1 > limit:
                pieces.append(("\n".join(current), current_page))
                current, size, current_page = [], 0, para.page
            current.append(part)
            size += len(part) + 1
    if current:
        pieces.append(("\n".join(current), current_page))
    return pieces


def split_legal(pages: list[Document], info: DocInfo, settings: Settings) -> list[Document] | None:
    """Chunk của một văn bản; None nếu không phải văn bản có cấu trúc "Điều"."""
    if not pages:
        return None
    source = pages[0].metadata.get("source", info.file)
    units = parse_units(tokenize_pages(pages))
    if sum(u.kind == "dieu" for u in units) < MIN_DIEU:
        return None
    chunks: list[Document] = []
    for unit in units:
        chunks.extend(chunk_unit(unit, info, source, settings))
    logger.debug("%s: %d chunk từ %d đơn vị", source, len(chunks), len(units))
    return chunks
