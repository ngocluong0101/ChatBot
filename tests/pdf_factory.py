"""Tạo PDF tối giản (chữ Helvetica ASCII + đường kẻ) để test bộ đọc PDF mà không cần file thật."""


def build_pdf(pages):
    """Mỗi trang là danh sách thao tác: ("text", x, y, chuỗi), ("line", ...) hoặc ("fillrect", x, y, w, h)."""
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    add(None)  # pages placeholder
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for ops in pages:
        stream = ""
        for op in ops:
            if op[0] == "text":
                _, x, y, t = op
                t = t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream += f"BT /F1 11 Tf {x} {y} Td ({t}) Tj ET\n"
            elif op[0] == "fillrect":  # hình chữ nhật tô nền, không viền
                _, x, y, w, h = op
                stream += f"0.8 g {x} {y} {w} {h} re f 0 g\n"
            else:
                _, x1, y1, x2, y2 = op
                stream += f"{x1} {y1} m {x2} {y2} l S\n"
        c = add(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream")
        p = add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {c} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        )
        kids.append(p)
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out = "%PDF-1.4\n"
    offs = []
    for i, body in enumerate(objs, 1):
        offs.append(len(out.encode("latin-1")))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode("latin-1"))
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


def grid(x0, y_top, col_w, row_h, rows, cols, cells):
    ops = []
    for r in range(rows + 1):
        ops.append(("line", x0, y_top - r * row_h, x0 + cols * col_w, y_top - r * row_h))
    for c in range(cols + 1):
        ops.append(("line", x0 + c * col_w, y_top, x0 + c * col_w, y_top - rows * row_h))
    for r in range(rows):
        for c in range(cols):
            ops.append(("text", x0 + c * col_w + 4, y_top - r * row_h - 16, cells[r][c]))
    return ops
