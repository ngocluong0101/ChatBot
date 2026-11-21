# data/corpus – nơi đặt tài liệu

Đặt các file PDF văn bản pháp luật vào thư mục này. **PDF không được commit vào git** (`.gitignore` đã chặn
`data/corpus/*.pdf`); ai cần chạy lại thì tải đúng bản theo [../SOURCES.md](../SOURCES.md).

## 1. Chọn đúng loại PDF (quan trọng)

Hệ thống đọc **lớp chữ** của PDF, không OCR. PDF phải có chữ chọn/copy được. Kiểm tra nhanh:

```bash
pdftotext 45-2019-QH14.pdf - | head -20      # phải ra chữ tiếng Việt đọc được
```

Kết quả khi tôi thử tải thật (2026-11-20):

| Nguồn | Kết quả |
|---|---|
| **Công báo** `congbao.chinhphu.vn` → trang văn bản → liên kết PDF (`congbaocdn.chinhphu.vn/...pdf`) | Có lớp chữ. **Nhưng văn bản dài được đăng qua nhiều số Công báo**: PDF NĐ 145/2020 tôi tải chỉ có phần đầu (đến Điều 101/115, dòng cuối ghi "Xem tiếp theo Công báo số 1205 + 1206"). Nếu dùng nguồn này phải kiểm tra file có đủ. |
| `datafiles.chinhphu.vn/cpp/files/vbpq/.../*.signed.pdf` (bản ký số trên Cổng TTĐT Chính phủ) | **Là ảnh scan** (mỗi trang một ảnh, chỉ có chữ ký số). 3/3 file thử đều không có chữ. Đừng dùng. |

Công cụ `ragbot.stats` (bước 4) báo rõ file nào không trích được chữ, và file nào có vẻ bị cắt dở (Điều cuối
không phải điều khoản thi hành).

## 2. Corpus hiện tại và văn bản gợi ý

Đã có 8 văn bản (xem [../SOURCES.md](../SOURCES.md)): Bộ luật Lao động 2019 (bản hợp nhất 18/VBHN-VPQH),
NĐ 145/2020, NĐ 12/2022 và NĐ 283/2026 (cặp văn bản cũ/mới cùng đề tài xử phạt), Luật BHXH 2024,
Luật Việc làm 2025, Luật An toàn, vệ sinh lao động 2015, Luật Bảo vệ dữ liệu cá nhân 2025.

Nếu muốn thêm (mục tiêu 8–15 văn bản, cần có **bảng** và **văn bản sửa đổi/thay thế**), gợi ý:

- Chuỗi nghị định lương tối thiểu vùng kế tiếp nhau: 90/2019, 38/2022, 74/2024, 293/2025. Có bảng mức lương theo
  vùng, và văn bản mới thay văn bản cũ nên rất hợp để kiểm tra việc truy hồi nhầm văn bản hết hiệu lực.
  (Cần kiểm tra xem đã có văn bản mới hơn 293/2025 hay chưa.)
- Nghị định hướng dẫn Luật Bảo hiểm xã hội 2024, Luật An ninh mạng 2018, Luật An toàn thông tin mạng 2015.

Chỉ dùng văn bản quy phạm pháp luật công khai từ cổng chính thức. Không đưa vào tài liệu có bản quyền.

## 3. Đặt file và ghi nguồn

1. Tải PDF, đổi tên ASCII ngắn (ví dụ `74-2024-ND-CP.pdf`), bỏ vào thư mục này.
2. Thêm một hàng cho file đó trong [../SOURCES.md](../SOURCES.md): tên file phải **khớp từng ký tự**.
   Thiếu hàng thì hệ thống vẫn chạy nhưng dùng tên suy ra từ tên file và báo cảnh báo.

## 4. Kiểm tra trước khi dựng chỉ mục

```bash
export RAG_DOCS_DIR=data/corpus RAG_PDF_LOADER=pdfplumber RAG_CHUNKING=legal
python -m ragbot.stats --out evals/reports/final-corpus-stats.json
```

Đọc bảng kết quả: mỗi văn bản phải có số Điều hợp lý, không có ghi chú "thiếu Điều" (trừ Điều đã bị bãi bỏ),
"KHÔNG nhận ra cấu trúc Điều" hay "KHÔNG trích được chữ". Nếu có, đối chiếu với PDF gốc trước khi đi tiếp.

## 5. Chạy chatbot trên corpus này

```bash
export RAG_DOCS_DIR=data/corpus RAG_PDF_LOADER=pdfplumber RAG_CHUNKING=legal
export RAG_INDEX_DIR=.index-corpus        # tách khỏi .index của bộ papers/ cũ để không bị dựng lại khi đổi qua lại
python chatbot.py
```

Lần đầu dựng chỉ mục mất vài phút (vài trăm chunk; trên CPU khoảng 0,4 giây/chunk, nhanh hơn nhiều với GPU).
