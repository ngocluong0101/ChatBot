# Nguồn tài liệu (corpus)

Bảng dưới đây là **nguồn dữ liệu duy nhất** về từng văn bản: code đọc nó để lấy *Tên văn bản* dùng trong
trích dẫn (ví dụ "Bộ luật Lao động 2019, Điều 113, Khoản 1 (trang 41)") và số hiệu gắn vào mỗi chunk.
Đổi nội dung bảng thì chỉ mục tự được dựng lại.

Quy ước:
- **File**: đúng tên file PDF trong `data/corpus/` (khuyên dùng tên ASCII ngắn, ví dụ `45-2019-QH14.pdf`).
- **Tên văn bản**: tên ngắn hiển thị cho người dùng, ví dụ `Bộ luật Lao động 2019`, `Nghị định 145/2020/NĐ-CP`.
- **Loại**: `gốc` | `sửa đổi` | `hướng dẫn` | `thay thế`.
- Ngày theo định dạng `dd/mm/yyyy`. Chỉ ghi điều đã đối chiếu với văn bản; chưa chắc thì để trống.
- Mọi văn bản ở đây là văn bản quy phạm pháp luật công khai (không thuộc đối tượng bảo hộ quyền tác giả).
  File PDF **không** được commit vào git (xem `data/corpus/README.md`), chỉ commit bảng này.

| File | Tên văn bản | Số hiệu | Loại | Ngày ban hành | Ngày hiệu lực | Link nguồn | Ghi chú |
|---|---|---|---|---|---|---|---|
| 18_VBHN-VPQH_m_695336.pdf | Bộ luật Lao động 2019 (văn bản hợp nhất) | 18/VBHN-VPQH | hợp nhất | 12/02/2026 | 01/01/2021 | | Văn bản hợp nhất của Văn phòng Quốc hội, gộp Bộ luật Lao động 45/2019/QH14 và các luật sửa đổi (71/2025, 113/2025, 124/2025). Ngày hiệu lực là của Bộ luật gốc; các điều sửa đổi có hiệu lực 01/01/2026 hoặc 01/07/2026. Ngày ban hành là ngày ký văn bản hợp nhất. |
| 145_2020_ND-CP_m_459400.pdf | Nghị định 145/2020/NĐ-CP | 145/2020/NĐ-CP | hướng dẫn | 14/12/2020 | 01/02/2021 | | Hướng dẫn Bộ luật Lao động về điều kiện lao động và quan hệ lao động. |
| 12_2022_ND-CP_m_479312.pdf | Nghị định 12/2022/NĐ-CP | 12/2022/NĐ-CP | gốc | 17/01/2022 | | | Xử phạt vi phạm hành chính trong lĩnh vực lao động, bảo hiểm xã hội, người lao động Việt Nam đi làm việc ở nước ngoài. Hết hiệu lực từ 10/09/2026 theo Điều 66 khoản 4 của NĐ 283/2026. Điều 62 của file này ghi hiệu lực "kể từ ngày 17/01/2022" (trùng ngày ban hành): cần đối chiếu với văn bản gốc nên chưa điền cột hiệu lực. |
| 283_2026_ND-CP_m_642116.pdf | Nghị định 283/2026/NĐ-CP | 283/2026/NĐ-CP | thay thế | 15/07/2026 | 10/09/2026 | | Thay thế NĐ 12/2022 (Điều 66 khoản 4: NĐ 12/2022 "hết hiệu lực thi hành kể từ ngày Nghị định này có hiệu lực"). Cặp 12/2022 và 283/2026 dùng để kiểm tra việc truy hồi nhầm văn bản cũ. |
| 41_2024_QH15_m_557190.pdf | Luật Bảo hiểm xã hội 2024 | 41/2024/QH15 | gốc | 29/06/2024 | 01/07/2025 | | |
| 74_2025_QH15_m_530912.pdf | Luật Việc làm 2025 | 74/2025/QH15 | gốc | 16/06/2025 | 01/01/2026 | | Điều 54: thay thế Luật Việc làm 38/2013/QH13. |
| 84_2015_QH13_m_281961.pdf | Luật An toàn, vệ sinh lao động 2015 | 84/2015/QH13 | gốc | 25/06/2015 | 01/07/2016 | | |
| 91_2025_QH15_m_625628.pdf | Luật Bảo vệ dữ liệu cá nhân 2025 | 91/2025/QH15 | gốc | 26/06/2025 | 01/01/2026 | | |

<!-- Thêm hàng mới ngay dưới hàng cuối cùng, mỗi văn bản một dòng, đủ 8 cột. Cột "Link nguồn" còn trống: điền trang chính thức bạn đã tải từng file. -->
