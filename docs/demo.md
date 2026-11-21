# Quay ảnh demo (GIF)

Mục tiêu: một ảnh GIF ngắn (15–30 giây, dưới khoảng 8 MB) đặt ở đầu README, cho người xem thấy ba hành vi
quan trọng nhất của chatbot.

## 1. Kịch bản: ba câu hỏi

Mở giao diện web ở http://localhost:8000 (cấu hình bộ văn bản pháp luật, xem mục "Dùng bộ văn bản pháp luật"
trong README) và hỏi lần lượt:

| # | Câu hỏi | Hành vi cần thể hiện | Kết quả đã kiểm tra (2026-11-21, chạy 2 lượt cho kết quả giống nhau) |
|---|---|---|---|
| 1 | `Người lao động làm việc đủ 12 tháng thì được nghỉ hằng năm bao nhiêu ngày?` | Câu thường: trả lời đúng, trích dẫn đến Điều/Khoản | "… được nghỉ hằng năm 12 ngày làm việc", nguồn: Bộ luật Lao động 2019, Điều 113, Khoản 1–3 (trang 39) |
| 2 | `hop dong lao dong xac dinh thoi han toi da bao nhieu thang` | Câu gõ không dấu vẫn hiểu | "… không quá 36 tháng"; danh sách nguồn có 5 mục, trong đó có Bộ luật Lao động, Điều 20 |
| 3 | `Thủ tục đăng ký kết hôn cần những giấy tờ gì?` | Câu ngoài phạm vi: từ chối thay vì bịa | "Tôi không tìm thấy thông tin này trong tài liệu." |

Lưu ý quan trọng:
- Mô hình ngôn ngữ nhỏ có thể cho kết quả khác nhau giữa các lần chạy, và việc viết lại câu không dấu đôi khi
  làm sai nghĩa. **Hãy chạy thử cả ba câu trước khi quay.** Nếu câu 2 trả lời sai hoặc bị từ chối, thử một câu
  không dấu khác hoặc quay lại; không chỉnh sửa kết quả trong ảnh.
- Đừng chọn câu về tuổi nghỉ hưu hay các quy định có lộ trình tính toán: mô hình từng trả lời không ổn định.
- Câu đầu tiên sau khi khởi động lại máy chủ rất chậm (nạp mô hình, có thể gần một phút). Hỏi trước một câu
  bất kỳ để "làm nóng", rồi mới bắt đầu quay.

## 2. Quay màn hình

Ubuntu/GNOME có sẵn công cụ quay, không cần cài thêm:

1. Nhấn `Ctrl + Shift + Alt + R` để bắt đầu quay, nhấn lại để dừng. File `.webm` nằm trong thư mục `Videos`.
2. Thu nhỏ cửa sổ trình duyệt vào khoảng 900×600 để chữ vẫn đọc được sau khi nén. Chỉ quay cửa sổ trình duyệt,
   đừng để lộ thông tin cá nhân (tab khác, thanh dấu trang, thông báo).

Công cụ thay thế: [Peek](https://github.com/phw/peek) (chỉ chạy trên X11, không chạy trên Wayland),
[Kooha](https://github.com/SeaDve/Kooha) hoặc OBS Studio.

## 3. Đổi sang GIF

```bash
sudo apt install ffmpeg gifsicle      # nếu chưa có
ffmpeg -i demo.webm \
  -vf "fps=8,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
  docs/demo.gif
gifsicle -O3 --lossy=60 docs/demo.gif -o docs/demo.gif   # giảm dung lượng
ls -lh docs/demo.gif                                      # mục tiêu < 8 MB
```

Nếu file còn nặng: giảm `fps` xuống 6, giảm `scale` xuống 800, hoặc cắt bớt đoạn chờ ở đầu/cuối
(`ffmpeg -ss 2 -t 25 -i demo.webm ...`).

## 4. Đặt vào README

Lưu file là `docs/demo.gif`, rồi trong [README.md](../README.md) mục "Demo" thay dòng chữ nghiêng bằng:

```markdown
![Demo: câu thường, câu không dấu, câu ngoài phạm vi bị từ chối](docs/demo.gif)
```
