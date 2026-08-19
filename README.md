# CPDS-AI (Child Presence Detection System - AI)

Dự án NCKH: **Hệ thống cảnh báo trẻ em bị bỏ quên trên xe ô tô ứng dụng Trí tuệ nhân tạo (AI) và cơ chế cập nhật firmware từ xa (FOTA).**

## 1. Giới thiệu (Overview)
Hệ thống này sử dụng các luồng học sâu (Deep Learning) để giám sát và phát hiện sự hiện diện của trẻ em trong cabin ô tô nhằm cảnh báo khi trẻ bị bỏ quên. 
Mục tiêu là chạy song song 2 mô hình học máy:
1. **Vision Model:** YOLOv8 (phát hiện Người lớn vs Trẻ em).
2. **Audio Model:** Phân loại âm thanh (Phát hiện tiếng khóc trẻ em, lọc các tạp âm nhiễu từ xe ô tô).

Toàn bộ hệ thống sẽ được đóng gói bằng Docker và triển khai lên thiết bị nhúng (Raspberry Pi 4, sau đó là Orange Pi 5 với NPU).

## 2. Cấu trúc thư mục dự án
- `data/`: Chứa dữ liệu âm thanh và hình ảnh (được bỏ qua bởi gitignore để tránh đẩy file nặng lên GitHub).
- `notebooks/`: Chứa các script Python/Jupyter Notebook để training mô hình (có thể chạy trên Kaggle/Colab).
- `src/`: Mã nguồn chính chứa logic tiền xử lý dữ liệu, định nghĩa mô hình, và suy luận (inference).
  - Kết quả suy luận/training sẽ được lưu tự động thành các thư mục `runs/run1`, `runs/run2`...
- `tests/`: Chứa các bài kiểm thử tự động bằng `pytest` để đảm bảo code sinh ra không bị lỗi.
- `docker/`: Chứa Dockerfile và cấu hình môi trường.
- `.github/workflows/`: Chứa kịch bản CI/CD để GitHub tự chạy test mỗi khi có code mới.

## 3. Cài đặt môi trường (Local / Laptop)

Để chạy thử code trên máy cá nhân trước khi đưa lên Pi:

```bash
# 1. Tạo môi trường ảo (khuyến nghị)
python3 -m venv venv
source venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 4. Chạy kiểm thử tự động (Testing)
Dự án sử dụng `pytest` để kiểm tra các luồng xử lý. Nếu bạn chưa từng dùng pytest, bạn chỉ cần gõ lệnh sau ở thư mục gốc của dự án:
```bash
pytest
```
Hệ thống sẽ tự quét thư mục `tests/` và chạy các kịch bản kiểm tra xem dữ liệu có nạp đúng không, model xuất ra có đúng định dạng không.

## 5. Chạy suy luận (Inference Demo)
Các kết quả sẽ được tự động gom vào các folder `run1`, `run2` để bạn dễ so sánh:
```bash
python src/inference/run_inference.py --audio sample.wav --image sample.jpg
```

## 6. Docker (Sắp ra mắt trên Raspberry Pi 4)
Để triển khai trơn tru trên Pi 4 (dùng Ubuntu Server), bạn có thể build và chạy Docker:
```bash
docker build -t cpds-inference -f docker/Dockerfile.inference .
docker run --rm -it cpds-inference
```
