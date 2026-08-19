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
- `tests/`: Chứa các bài kiểm thử tự động bằng `pytest`; không cần model thật để kiểm tra logic.
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
python3 -m pytest
```
Hệ thống sẽ tự quét thư mục `tests/`. GitHub Actions cũng chạy cùng lệnh này trên mỗi push/PR.

## 5. Train và export model trên Kaggle
Audio: gắn dataset vào notebook Kaggle rồi cập nhật `DATASET_DIR` trong `01_audio_training.ipynb`. Dataset phải có cấu trúc `train/noise`, `train/cry` (và tùy chọn `val/noise`, `val/cry`).

Vision: tạo Kaggle Secret `ROBOFLOW_API_KEY`, cấp quyền cho notebook và cập nhật workspace/project/version trong `02_vision_training.ipynb`. Không ghi API key vào source code. Notebook lấy `results.save_dir` từ Ultralytics nên không phụ thuộc cấu trúc `runs/` của từng phiên bản.

Tải các artifact sau khi train về thư mục `data/models/` (thư mục này không được commit):

- `best.onnx` (và `vision_metadata.json`) từ `/kaggle/working/artifacts/` của notebook vision;
- `audio_model.onnx` và `audio_labels.json` từ notebook audio.

## 6. Chạy suy luận ONNX
Suy luận dùng model thật, không còn kết quả mock. Kết quả được lưu trong `runs/run1`, `runs/run2`...:

```bash
python3 -m src.inference.run_inference \
  --image sample.jpg --audio sample.wav \
  --vision-model data/models/best.onnx \
  --audio-model data/models/audio_model.onnx \
  --audio-labels data/models/audio_labels.json
```

Để kiểm tra từng model và lưu output trực quan:

```bash
python3 -m src.inference.verify_vision --model data/models/best.onnx --image sample.jpg
python3 -m src.inference.verify_audio --model data/models/audio_model.onnx --audio sample.wav --labels data/models/audio_labels.json
```

## 7. Docker
Để triển khai trơn tru trên Pi 4 (dùng Ubuntu Server), bạn có thể build và chạy Docker:
```bash
docker build -t cpds-inference -f docker/Dockerfile.inference .
docker run --rm -it -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/runs:/app/runs" cpds-inference \
  python3 -m src.inference.run_inference --image /app/data/sample.jpg --audio /app/data/sample.wav
```
