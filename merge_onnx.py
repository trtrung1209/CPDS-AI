import onnx
from pathlib import Path

models_dir = Path("data/models")
onnx_file = models_dir / "audio_model.onnx"
data_file = models_dir / "audio_model_fp32.onnx.data"

if not onnx_file.exists():
    print(f"Không tìm thấy {onnx_file}")
elif not data_file.exists():
    print(f"Không tìm thấy {data_file}")
else:
    print("Đang đọc và gộp 2 file...")
    # Load model and explicitly load external data
    model = onnx.load(str(onnx_file), load_external_data=True)
    
    # Save model without external data (embeds everything into one file)
    print("Đang lưu lại thành 1 file duy nhất...")
    onnx.save_model(model, str(onnx_file), save_as_external_data=False)
    
    print("Thành công! Đã gộp toàn bộ vào audio_model.onnx.")
    print("Bây giờ bạn có thể xóa file audio_model_fp32.onnx.data đi cho gọn!")
