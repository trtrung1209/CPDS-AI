import zipfile
import shutil
from pathlib import Path

def setup():
    base_dir = Path("/media/trtrung1209/WORKSPACES/WORKSPACES/01_PROJECTS/2026_NCKH_CPDS-AI")
    zip_path = base_dir / "data" / "kaggle_raw.zip"
    extract_dir = base_dir / "data" / "kaggle_raw"
    cry_dir = base_dir / "data" / "test_audio" / "cry"
    noise_dir = base_dir / "data" / "test_audio" / "noise"
    
    print("--- BẮT ĐẦU CÀI ĐẶT DỮ LIỆU TEST TỪ KAGGLE ---")
    
    # 1. Giải nén nếu thấy file ZIP
    if zip_path.exists():
        print(f"[1] Đang giải nén {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print("[1] Không thấy file ZIP, kiểm tra thư mục giải nén...")

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print("LỖI: Chưa tìm thấy dữ liệu! Hãy đảm bảo bạn đã tải file về và đặt tên là kaggle_raw.zip hoặc giải nén sẵn vào thư mục data/kaggle_raw/")
        return

    # 2. Dọn dẹp tiếng khóc cũ (Donate-a-cry) để nhường chỗ cho hàng xịn
    print("[2] Đang xóa tiếng khóc cũ (Donate-a-cry)...")
    if cry_dir.exists():
        shutil.rmtree(cry_dir)
    cry_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Lọc và gom toàn bộ file âm thanh từ Kaggle
    audio_exts = {".wav", ".mp3", ".ogg", ".flac"}
    audio_files = [p for p in extract_dir.rglob("*") if p.suffix.lower() in audio_exts]
        
    print(f"[3] Tìm thấy {len(audio_files)} file tiếng khóc từ Kaggle. Đang đưa vào lồng test...")
    
    for i, audio_file in enumerate(audio_files):
        dest_path = cry_dir / f"kaggle_cry_{i:04d}{audio_file.suffix}"
        shutil.copy2(audio_file, dest_path)
        
    print(f"\n[HOÀN TẤT] Tuyệt vời! Bộ Test của bạn hiện có:")
    print(f" 👶 Cry (Từ Kaggle): {len(list(cry_dir.glob('*')))} file")
    print(f" 🌧️ Noise (Tiếng ồn): {len(list(noise_dir.glob('*')))} file")
    print("\n👉 BƯỚC TIẾP THEO: Gõ lệnh 'python3 main.py --mode evaluate' để chạy chấm điểm nhé!")

if __name__ == "__main__":
    setup()
