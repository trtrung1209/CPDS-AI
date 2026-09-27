import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

def find_mic():
    print("=== DANH SÁCH THIẾT BỊ ÂM THANH ===")
    print(sd.query_devices())
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
            return i, int(dev['default_samplerate'])
    return None, 44100

idx, sr = find_mic()
if idx is None:
    print("\n[CẢNH BÁO] Không tìm thấy USB Mic! Sẽ dùng Mic tích hợp của Laptop.")
else:
    print(f"\n[OK] Đã chọn USB Mic tại index {idx} (Tần số: {sr}Hz)")

print("\nChuẩn bị thu âm 3 giây. HÃY MỞ TIẾNG KHÓC GHÉ SÁT VÀO MIC...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1... BẮT ĐẦU THU ÂM!")

duration = 3.0
recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=idx)
sd.wait()

max_val = np.max(np.abs(recording))
print(f"\n[KẾT QUẢ] Peak amplitude (Âm lượng lớn nhất): {max_val:.4f} (Mức tối đa là 1.0)")

if max_val < 0.05:
    print("-> ❌ LỖI: Mic thu quá bé (gần như bị điếc).")
else:
    print("-> ✅ TỐT: Âm lượng thu được đủ to!")

write("laptop_mic_test.wav", sr, recording)
print("Đã lưu thành công file: laptop_mic_test.wav")
print("-> BẠN HÃY MỞ THƯ MỤC LÊN VÀ CLICK ĐÚP VÀO FILE NÀY ĐỂ NGHE THỬ NHÉ!")
