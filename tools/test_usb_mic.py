import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import time
import sys

def main():
    print("=== Raspberry Pi USB Microphone (MI 305) Test ===")
    
    # In ra danh sách tất cả các thiết bị âm thanh đang cắm vào Pi 4
    print("\n[Available Audio Devices]")
    print(sd.query_devices())
    
    print("\n-------------------------------------------------")
    print("Recording for 3 seconds from DEFAULT input device...")
    sample_rate = 16000
    duration = 3.0
    
    try:
        # sd.rec bắt đầu thu âm ngầm (non-blocking)
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        
        # In ra màn hình cho sinh động
        for i in range(3):
            print(f"🎙️ Recording... {3-i}s remaining")
            time.sleep(1)
            
        sd.wait() # Chờ thu âm xong
        
        # Xử lý âm thanh (chuẩn hóa âm lượng)
        recording = recording.flatten()
        max_amp = np.max(np.abs(recording))
        if max_amp > 0:
            recording = (recording / max_amp) * 0.95  # Chuẩn hóa để nghe rõ hơn
            
        # Chuyển đổi sang định dạng int16 để lưu file wav
        recording_int16 = np.int16(recording * 32767)
        
        filename = "usb_mic_test.wav"
        wavfile.write(filename, sample_rate, recording_int16)
        
        print(f"\n🚀 SUCCESS! Audio saved to '{filename}'.")
        print("Mẹo: Dùng phần mềm truyền file (như WinSCP hoặc scp) chép file này về máy tính để nghe thử xem Mic thu có trong không nhé!")
        
    except Exception as e:
        print(f"\n❌ FAILED to record audio!")
        print(f"Error details: {e}")
        print("\nCách sửa lỗi:")
        print("1. Kiểm tra xem đã cắm cáp USB của Micro MI 305 chưa.")
        print("2. Đảm bảo bạn đã cài thư viện lõi ALSA bằng lệnh: sudo apt install portaudio19-dev")

if __name__ == "__main__":
    main()
