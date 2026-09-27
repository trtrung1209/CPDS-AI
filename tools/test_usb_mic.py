import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import time

def find_usb_mic():
    """Tìm thiết bị USB Microphone trong danh sách."""
    devices = sd.query_devices()
    print("[Available Audio Devices]")
    print(devices)
    print()
    
    # Tìm thiết bị input có chữ "USB" trong tên
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
            print(f"🎯 Found USB Mic: [{i}] {dev['name']}")
            print(f"   Max input channels: {dev['max_input_channels']}")
            print(f"   Default sample rate: {dev['default_samplerate']}")
            return i, int(dev['default_samplerate'])
    
    return None, None

def main():
    print("=== Raspberry Pi USB Microphone (MI 305) Test ===\n")
    
    device_index, native_sr = find_usb_mic()
    
    if device_index is None:
        print("❌ No USB Microphone found! Make sure the MI 305 is plugged in.")
        return
    
    # Dùng sample rate gốc của thiết bị (thường 44100 hoặc 48000)
    # để tránh lỗi "Invalid sample rate"
    duration = 3.0
    print(f"\n🎙️ Recording {duration}s from device [{device_index}] at {native_sr}Hz...")
    
    try:
        recording = sd.rec(
            int(duration * native_sr),
            samplerate=native_sr,
            channels=1,
            dtype='float32',
            device=device_index  # Chỉ định đúng cái Mic USB, không dùng default
        )
        
        for i in range(int(duration)):
            print(f"   ⏳ {int(duration)-i}s remaining...")
            time.sleep(1)
        sd.wait()
        
        recording = recording.flatten()
        
        # Kiểm tra xem Mic có thực sự thu được âm thanh hay toàn số 0
        max_amp = np.max(np.abs(recording))
        print(f"\n   Peak amplitude: {max_amp:.6f}")
        if max_amp < 1e-6:
            print("⚠️ WARNING: Mic seems silent (amplitude ~0). Check if it's muted or broken.")
        
        # Chuẩn hóa âm lượng
        if max_amp > 1e-4:
            recording = (recording / max_amp) * 0.95
        
        # Lưu file WAV ở sample rate gốc
        recording_int16 = np.int16(recording * 32767)
        filename = "usb_mic_test.wav"
        wavfile.write(filename, native_sr, recording_int16)
        
        print(f"🚀 SUCCESS! Audio saved to '{filename}' ({native_sr}Hz, {duration}s).")
        print(f"   File size: {len(recording_int16) * 2 / 1024:.1f} KB")
        print("\nMẹo: Copy file về laptop bằng lệnh:")
        print(f"   scp tt04@pi4-server:~/Projects/2026_CPDS-AI/CPDS-AI/{filename} .")
        
    except Exception as e:
        print(f"\n❌ FAILED to record audio!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Run: sudo apt install portaudio19-dev")
        print("2. Check USB cable connection.")

if __name__ == "__main__":
    main()
