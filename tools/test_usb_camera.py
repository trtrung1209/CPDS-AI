import cv2
import time

def main():
    print("=== Raspberry Pi USB Camera Test ===")
    
    # Trên Raspberry Pi, cổng video0 đôi khi là bộ giải mã phần cứng.
    # Nên Webcam USB thường sẽ nhảy sang cổng video1, video2 hoặc video0.
    for index in [0, 1, 2]:
        print(f"\nTrying USB Camera at index {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print(f"❌ Could not open camera at index {index}.")
            continue
            
        print("✅ Camera opened! Warming up for 2 seconds...")
        # Đọc bỏ 10 frames đầu để camera tự động lấy nét và cân bằng sáng
        for _ in range(10):
            ret, frame = cap.read()
            time.sleep(0.2)
            
        if ret and frame is not None:
            filename = f"usb_camera_test_idx{index}.jpg"
            cv2.imwrite(filename, frame)
            print(f"🚀 SUCCESS! Saved frame to {filename}. Size: {frame.shape}")
            cap.release()
            return  # Dừng lại ngay khi tìm thấy camera hoạt động
        else:
            print("❌ Opened, but failed to capture frame (Image is None).")
            cap.release()
            
    print("\n⚠️ All attempts failed!")
    print("Troubleshooting: Run 'ls /dev/video*' in terminal to check if the USB Camera is plugged in correctly.")

if __name__ == "__main__":
    main()
