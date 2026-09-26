import cv2
import time

def test_camera(source, name, api_preference=None):
    print(f"Trying to open camera source: {name}...")
    
    if api_preference:
        cap = cv2.VideoCapture(source, api_preference)
    else:
        cap = cv2.VideoCapture(source)
        
    if not cap.isOpened():
        print(f"❌ Failed to open {name}.")
        return False
        
    print(f"✅ Camera opened! Warming up sensor...")
    # Read a few frames to let camera auto-exposure adjust
    for _ in range(10):
        ret, frame = cap.read()
        time.sleep(0.1)
        
    if ret and frame is not None:
        filename = "csi_camera_test_result.jpg"
        cv2.imwrite(filename, frame)
        print(f"🚀 SUCCESS! Captured an image of size {frame.shape} and saved to '{filename}'.")
        cap.release()
        return True
    else:
        print(f"❌ Opened camera but failed to grab frame using {name}.")
        cap.release()
        return False

def main():
    print("=== Raspberry Pi CSI Camera Test ===")
    
    # 1. Standard V4L2 (Index 0) - This is standard for Ubuntu 22.04 with bcm2835-v4l2 driver
    if test_camera(0, "Standard Index 0"):
        return
        
    # 2. Explicit V4L2 API
    if test_camera(0, "Explicit V4L2 API", cv2.CAP_V4L2):
        return
        
    # 3. GStreamer Pipeline (libcamera) - Standard for newer Raspberry Pi OS (Bullseye/Bookworm)
    gstreamer_pipeline = (
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 "
        "! videoconvert ! appsink"
    )
    if test_camera(gstreamer_pipeline, "GStreamer (libcamera)", cv2.CAP_GSTREAMER):
        return
        
    print("\n⚠️ All camera attempts failed!")
    print("Troubleshooting checklist for Pi 4:")
    print("1. Are you sure the CSI ribbon cable is plugged in the right way? (Blue side facing ethernet port usually)")
    print("2. Did you enable legacy camera in 'sudo raspi-config'? (If using Raspberry Pi OS Legacy)")
    print("3. Try running: ls /dev/video* to see if the system even detects the camera hardware.")

if __name__ == "__main__":
    main()
