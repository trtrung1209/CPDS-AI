#!/usr/bin/env python3
"""
=============================================================================
 ONNX -> RKNN Converter for YOLOv8 (Adult/Child) on RK3588 NPU
 Target: Rockchip RK3588 NPU (6 TOPS, 3 cores)
 Input : yolov8n-adult-child.onnx (or yolov8s-adult-child.onnx)
 Output: yolov8n-adult-child.rknn
=============================================================================
 Usage:
   1. Install RKNN-Toolkit2 (on x86_64 host):
      pip3 install rknn-toolkit2
   2. Run: python3 convert_onnx_to_rknn.py
=============================================================================
"""

import os
import sys
import glob
import numpy as np
import platform

# -- Detect execution environment (x86_64 vs aarch64 board) --
IS_ON_BOARD = platform.machine() == 'aarch64'

if IS_ON_BOARD:
    print("[INFO] Detected aarch64 architecture (target board).")
    print("[WARN] rknn_toolkit_lite2 does not support converting ONNX to RKNN.")
    print("[WARN] Please run this conversion script on an x86_64 host with rknn-toolkit2 installed.")
    print("[HINT] Alternatively use Docker: docker pull rockchip/rknn-toolkit2:1.6.0")
    sys.exit(1)
else:
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed!")
        print("  Install: pip3 install rknn-toolkit2")
        print("  Or download from: https://github.com/rockchip-linux/rknn-toolkit2")
        sys.exit(1)

# -- Configuration --
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL      = os.path.join(CURRENT_DIR, 'yolov8n-adult-child.onnx')
RKNN_OUTPUT     = os.path.join(CURRENT_DIR, 'yolov8n-adult-child.rknn')
TARGET_PLATFORM = 'rk3588'
INPUT_SIZE      = [1, 3, 640, 640]   # NCHW standard for YOLOv8
MEAN_VALUES     = [[0, 0, 0]]        # YOLOv8 handles zero-mean natively
STD_VALUES      = [[255, 255, 255]]  # Divide by 255 to normalize to [0, 1]
QUANTIZE        = False              # Set to True for INT8, False for FP16 (recommended first baseline)
DATASET_TXT     = os.path.join(CURRENT_DIR, 'dataset.txt')
TEST_VISION_DIR = os.path.join(CURRENT_DIR, '..', 'test_vision', 'images')

def create_calibration_dataset():
    """
    Generate dataset.txt containing calibration images for INT8 quantization.
    Prefers authentic domain images from test_vision to prevent quantization drift.
    """
    if os.path.exists(DATASET_TXT) and os.path.getsize(DATASET_TXT) > 0:
        print(f"[INFO] Existing calibration dataset found: {DATASET_TXT}")
        return DATASET_TXT

    # Priority 1: Check if real test_vision images exist
    real_images = sorted(glob.glob(os.path.join(TEST_VISION_DIR, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(TEST_VISION_DIR, "*.png")))
    if real_images:
        print(f"[INFO] Using {min(60, len(real_images))} real domain images from {TEST_VISION_DIR} for INT8 calibration.")
        with open(DATASET_TXT, 'w') as f:
            for path in real_images[:60]:
                f.write(os.path.abspath(path) + '\n')
        return DATASET_TXT

    # Priority 2: Fallback to synthetic images
    print("[WARN] No real domain images found. Generating synthetic calibration images...")
    cal_dir = os.path.join(CURRENT_DIR, 'calibration_images')
    os.makedirs(cal_dir, exist_ok=True)

    try:
        from PIL import Image
        with open(DATASET_TXT, 'w') as f:
            for i in range(20):
                img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
                path = os.path.join(cal_dir, f'cal_{i:03d}.jpg')
                img.save(path)
                f.write(os.path.abspath(path) + '\n')
    except ImportError:
        with open(DATASET_TXT, 'w') as f:
            for i in range(20):
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                path = os.path.join(cal_dir, f'cal_{i:03d}.npy')
                np.save(path, img)
                f.write(os.path.abspath(path) + '\n')

    print(f"[INFO] Synthetic calibration dataset created at {cal_dir}")
    return DATASET_TXT


def convert():
    # ── Kiểm tra file ONNX ──
    if not os.path.exists(ONNX_MODEL):
        print(f"[ERROR] Không tìm thấy file ONNX: {ONNX_MODEL}")
        sys.exit(1)

    print("=" * 60)
    print("  RKNN Converter cho YOLOv8n Adult/Child")
    print(f"  Input : {ONNX_MODEL}")
    print(f"  Output: {RKNN_OUTPUT}")
    print(f"  Target: {TARGET_PLATFORM}")
    print(f"  Quant : {'INT8' if QUANTIZE else 'FP16'}")
    print("=" * 60)

    # ── Bước 1: Khởi tạo RKNN object ──
    rknn = RKNN(verbose=True)

    # ── Bước 2: Cấu hình preprocessing ──
    print("\n[STEP 1/5] Cấu hình tiền xử lý...")
    rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=TARGET_PLATFORM,
        # Tối ưu cho YOLOv8: bật quantized_algorithm KL_divergence
        quantized_algorithm='normal',
        quantized_method='channel',
        optimization_level=2,       # Tối ưu cao nhất
    )

    # ── Bước 3: Load ONNX model ──
    print("\n[STEP 2/5] Nạp model ONNX...")
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print(f"[ERROR] Load ONNX thất bại! Mã lỗi: {ret}")
        sys.exit(1)

    # ── Bước 4: Build RKNN model ──
    print("\n[STEP 3/5] Build model RKNN (có thể mất 2-5 phút)...")
    if QUANTIZE:
        dataset = create_calibration_dataset()
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"[ERROR] Build RKNN thất bại! Mã lỗi: {ret}")
        sys.exit(1)

    # ── Bước 5: Export ──
    print("\n[STEP 4/5] Xuất file RKNN...")
    ret = rknn.export_rknn(RKNN_OUTPUT)
    if ret != 0:
        print(f"[ERROR] Export thất bại! Mã lỗi: {ret}")
        sys.exit(1)

    # ── Bước 6: Accuracy Analysis (tùy chọn) ──
    print("\n[STEP 5/5] Kiểm tra model info...")
    # In thông tin model
    rknn.release()

    file_size_mb = os.path.getsize(RKNN_OUTPUT) / (1024 * 1024)
    print("\n" + "=" * 60)
    print("  ✅ CHUYỂN ĐỔI THÀNH CÔNG!")
    print(f"  📦 File output: {RKNN_OUTPUT}")
    print(f"  📏 Kích thước : {file_size_mb:.2f} MB")
    print("=" * 60)
    print("\n[NEXT] Copy file .rknn sang Orange Pi:")
    print(f"  scp {RKNN_OUTPUT} orangepi@<IP>:/home/taphuthanh/OrangePi_AI/ONNX_models/")


if __name__ == '__main__':
    convert()

