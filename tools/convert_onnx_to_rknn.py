#!/usr/bin/env python3
"""
=============================================================================
 ONNX -> RKNN Converter for YOLOv8 (Adult/Child) on RK3588 NPU
 Target: Rockchip RK3588 NPU (6 TOPS, 3 cores)
=============================================================================
 Usage:
   1. Install RKNN-Toolkit2 (on x86_64 host):
      pip3 install rknn-toolkit2
   2. Run conversion:
      python3 tools/convert_onnx_to_rknn.py
      # Or specify paths:
      python3 tools/convert_onnx_to_rknn.py --onnx data/models/yolov8s-adult-child.onnx --quantize
=============================================================================
"""

import argparse
import glob
import os
import platform
import sys
from pathlib import Path

# Detect architecture (x86_64 host vs aarch64 board)
IS_ON_BOARD = platform.machine() == "aarch64"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_environment() -> None:
    if IS_ON_BOARD:
        print("[INFO] Detected aarch64 architecture (target edge board).")
        print("[WARN] rknn_toolkit_lite2 does not support converting ONNX to RKNN.")
        print("[WARN] Please run this conversion script on an x86_64 PC/server with rknn-toolkit2.")
        print("[HINT] Alternatively use Docker on PC: docker pull rockchip/rknn-toolkit2:1.6.0")
        sys.exit(1)

    try:
        from rknn.api import RKNN  # noqa: F401
    except ImportError:
        print("[ERROR] rknn-toolkit2 is not installed!")
        print("  Install: pip3 install rknn-toolkit2")
        print("  Official repo: https://github.com/rockchip-linux/rknn-toolkit2")
        sys.exit(1)


def get_calibration_dataset(cache_dir: Path, test_vision_dir: Path) -> Path:
    """Prepare a dataset.txt containing sample images for INT8 calibration."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_txt = cache_dir / "dataset.txt"

    if dataset_txt.is_file() and dataset_txt.stat().st_size > 0:
        return dataset_txt

    # Priority 1: Use authentic domain images from test_vision
    real_images = sorted(list(test_vision_dir.glob("*.jpg")) + list(test_vision_dir.glob("*.png")))
    if real_images:
        sample_count = min(60, len(real_images))
        print(f"[INFO] Using {sample_count} real domain images from {test_vision_dir} for INT8 calibration.")
        with open(dataset_txt, "w") as f:
            for path in real_images[:sample_count]:
                f.write(str(path.resolve()) + "\n")
        return dataset_txt

    # Priority 2: Fallback to synthetic images
    print("[WARN] No real test images found. Creating synthetic calibration data...")
    import numpy as np
    img_dir = cache_dir / "synthetic_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with open(dataset_txt, "w") as f:
            for i in range(20):
                arr = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                p = img_dir / f"cal_{i:03d}.jpg"
                Image.fromarray(arr).save(p)
                f.write(str(p.resolve()) + "\n")
    except ImportError:
        with open(dataset_txt, "w") as f:
            for i in range(20):
                arr = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                p = img_dir / f"cal_{i:03d}.npy"
                np.save(p, arr)
                f.write(str(p.resolve()) + "\n")

    return dataset_txt


def convert(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str = "rk3588",
    quantize: bool = False,
) -> int:
    check_environment()
    from rknn.api import RKNN

    if not onnx_path.is_file():
        print(f"[ERROR] ONNX model not found: {onnx_path}", file=sys.stderr)
        return 1

    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CPDS-AI: ONNX -> RKNN Model Converter")
    print("=" * 60)
    print(f"  Input ONNX : {onnx_path}")
    print(f"  Output RKNN: {rknn_path}")
    print(f"  Platform   : {target_platform}")
    print(f"  Quantize   : {'INT8' if quantize else 'FP16'}")
    print("=" * 60)

    rknn = RKNN(verbose=False)

    # 1. Preprocessing configuration
    print("\n[STEP 1/4] Configuring preprocessing...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
        quantized_algorithm="normal",
        quantized_method="channel",
        optimization_level=2,
    )

    # 2. Load ONNX graph
    print("\n[STEP 2/4] Loading ONNX model...")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print(f"[ERROR] Failed to load ONNX model (error code: {ret})", file=sys.stderr)
        return 1

    # 3. Build RKNN graph
    print(f"\n[STEP 3/4] Building RKNN graph (quantization={'INT8' if quantize else 'FP16'})...")
    if quantize:
        cal_cache = PROJECT_ROOT / ".cache" / "rknn_calibration"
        test_dir = PROJECT_ROOT / "data" / "test_vision" / "images"
        dataset_file = get_calibration_dataset(cal_cache, test_dir)
        ret = rknn.build(do_quantization=True, dataset=str(dataset_file))
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"[ERROR] Failed to build RKNN model (error code: {ret})", file=sys.stderr)
        return 1

    # 4. Export RKNN
    print("\n[STEP 4/4] Exporting RKNN model...")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print(f"[ERROR] Failed to export RKNN model (error code: {ret})", file=sys.stderr)
        return 1

    rknn.release()

    size_mb = rknn_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("  [SUCCESS] RKNN Model Exported!")
    print(f"  Path : {rknn_path}")
    print(f"  Size : {size_mb:.2f} MB")
    print("=" * 60)
    print(f"\nTo deploy to Orange Pi 5:")
    print(f"  scp {rknn_path} <user>@<board_ip>:/path/to/models/\n")
    return 0


def get_default_models() -> tuple[Path, Path]:
    v8s_onnx = PROJECT_ROOT / "data" / "models" / "yolov8s-adult-child.onnx"
    v8n_onnx = PROJECT_ROOT / "data" / "models" / "yolov8n-adult-child.onnx"
    chosen_onnx = v8n_onnx if v8n_onnx.is_file() else v8s_onnx
    chosen_rknn = chosen_onnx.with_suffix(".rknn")
    return chosen_onnx, chosen_rknn


def main() -> int:
    default_onnx, default_rknn = get_default_models()
    parser = argparse.ArgumentParser(description="Convert YOLOv8 ONNX model to RKNN for Rockchip NPU.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=default_onnx,
        help=f"Input ONNX model path (default: {default_onnx.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_rknn,
        help=f"Output RKNN model path (default: {default_rknn.name})",
    )
    parser.add_argument(
        "--target",
        default="rk3588",
        choices=["rk3588", "rk3566", "rk3568", "rk3562", "rv1103", "rv1106"],
        help="Target Rockchip platform (default: rk3588)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (default: FP16 without quantization)",
    )
    args = parser.parse_args()
    return convert(args.onnx, args.output, args.target, args.quantize)


if __name__ == "__main__":
    raise SystemExit(main())
