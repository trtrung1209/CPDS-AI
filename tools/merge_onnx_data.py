#!/usr/bin/env python3
"""
Consolidate external ONNX tensor data into a single self-contained .onnx model file.
Prevents file desynchronization when deploying models to edge platforms like RK3588.
"""

import argparse
import sys
from pathlib import Path


def merge_onnx_external_data(model_path: Path, output_path: Path | None = None) -> int:
    """Load an ONNX model with external data and save it as a unified standalone file."""
    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        import onnx
    except ImportError:
        print("[ERROR] 'onnx' Python package is required. Install via: pip install onnx", file=sys.stderr)
        return 1

    save_path = output_path or model_path
    print(f"[INFO] Loading ONNX model: {model_path}")

    try:
        # Load model and explicitly resolve external data
        model = onnx.load(str(model_path), load_external_data=True)
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}", file=sys.stderr)
        return 1

    print(f"[INFO] Embedding external weights into standalone file: {save_path}")
    try:
        onnx.save_model(model, str(save_path), save_as_external_data=False)
    except Exception as e:
        print(f"[ERROR] Failed to save consolidated model: {e}", file=sys.stderr)
        return 1

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Model successfully consolidated ({file_size_mb:.2f} MB): {save_path}")

    # Check if obsolete external data files remain
    possible_data_file = model_path.with_name(model_path.name + ".data")
    alternative_data_file = model_path.with_suffix(".onnx.data")
    for extra_file in (possible_data_file, alternative_data_file):
        if extra_file.is_file() and extra_file != save_path:
            print(f"[HINT] You can now safely remove the obsolete external data file: {extra_file}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate external ONNX weights (.data) into a single standalone .onnx file."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/models/audio_model.onnx"),
        help="Path to the target ONNX model (default: data/models/audio_model.onnx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination path (defaults to overwriting the input model)",
    )
    args = parser.parse_args()
    return merge_onnx_external_data(args.model, args.output)


if __name__ == "__main__":
    raise SystemExit(main())

