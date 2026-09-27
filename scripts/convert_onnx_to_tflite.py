"""Convert an ONNX YOLO model to a TFLite INT8-quantized model.

Notes:
- This script attempts to convert ONNX -> TensorFlow SavedModel (via onnx-tf),
  then SavedModel -> TFLite using the TensorFlow Lite converter with a
  representative dataset for INT8 quantization.
- On Raspberry Pi you should install `tflite-runtime` for inference. The
  conversion step (TensorFlow) typically runs on your workstation (x86)
  and not on the Pi.

Usage:
  python scripts/convert_onnx_to_tflite.py --onnx path/to/model.onnx \
      --out path/to/model_int8.tflite --rep-data-dir ./rep_images --input-size 320

Representative images (a handful, 100-500 images) should be provided in
`--rep-data-dir` for accurate INT8 calibration.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def load_image(path, size=320):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def representative_dataset_gen(rep_dir, input_size=320, max_images=200):
    files = list(Path(rep_dir).glob("**/*.*"))
    files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for i, f in enumerate(files[:max_images]):
        arr = load_image(f, size=input_size)
        # TFLite converter representative dataset yields a list of input arrays
        yield [arr.astype(np.float32)]


def convert(onnx_path: str, out_tflite: str, rep_data_dir: str, input_size: int = 320):
    # Import here because conversion tools are heavy
    try:
        import onnx
    except Exception:
        raise RuntimeError("Please install onnx (pip install onnx) on your conversion host.")

    # Try onnx-tf to convert ONNX to a TensorFlow SavedModel
    try:
        from onnx_tf.backend import prepare
    except Exception:
        raise RuntimeError(
            "Please install onnx-tf (pip install onnx-tf) to convert ONNX to TensorFlow.")

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = Path(out_tflite).with_suffix("")
    saved_model_dir = Path(str(saved_model_dir) + "_savedmodel")
    if saved_model_dir.exists():
        # remove older files? keep it simple
        pass
    tf_rep.export_graph(str(saved_model_dir))

    # Now convert SavedModel -> TFLite with INT8 quantization
    try:
        import tensorflow as tf
    except Exception:
        raise RuntimeError("Please install tensorflow on the conversion host to run the TFLite converter.")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(rep_data_dir, input_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Ensure int8 input/output if needed for full integer quant
    converter.inference_input_type = tf.uint8 if False else tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = Path(out_tflite)
    out_path.write_bytes(tflite_model)
    print(f"Wrote INT8 TFLite model to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--out", required=True, help="Path to output TFLite model")
    p.add_argument("--rep-data-dir", required=True, help="Representative images directory")
    p.add_argument("--input-size", type=int, default=320, help="Model input size (320)")
    args = p.parse_args()
    convert(args.onnx, args.out, args.rep_data_dir, args.input_size)


if __name__ == "__main__":
    main()
