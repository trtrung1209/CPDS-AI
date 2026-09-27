"""Lightweight TFLite-based vision verifier optimized for Raspberry Pi 4.

This module replaces the heavier ONNX Runtime approach with a TFLite
interpreter (or `tflite_runtime` if available) and uses a threaded camera
reader to avoid IO bottlenecks. Input images are downscaled to 320x320 to
maximize FPS on CPU-only devices.

Techniques used for speeding up inference:
- Use `tflite_runtime` / `tensorflow.lite.Interpreter` with delegate support if
  available on the device.
- Reduce input resolution to 320x320 (user requested, mandatory).
- Keep a tiny frame queue (maxsize=2) to avoid memory growth and reduce
  producer/consumer stalls.
- Reuse interpreter and tensors between inferences.
"""

import argparse
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Dict, List

import cv2
import numpy as np

try:
    # Prefer the lightweight runtime on Pi
    from tflite_runtime.interpreter import Interpreter
    TFLITE_RUNTIME = True
except Exception:
    try:
        from tensorflow.lite import Interpreter
        TFLITE_RUNTIME = False
    except Exception:
        raise RuntimeError("No TFLite interpreter available. Install tflite-runtime or tensorflow.")


class ThreadedVideoCapture:
    """Background frame reader to avoid blocking on cv2.VideoCapture.read().

    - Keeps a bounded queue (maxsize=2) to avoid memory growth.
    - Automatically drops oldest frames if the consumer is slow.
    """

    def __init__(self, src=0, width=320, height=320, queue_size=2):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.queue: Queue = Queue(maxsize=queue_size)
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # Drop oldest frame if queue is full (we want freshest frame)
            if self.queue.full():
                try:
                    _ = self.queue.get_nowait()
                except Exception:
                    pass
            self.queue.put(frame)

    def read(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()


class TFLiteDetector:
    def __init__(self, model_path: str, input_size: int = 320, score_threshold: float = 0.3):
        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.interpreter = Interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self._get_io_details()

    def _get_io_details(self):
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Expect BGR input from OpenCV; convert to RGB and resize to input_size
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, frame: np.ndarray) -> Dict:
        """Run inference and return normalized detection summary dict.

        Returns a dict with at least: {"child_detected": bool, "confidence": float, "detections": [...]}
        Where detections is a list of {"box": [x1,y1,x2,y2], "score": float, "class_id": int} in image pixel coords.
        """
        input_tensor = self.preprocess(frame)
        # set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Collect output arrays
        outputs: List[np.ndarray] = [self.interpreter.get_tensor(o['index']) for o in self.output_details]

        # Flexible parsing: many exported TFLite detection models return either:
        # - [boxes, scores, classes, nums] (common format) OR
        # - single array [N,6] with [x1,y1,x2,y2,score,class]
        detections = []
        if len(outputs) == 1:
            out = outputs[0]
            # If shape is (N,6) assume columns x1,y1,x2,y2,score,class
            if out.ndim == 2 and out.shape[1] >= 6:
                for row in out:
                    x1, y1, x2, y2, score, cls = row[:6]
                    if score < self.score_threshold:
                        continue
                    detections.append({
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "class_id": int(cls),
                    })
        elif len(outputs) >= 3:
            # heuristic: try to find boxes (N,4), scores (N,) and classes (N,)
            boxes, scores, classes = outputs[0], outputs[1], outputs[2]
            # Some exporters give shapes [1,N,4] etc.
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            # Normalize to image coords (assume boxes are normalized [0-1]) if needed
            h, w = frame.shape[:2]
            if boxes.max() <= 1.01:
                boxes_px = boxes.copy()
                boxes_px[:, [0,2]] *= w
                boxes_px[:, [1,3]] *= h
            else:
                boxes_px = boxes
            for i in range(boxes_px.shape[0]):
                sc = float(scores[i])
                if sc < self.score_threshold:
                    continue
                x1, y1, x2, y2 = boxes_px[i].tolist()
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": sc,
                    "class_id": int(classes[i]) if classes.ndim > 0 else int(classes),
                })

        # Summarize: find most confident 'child' detection
        child_detected = False
        best_conf = 0.0
        best_det = None
        for d in detections:
            # Heuristic: class_id 0 often corresponds to 'person' or 'child' depending on labels.
            # For robust behavior, caller can map class_id -> name externally.
            if d['score'] > best_conf:
                best_conf = d['score']
                best_det = d
        if best_det is not None:
            # Very simple heuristic: if best confidence > 0.4 treat as detection
            child_detected = best_conf >= 0.4

        return {"child_detected": bool(child_detected), "confidence": float(best_conf), "detections": detections}


def verify_vision_model(model_path: str, image_path: str):
    """Compatibility wrapper that runs model on a single image and saves annotated output."""
    model_path = Path(model_path)
    image_path = Path(image_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Vision model does not exist: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    detector = TFLiteDetector(str(model_path), input_size=320)
    img = cv2.imread(str(image_path))
    res = detector.infer(img)

    # draw detections on image for verification and save next to run dir
    for d in res.get("detections", []):
        x1, y1, x2, y2 = map(int, d['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{d['score']:.2f}", (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out_path = Path.cwd() / "verified_output_tflite.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Verification saved to: {out_path}")
    return res


def demo_camera_loop(model_path: str, src=0, runtime_seconds: int = 10):
    """Demo loop: threaded capture -> inference -> print summary.

    Intended to run on Pi 4 to measure FPS. Keep runtime_seconds small for quick tests.
    """
    detector = TFLiteDetector(str(model_path), input_size=320)
    cap = ThreadedVideoCapture(src=src, width=320, height=320, queue_size=2)
    cap.start()
    t0 = time.time()
    frames = 0
    try:
        while time.time() - t0 < runtime_seconds:
            frame = cap.read()
            if frame is None:
                time.sleep(0.005)
                continue
            frames += 1
            r = detector.infer(frame)
            print(f"Child: {r['child_detected']} conf={r['confidence']:.3f} detections={len(r['detections'])}")
    finally:
        cap.stop()
        elapsed = time.time() - t0
        print(f"Processed {frames} frames in {elapsed:.2f}s -> {frames/elapsed:.2f} FPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TFLite vision model on Raspberry Pi (320x320)")
    parser.add_argument("--model", type=str, required=True, help="Path to TFLite model")
    parser.add_argument("--image", type=str, required=False, help="Path to test image")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (int)")
    parser.add_argument("--demo-seconds", type=int, default=10, help="Run camera demo for N seconds")
    args = parser.parse_args()
    if args.image:
        verify_vision_model(args.model, args.image)
    else:
        demo_camera_loop(args.model, src=args.camera, runtime_seconds=args.demo_seconds)
