from typing import Protocol, Dict, Any
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write as write_wav

class BackendProtocol(Protocol):
    def predict_vision(self, image_data: bytes) -> Dict[str, Any]: ...
    def predict_audio(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]: ...

class ONNXBackend:
    def __init__(self, config):
        self.config = config
        self.vision_model = None
        self.audio_session = None
        self.audio_labels = None
        self._load_models()

    def _load_models(self):
        from ultralytics import YOLO
        import onnxruntime as ort
        from src.inference.verify_audio import load_labels
        
        self.vision_model = YOLO(self.config.vision_model_path, task="detect")
        self.audio_session = ort.InferenceSession(self.config.audio_model_path)
        self.audio_labels = load_labels(self.config.audio_labels_path if self.config.audio_labels_path else None)

    def predict_vision(self, image_data: bytes) -> Dict[str, Any]:
        import cv2
        from src.inference.verify_vision import summarize_detections
        
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data")
            
        results = self.vision_model(img, verbose=False)
        result = results[0]
        
        # Trích xuất toạ độ bounding box để vẽ lên hình
        boxes_data = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id])
                confidence = float(box.conf.item())
                boxes_data.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "class": class_name,
                    "confidence": confidence,
                    "is_child": class_name.strip().lower() in {"child", "children", "kid"},
                })
        
        if result.boxes is None or len(result.boxes) == 0:
            summary = summarize_detections(result.names, [])
        else:
            summary = summarize_detections(result.names, result.boxes)
        
        summary["boxes"] = boxes_data
        return summary

    def predict_audio(self, audio_data: bytes, sample_rate: int) -> Dict[str, Any]:
        from src.inference.verify_audio import preprocess_audio
        pcm_data = np.frombuffer(audio_data, dtype=np.float32)
        
        # Tái sử dụng logic Peak normalization từ record_and_infer_audio
        max_val = np.max(np.abs(pcm_data))
        if max_val > 1e-4:
            pcm_data = (pcm_data / max_val) * 0.95
            
        # Ghi ra file tạm để tái sử dụng preprocess_audio nguyên vẹn
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            write_wav(temp_path, sample_rate, pcm_data)
            # Ép resample về 16000Hz để khớp với dữ liệu huấn luyện của ONNX
            input_data = preprocess_audio(temp_path, sr=16000, duration=2.0)
        finally:
            os.remove(temp_path)
            
        input_name = self.audio_session.get_inputs()[0].name
        output_name = self.audio_session.get_outputs()[0].name
        result = self.audio_session.run([output_name], {input_name: input_data})[0]
        
        logits = np.asarray(result).squeeze()
        exp_res = np.exp(logits - np.max(logits))
        probs = exp_res / exp_res.sum()
        
        probabilities = {label: float(probability) for label, probability in zip(self.audio_labels, probs)}
        cry_index = self.audio_labels.index("cry")
        return {
            "is_crying": bool(cry_index == int(np.argmax(probs))),
            "confidence": float(probs[cry_index]),
            "probabilities": probabilities,
        }
