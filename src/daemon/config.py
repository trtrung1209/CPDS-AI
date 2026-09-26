import os
from pathlib import Path

class Config:
    def __init__(self):
        self.vision_model_path = os.environ.get("CPDS_VISION_MODEL", "data/models/yolov8n-adult-child.onnx")
        self.audio_model_path = os.environ.get("CPDS_AUDIO_MODEL", "data/models/audio_model.onnx")
        self.audio_labels_path = os.environ.get("CPDS_AUDIO_LABELS", "")
        self.vision_threshold = float(os.environ.get("CPDS_VISION_THRESHOLD", "0.60"))
        self.cry_threshold = float(os.environ.get("CPDS_CRY_THRESHOLD", "0.70"))
        self.socket_path = os.environ.get("CPDS_SOCKET_PATH", "/tmp/cpds_inference.sock")
        self.gpio_buzzer_pin = int(os.environ.get("CPDS_GPIO_BUZZER_PIN", "17"))
        self.staleness_window = float(os.environ.get("CPDS_STALENESS_WINDOW", "5.0"))
        self.backend_type = os.environ.get("CPDS_BACKEND", "onnx")
