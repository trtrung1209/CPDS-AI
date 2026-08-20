# CPDS-AI — Child Protection & Distress Detection System

An edge-deployable, dual-modal AI system that detects children via camera (YOLOv8) and recognises baby cries via microphone (ResNet18), triggering real-time alerts when a child is in distress.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       CPDS-AI                            │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   Camera     │    │  Microphone   │    │  main.py   │  │
│  │  (Webcam)    │    │   (Mic)       │    │ (Unified   │  │
│  └──────┬───────┘    └──────┬────────┘    │  Entry     │  │
│         │                   │             │  Point)    │  │
│         ▼                   ▼             └─────┬──────┘  │
│  ┌─────────────┐    ┌──────────────┐            │         │
│  │  YOLOv8n    │    │  ResNet18    │            │         │
│  │  (Vision)   │    │  (Audio)     │◄───────────┘         │
│  │  .onnx      │    │  .onnx       │                      │
│  └──────┬───────┘    └──────┬────────┘                    │
│         │                   │                             │
│         ▼                   ▼                             │
│  ┌──────────────────────────────────┐                    │
│  │     Alarm Decision Engine        │                    │
│  │  Child detected + Crying = 🚨    │                    │
│  └──────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/trtrung1209/CPDS-AI.git && cd CPDS-AI

# 2. Setup environment
bash setup_environment.sh --full --microphone

# 3. Check readiness
.venv/bin/python main.py
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10, 3.12 |
| Git | any | For cloning datasets |
| ffmpeg | any | Audio format conversion |

```bash
# Ubuntu / Raspberry Pi OS
sudo apt-get install -y ffmpeg python3-venv

# macOS
brew install ffmpeg
```

---

## Environment Setup

```bash
# Audio only (librosa, onnxruntime, scikit-learn)
bash setup_environment.sh --audio

# Full (audio + vision + webcam)
bash setup_environment.sh --full

# Full + microphone recording
bash setup_environment.sh --full --microphone

# Recreate from scratch
bash setup_environment.sh --full --microphone --recreate
```

After setup, always use the venv Python:
```bash
.venv/bin/python main.py        # Linux / macOS / Pi
.venv\Scripts\python main.py    # Windows
```

---

## Usage (All via `main.py`)

### Check System
```bash
.venv/bin/python main.py
```
Verifies both ONNX models are present in `data/models/`.

---

### 👁️ Vision Testing

#### Test on a single image
```bash
.venv/bin/python main.py --mode file --image path/to/photo.jpg --audio path/to/sound.wav
```
The vision model draws bounding boxes and classifies each person as `adult` or `child`.

#### Live webcam detection
```bash
.venv/bin/python main.py --mode camera
```
Opens webcam, draws real-time bounding boxes. Green = Adult, Red = Child. Press **q** to quit.

#### Verify vision model on a single image (standalone)
```bash
.venv/bin/python -m src.inference.verify_vision --model data/models/yolov8n-adult-child.onnx --image test.jpg
```
Saves annotated output image to `runs/runN/verified_output.jpg`.

---

### 🔊 Audio Testing

#### Download test audio data
```bash
.venv/bin/python main.py --mode prepare
```
Downloads 20 cry + 20 noise samples into `data/test_audio/`. Converts all to 16kHz WAV.

#### Test a single audio file directly
```bash
.venv/bin/python main.py --mode audio --audio path/to/sound.wav
```

#### Run evaluation metrics (Accuracy, F1, Confusion Matrix)
```bash
.venv/bin/python main.py --mode evaluate
```
Runs unit tests + evaluates dataset. Prints report and saves JSON to `test_reports/audio_evaluation_report.json`.

#### Live microphone detection
```bash
.venv/bin/python main.py --mode mic
```
Records 2-second clips and classifies as `cry` or `noise`. Press **Ctrl+C** to stop.

#### Verify audio model on a single file (standalone)
```bash
.venv/bin/python -m src.inference.verify_audio --model data/models/audio_model.onnx --audio test.wav
```
Saves result JSON to `runs/runN/audio_verified.json`.

---

### 🚨 Dual-Modal Inference (Vision + Audio combined)

```bash
.venv/bin/python main.py --mode file --image photo.jpg --audio sound.wav
```
Runs **both** models and outputs an alarm decision:
- `🚨 ALARM TRIGGERED` = Child detected **AND** baby is crying
- `💤 No alarm` = Normal situation

---

## Project Structure

```
CPDS-AI/
├── main.py                     # 🎯 Unified entry point
├── setup_environment.sh        # 🔧 Creates .venv with correct deps
├── README.md                   # 📖 This file
│
├── data/
│   ├── models/
│   │   ├── yolov8n-adult-child.onnx   # Vision model (YOLOv8)
│   │   └── audio_model.onnx           # Audio model (ResNet18)
│   └── test_audio/                    # Generated by --mode prepare
│       ├── cry/                       # Baby cry WAV samples
│       └── noise/                     # Environmental noise WAV samples
│
├── src/
│   ├── inference/
│   │   ├── run_inference.py       # Dual-modal inference engine
│   │   ├── verify_audio.py        # Audio preprocessing + ONNX inference
│   │   ├── verify_vision.py       # YOLO inference + summarization
│   │   └── camera_vision.py       # Live webcam loop
│   └── utils.py                   # Run directory management
│
├── scripts/
│   ├── evaluate_audio_model.py    # Batch metrics with sklearn
│   ├── prepare_audio_evaluation_data.py  # Download + convert test data
│   ├── record_and_infer_audio.py  # Mic recording + inference
│   ├── run_tests.sh               # Full pytest suite
│   ├── run_vision_tests.sh        # Vision-specific tests
│   ├── run_audio_tests.sh         # Audio-specific tests
│   ├── shell_helpers.sh           # Shared bash utilities
│   └── generate_test_report.py    # Report generation
│
├── notebooks/
│   ├── 01_audio_training.ipynb    # Train audio on Kaggle
│   └── 02_vision_training.ipynb   # Train vision on Kaggle
│
├── tests/                         # pytest unit tests
├── docker/                        # Docker configs
│
├── requirements.txt               # Full deps
├── requirements-audio.txt         # Audio-only deps
├── requirements-test.txt          # Minimal test deps
└── requirements-microphone.txt    # Mic recording deps
```

---

## Deployment

### Raspberry Pi 4 / Orange Pi 5

**Transfer & Setup:**
```bash
scp -r CPDS-AI/ pi@<PI_IP>:~/CPDS-AI/
ssh pi@<PI_IP>
cd ~/CPDS-AI
sudo apt-get install -y python3-venv ffmpeg
bash setup_environment.sh --full --microphone
.venv/bin/python main.py
```

**Run as systemd service (24/7):**
```bash
sudo tee /etc/systemd/system/cpds-ai.service << EOF
[Unit]
Description=CPDS-AI Child Protection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/CPDS-AI
ExecStart=/home/pi/CPDS-AI/.venv/bin/python main.py --mode mic
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cpds-ai
sudo systemctl start cpds-ai
```

### INT8 Quantization (Orange Pi 5 NPU)
```bash
python3 -c "
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588', quantized_dtype='asymmetric_quantized-8')
rknn.load_onnx(model='data/models/audio_model.onnx')
rknn.build(do_quantization=True)
rknn.export_rknn('data/models/audio_model.rknn')
"
```

---

## Training (Re-training)

Both models are trained on **Kaggle** (free GPU):

1. **Audio**: Upload `notebooks/01_audio_training.ipynb` → Run All → Download `audio_model.onnx`
2. **Vision**: Upload `notebooks/02_vision_training.ipynb` → Run All → Download `yolov8n-adult-child.onnx`

Place models into `data/models/`.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Vision AI | YOLOv8n (Ultralytics) | Detect adults vs children |
| Audio AI | ResNet18 (PyTorch → ONNX) | Classify baby cry vs noise |
| Inference | ONNX Runtime | Cross-platform model execution |
| Audio Processing | librosa + ffmpeg | Mel-spectrogram extraction |
| Evaluation | scikit-learn | Precision, Recall, F1, Confusion Matrix |
| Deployment | systemd | 24/7 edge operation |
