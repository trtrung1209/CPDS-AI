# CPDS-AI (Child Presence Detection System - AI)

Scientific Research Project: **AI-based Child Presence Detection and Alert System in Vehicles with Remote Firmware Update (FOTA) Capabilities.**

## 1. Overview
This system utilizes Deep Learning pipelines to monitor and detect the presence of children in vehicle cabins, issuing alerts when a child is left behind.
The goal is to run two machine learning models concurrently:
1. **Vision Model:** YOLOv8 (Adult vs. Child classification).
2. **Audio Model:** Audio classification (Detecting baby cries while filtering out ambient vehicle noise).

The entire system will be containerized using Docker and deployed on embedded devices (Raspberry Pi 4, and later Orange Pi 5 with NPU).

## 2. Project Directory Structure
- `data/`: Contains audio and image datasets (ignored by gitignore to avoid pushing heavy files to GitHub).
- `notebooks/`: Contains Python scripts/Jupyter Notebooks for model training (can be run on Kaggle/Colab).
- `src/`: Main source code containing data preprocessing logic, model definitions, and inference.
  - Inference/training results are automatically saved into directories like `runs/run1`, `runs/run2`, etc.
- `tests/`: Contains automated test suites using `pytest`; no real model is needed to test the logic.
- `docker/`: Contains Dockerfiles and environment configurations.
- `.github/workflows/`: Contains CI/CD scripts for GitHub to automatically run tests on every push/PR.
- `run_vision_tests.sh`: Helper bash script to automatically run all vision-related tests.

## 3. Environment Setup (Local / Laptop)

To test the code locally before deploying to the Pi:

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install required libraries
pip install -r requirements.txt
```

## 4. Automated Testing
The project uses `pytest` to verify the processing pipelines. To run all unit tests, execute the following command at the project root:
```bash
python3 -m pytest
```
The system will automatically scan the `tests/` directory. GitHub Actions also runs this exact command on every push/PR.

## 5. Train and Export Models on Kaggle
**Audio:** Attach the dataset to the Kaggle notebook and update `DATASET_DIR` in `01_audio_training.ipynb`. The dataset must follow the structure `train/noise`, `train/cry` (and optionally `val/noise`, `val/cry`).

**Vision:** Create a Kaggle Secret named `ROBOFLOW_API_KEY`, grant access to the notebook, and update the workspace/project/version in `02_vision_training.ipynb`. Do not hardcode the API key in the source code. The notebook retrieves `results.save_dir` from Ultralytics, so it does not depend on a hardcoded `runs/` structure.

Download the artifacts after training into the `data/models/` directory (this directory is not committed):

- `best.onnx` (and `vision_metadata.json`) from `/kaggle/working/artifacts/` of the vision notebook;
- `audio_model.onnx` and `audio_labels.json` from the audio notebook.

## 6. Running ONNX Inference
Inference uses the real models, no more mock results. Outputs are saved in `runs/run1`, `runs/run2`, etc.:

```bash
python3 -m src.inference.run_inference \
  --image sample.jpg --audio sample.wav \
  --vision-model data/models/best.onnx \
  --audio-model data/models/audio_model.onnx \
  --audio-labels data/models/audio_labels.json
```

To verify the vision model in real-time using your webcam:
```bash
python3 src/inference/camera_vision.py --model data/models/best.onnx
```

Use the helper script to run fast vision unit tests. Add `--image` to run a real ONNX smoke test, or `--camera` for a webcam demo:
```bash
bash run_vision_tests.sh --image test_anh.jpg data/models/best.onnx
bash run_vision_tests.sh --camera data/models/best.onnx
```

## 7. Docker
To deploy smoothly on a Pi 4 (using Ubuntu Server), you can build and run Docker:
```bash
docker build -t cpds-inference -f docker/Dockerfile.inference .
docker run --rm -it -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/runs:/app/runs" cpds-inference \
  python3 -m src.inference.run_inference --image /app/data/sample.jpg --audio /app/data/sample.wav
```

## 8. MLOps & Workflow Best Practices

- **Never rename model files dynamically:** Do not rename your ONNX files (e.g., from `best.onnx` to `test_best.onnx`) just to test them. This is an anti-pattern. Always keep the original names (e.g., `v1.onnx`, `v2.onnx`) and use the `--model` command-line argument to specify which model the script should load.
- **Docker vs. Native Execution:** 
  - Scripts that require hardware peripherals (like `camera_vision.py` which needs your webcam) or graphical UI windows should be run **natively** on your host machine (Laptop/PC) during development.
  - **Docker** is intended for the final headless deployment on the embedded device (Raspberry Pi / Orange Pi). The Docker container perfectly replicates the OS and dependencies without interfering with the host machine.
- **Test Automation:** Always run `bash run_vision_tests.sh` before committing changes. It automatically verifies both the software logic (`pytest`) and the physical ONNX model execution, ensuring nothing is broken before deployment.
