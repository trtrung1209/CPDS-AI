import argparse
import json
from pathlib import Path
import numpy as np

from src.utils import get_next_run_dir, save_result


def validate_audio_runtime():
    """Return librosa only after its lazy-loaded audio backend has been verified."""
    try:
        import librosa
        from librosa.core import audio as _audio_backend  # noqa: F401
    except Exception as error:
        raise RuntimeError(
            "Audio dependencies could not be imported. Create a virtual environment and install the pinned "
            "project dependencies with: python3 -m pip install -r requirements.txt"
        ) from error
    return librosa


def select_loudest_window(y, sr, duration=2.0, hop_fraction=0.1):
    target_len = int(sr * duration)
    if len(y) <= target_len:
        return np.pad(y, (0, target_len - len(y)))
    hop = max(1, int(target_len * hop_fraction))
    best_start, best_energy = 0, -1.0
    for start in range(0, len(y) - target_len + 1, hop):
        segment = y[start:start + target_len]
        energy = float(np.sum(segment.astype(np.float64) ** 2))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return y[best_start:best_start + target_len]

def preprocess_audio(audio_path, sr=16000, duration=2.0):
    """
    Extract a Mel spectrogram using the EXACT SAME logic as Kaggle training.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    librosa = validate_audio_runtime()

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    if y.size == 0:
        y = np.zeros(int(sr * duration), dtype=np.float32)
        
    # 1. Tìm đoạn 2 giây chứa âm thanh to nhất (tránh cắt nhầm khoảng lặng đầu clip)
    y = select_loudest_window(y, sr, duration=duration)
        
    # 2. Peak normalization (This was done before saving wavs in Kaggle!)
    max_val = np.max(np.abs(y))
    if np.isfinite(max_val) and max_val > 1e-4:
        y = (y / max_val) * 0.95
        
    target_len = max(1, int(sr * duration))
    if y.size > target_len:
        y = y[:target_len]

    n_fft = 2048
    n_mels = 128
    n_frames = 63
    
    if len(y) < n_fft:
        pad_len = n_fft + (n_frames - 1) - len(y)
        y = np.pad(y, (0, max(0, pad_len)), mode="constant")
        hop_length = 1
    else:
        hop_length = max(1, (len(y) - n_fft) // (n_frames - 1))
        target_samples = n_fft + (n_frames - 1) * hop_length
        if len(y) > target_samples:
            y = y[:target_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0, fmax=sr / 2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.asarray(mel_db, dtype=np.float32)
    
    mel_min = np.min(mel_db)
    mel_max = np.max(mel_db)
    if np.isfinite(mel_min) and np.isfinite(mel_max) and (mel_max - mel_min) > 1e-8:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    else:
        mel_db = np.zeros_like(mel_db, dtype=np.float32)
        
    mel_db = np.clip(mel_db, 0.0, 1.0)
    
    if mel_db.shape[1] < n_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, n_frames - mel_db.shape[1])), mode="constant")
    elif mel_db.shape[1] > n_frames:
        mel_db = mel_db[:, :n_frames]
        
    # ONNX expects (batch, channel, mels, time_steps) -> (1, 1, 128, 63)
    input_data = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=0)
    return input_data.astype(np.float32)

def load_labels(labels_path=None):
    """Load the class order exported next to the audio model."""
    if labels_path is None:
        return ["noise", "cry"]

    with Path(labels_path).open(encoding="utf-8") as label_file:
        labels = json.load(label_file)
    if (
        not isinstance(labels, list)
        or len(labels) != 2
        or not all(isinstance(label, str) for label in labels)
        or labels.count("cry") != 1
    ):
        raise ValueError("Audio labels must be a JSON list of two classes and include exactly one 'cry' label.")
    return labels


def infer_audio(model_path, audio_path, labels_path=None):
    """Run ONNX audio inference and return probabilities with explicit labels."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Audio model does not exist: {model_path}")

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    input_data = preprocess_audio(audio_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})[0]

    logits = np.asarray(result).squeeze()
    labels = load_labels(labels_path)
    if logits.ndim != 1 or logits.size != len(labels):
        raise ValueError(f"Expected {len(labels)} audio logits, got shape {np.asarray(result).shape}.")

    exp_res = np.exp(logits - np.max(logits))
    probs = exp_res / exp_res.sum()

    probabilities = {label: float(probability) for label, probability in zip(labels, probs)}
    cry_index = labels.index("cry")
    
    # MẸO TĂNG ĐỘ NHẠY (RECALL): Hạ ngưỡng threshold xuống 0.25 thay vì 0.5 (argmax)
    # Vì False Positive của ta rất thấp (chỉ 1 ca), ta có quyền bắt nhạy hơn để không bỏ sót bé nào!
    threshold = 0.25
    is_crying_detected = bool(probs[cry_index] > threshold)
    
    return {
        "file": str(audio_path),
        "is_crying": is_crying_detected,
        "confidence": float(probs[cry_index]),
        "probabilities": probabilities,
    }


def verify_audio_model(model_path, audio_path, labels_path=None):
    """Run audio inference, save the result and return it."""
    output_json = infer_audio(model_path, audio_path, labels_path)
    run_dir = get_next_run_dir()
    save_result(run_dir, "audio_verified.json", json.dumps(output_json, indent=4))
    print(json.dumps(output_json, indent=4))
    print(f"Saved to: {run_dir}/audio_verified.json")
    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Audio ONNX Model")
    parser.add_argument("--model", type=str, required=True, help="Path to audio_model.onnx")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio (.wav)")
    parser.add_argument("--labels", type=str, help="Optional path to audio_labels.json exported by training")
    
    args = parser.parse_args()
    try:
        verify_audio_model(args.model, args.audio, args.labels)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
