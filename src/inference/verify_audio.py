import argparse
import json
from pathlib import Path
import numpy as np

from src.utils import get_next_run_dir, save_result

def preprocess_audio(audio_path, sr=16000, duration=2.0):
    """
    Extract a Mel spectrogram using the same shape as Kaggle training.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    import librosa

    y, sr = librosa.load(str(audio_path), sr=sr, duration=duration)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
        
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    
    # ONNX input shape: (batch, channel, mel_bins, time_steps)
    input_data = np.expand_dims(np.expand_dims(mel_spec_db, axis=0), axis=0)
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
    import onnxruntime as ort

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Audio model does not exist: {model_path}")

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
    return {
        "file": str(audio_path),
        "is_crying": bool(cry_index == int(np.argmax(probs))),
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
