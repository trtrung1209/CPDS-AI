import numpy as np
import pytest
import wave

from src.inference.verify_audio import load_labels


def test_default_audio_label_order_is_explicit():
    assert load_labels() == ["noise", "cry"]


def test_invalid_audio_labels_are_rejected(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text('["noise", "alarm"]', encoding="utf-8")

    with pytest.raises(ValueError, match="'cry'"):
        load_labels(labels_path)

# MOCK: Bỏ qua test nếu file ONNX chưa được tải về máy
@pytest.fixture
def dummy_audio_file(tmp_path):
    # Tạo một file âm thanh rác (.wav) 2 giây
    file_path = tmp_path / "dummy_audio.wav"
    sr = 16000
    audio_data = (np.random.uniform(-1, 1, int(sr * 2.0)) * 32767).astype(np.int16)
    with wave.open(str(file_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sr)
        output.writeframes(audio_data.tobytes())
    return str(file_path)

def test_audio_preprocessing_shape(dummy_audio_file):
    pytest.importorskip("librosa")
    from src.inference.verify_audio import preprocess_audio
    
    input_data = preprocess_audio(dummy_audio_file)
    
    # Kích thước phải là (1, 1, 128, T)
    assert len(input_data.shape) == 4
    assert input_data.shape[0] == 1
    assert input_data.shape[1] == 1
    assert input_data.shape[2] == 128
