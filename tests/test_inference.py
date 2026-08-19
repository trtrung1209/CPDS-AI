from src.inference.run_inference import mock_vision_inference, mock_audio_inference

def test_vision_inference():
    res = mock_vision_inference("dummy.jpg")
    assert "class" in res
    assert "confidence" in res
    assert res["class"] in ["Child", "Adult"]

def test_audio_inference():
    res = mock_audio_inference("dummy.wav")
    assert "is_crying" in res
    assert type(res["is_crying"]) is bool
