import pytest
import json
import base64
from unittest.mock import MagicMock, patch
from src.daemon.server import InferenceServer
from src.daemon.config import Config
from src.daemon.backend import BackendProtocol

class MockBackend:
    def predict_vision(self, image_data: bytes):
        return {"child_detected": True, "confidence": 0.9}
    
    def predict_audio(self, audio_data: bytes, sample_rate: int):
        return {"is_crying": True, "confidence": 0.8}

@pytest.fixture
def mock_config():
    config = Config()
    config.socket_path = "/tmp/test_socket.sock"
    config.backend_type = "onnx"
    return config

@pytest.fixture
def server(mock_config):
    with patch("src.daemon.server.ONNXBackend"):
        srv = InferenceServer(mock_config)
        srv.backend = MockBackend()
        return srv

def test_process_request_vision(server):
    req = {
        "type": "vision",
        "jpeg_b64": base64.b64encode(b"dummy_image").decode("utf-8")
    }
    line = json.dumps(req)
    res = server._process_request(line)
    assert res["ok"] is True
    assert res["result"]["child_detected"] is True

def test_process_request_audio(server):
    req = {
        "type": "audio",
        "pcm_b64": base64.b64encode(b"dummy_audio").decode("utf-8"),
        "sample_rate": 16000
    }
    line = json.dumps(req)
    res = server._process_request(line)
    assert res["ok"] is True
    assert res["result"]["is_crying"] is True

def test_process_request_unknown_type(server):
    req = {"type": "unknown"}
    line = json.dumps(req)
    res = server._process_request(line)
    assert res["ok"] is False
    assert "Unknown request type" in res["error"]

def test_process_request_invalid_json(server):
    line = "{invalid_json: true"
    res = server._process_request(line)
    assert res["ok"] is False
    assert "error" in res

def test_process_request_missing_b64(server):
    req = {"type": "vision"}
    line = json.dumps(req)
    res = server._process_request(line)
    assert res["ok"] is False

@patch("socket.socket")
def test_handle_client(mock_sock_class, server):
    mock_conn = MagicMock()
    
    # 2 requests then EOF
    valid_req = json.dumps({"type": "vision", "jpeg_b64": base64.b64encode(b"d").decode('utf-8')}) + "\n"
    invalid_req = json.dumps({"type": "unknown"}) + "\n"
    
    # Mock infile and outfile as context managers
    mock_infile = MagicMock()
    mock_outfile = MagicMock()
    
    # Iterate over the two requests
    mock_infile.__iter__.return_value = iter([valid_req, invalid_req])
    mock_infile.__enter__.return_value = mock_infile
    mock_infile.__exit__.return_value = None
    
    mock_outfile.__enter__.return_value = mock_outfile
    mock_outfile.__exit__.return_value = None
    
    # makefile returns these mocks
    mock_conn.makefile.side_effect = [mock_infile, mock_outfile]
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    
    server._handle_client(mock_conn)
    
    assert mock_outfile.write.call_count == 2
    mock_outfile.flush.assert_called()

def test_init_invalid_backend():
    config = Config()
    config.backend_type = "invalid"
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        InferenceServer(config)
