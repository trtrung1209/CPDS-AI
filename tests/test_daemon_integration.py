import pytest
import threading
import time
import socket
import base64
import json
import cv2
import numpy as np
from pathlib import Path

from src.daemon.server import InferenceServer
from src.daemon.config import Config

@pytest.fixture(scope="module")
def integration_server():
    # Only run this if the models actually exist
    config = Config()
    config.socket_path = "/tmp/test_integration.sock"
    if not Path(config.vision_model_path).exists() or not Path(config.audio_model_path).exists():
        pytest.skip("Models not found, skipping integration test.")

    server = InferenceServer(config)
    
    # Run server in a background thread
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    
    # Give it a moment to bind the socket
    time.sleep(1)
    
    yield config.socket_path
    
    # Cleanup (it runs as daemon so thread dies with pytest)
    if Path(config.socket_path).exists():
        Path(config.socket_path).unlink()

@pytest.mark.integration
def test_full_stack_socket(integration_server):
    socket_path = integration_server
    
    client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_sock.connect(str(socket_path))
    
    # 1. Test Vision
    # Create a dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    ret, buf = cv2.imencode('.jpg', img)
    b64_img = base64.b64encode(buf).decode('utf-8')
    
    req_vision = {"type": "vision", "jpeg_b64": b64_img}
    client_sock.sendall((json.dumps(req_vision) + "\n").encode('utf-8'))
    
    resp_line = client_sock.recv(4096).decode('utf-8')
    resp = json.loads(resp_line)
    
    assert resp["ok"] is True
    assert "child_detected" in resp["result"]
    assert resp["result"]["child_detected"] is False # It's a black image
    
    # 2. Test Audio
    # Create dummy pcm float32 audio
    pcm = np.zeros(16000 * 2, dtype=np.float32)
    b64_audio = base64.b64encode(pcm.tobytes()).decode('utf-8')
    
    req_audio = {"type": "audio", "pcm_b64": b64_audio, "sample_rate": 16000}
    client_sock.sendall((json.dumps(req_audio) + "\n").encode('utf-8'))
    
    resp_line2 = client_sock.recv(4096).decode('utf-8')
    resp2 = json.loads(resp_line2)
    
    assert resp2["ok"] is True
    assert "is_crying" in resp2["result"]
    
    client_sock.close()
