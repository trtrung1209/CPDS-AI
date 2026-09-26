import pytest
import time
import json
from unittest.mock import MagicMock, patch
from src.daemon.client import WatchClient
from src.daemon.config import Config

@pytest.fixture
def client():
    config = Config()
    config.staleness_window = 1.0 # short staleness for testing
    config.vision_threshold = 0.60
    config.cry_threshold = 0.70
    return WatchClient(config)

def test_trigger_gpio(client):
    assert client.alarm_state is False
    client._trigger_gpio(True)
    assert client.alarm_state is True
    client._trigger_gpio(False)
    assert client.alarm_state is False

@patch("src.daemon.client.save_result")
def test_check_alarm_staleness(mock_save, client):
    # Both fresh
    client.vision_time = time.time()
    client.latest_vision = {"child_detected": True, "confidence": 0.8}
    client.audio_time = time.time()
    client.latest_audio = {"is_crying": True, "confidence": 0.8}
    
    client._check_alarm()
    assert client.alarm_state is True
    mock_save.assert_called_once()
    
    # Audio gets stale
    client.audio_time = time.time() - 2.0
    client._check_alarm()
    assert client.alarm_state is False
    
def test_check_alarm_thresholds(client):
    client.vision_time = time.time()
    client.audio_time = time.time()
    
    # Below threshold
    client.latest_vision = {"child_detected": True, "confidence": 0.5}
    client.latest_audio = {"is_crying": True, "confidence": 0.8}
    client._check_alarm()
    assert client.alarm_state is False
    
    # Vision ok, Audio below threshold
    client.latest_vision = {"child_detected": True, "confidence": 0.8}
    client.latest_audio = {"is_crying": True, "confidence": 0.5}
    client._check_alarm()
    assert client.alarm_state is False

@patch("time.sleep", return_value=None)
@patch("cv2.VideoCapture")
def test_vision_loop_error_handling(mock_cap, mock_sleep, client):
    mock_vc = MagicMock()
    mock_cap.return_value = mock_vc
    
    # Ret False (frame cannot be read)
    mock_vc.read.side_effect = [(False, None), (True, MagicMock())]
    
    def stop_loop(*args, **kwargs):
        client.running = False
        return {"ok": True, "result": {"child_detected": True}}
        
    client._send_request = MagicMock(side_effect=stop_loop)
    
    # Also patch imencode
    with patch("cv2.imencode", return_value=(True, b"img_data")):
        client.vision_loop()
            
    assert mock_vc.read.call_count == 2
    mock_sleep.assert_called()

@patch("socket.socket")
@patch("time.sleep", return_value=None)
def test_connect_reconnect(mock_sleep, mock_socket_cls, client):
    mock_sock = MagicMock()
    # First connect fails, second succeeds
    mock_sock.connect.side_effect = [ConnectionError, None]
    mock_socket_cls.return_value = mock_sock
    
    # We must mock infile/outfile so it doesn't crash on makefile
    mock_sock.makefile.return_value = MagicMock()
    
    client._connect()
    assert mock_sock.connect.call_count == 2
    assert mock_sleep.call_count == 1
    
@patch("socket.socket")
@patch("time.sleep", return_value=None)
def test_send_request_reconnects(mock_sleep, mock_socket_cls, client):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    
    mock_infile = MagicMock()
    # First readline returns empty (socket closed), second returns valid JSON
    mock_infile.readline.side_effect = ["", json.dumps({"ok": True}) + "\n"]
    mock_sock.makefile.side_effect = [mock_infile, MagicMock(), mock_infile, MagicMock()]
    
    resp = client._send_request({"type": "ping"})
    assert resp["ok"] is True
    # Reconnected once
    assert mock_sock.connect.call_count == 2

@patch("sounddevice.rec")
@patch("sounddevice.wait")
@patch("time.sleep", return_value=None)
def test_audio_loop(mock_sleep, mock_wait, mock_rec, client):
    mock_rec.return_value = MagicMock()
    mock_rec.return_value.flatten.return_value.tobytes.return_value = b"pcm"
    
    call_count = [0]
    def mock_send(*args, **kwargs):
        call_count[0] += 1
        # Stop after the second iteration (one successful send, then one more)
        if call_count[0] >= 2:
            client.running = False
        return {"ok": True, "result": {"is_crying": True}}
        
    client._send_request = MagicMock(side_effect=mock_send)
    
    client.audio_loop()
        
    # Wait is called, sleep is called many times in duty cycle
    mock_wait.assert_called()
    # Duty cycle logic loops 30 times of 0.1s - should have been called since running=False happens after first send
    assert mock_sleep.call_count >= 30

@patch("src.daemon.client.Thread")
@patch("time.sleep", side_effect=KeyboardInterrupt)
def test_client_run(mock_sleep, mock_thread_cls, client):
    mock_vt = MagicMock()
    mock_at = MagicMock()
    mock_thread_cls.side_effect = [mock_vt, mock_at]
    
    client.run()
    
    mock_vt.start.assert_called()
    mock_at.start.assert_called()
    assert client.running is False
