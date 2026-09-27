import socket
import json
import base64
import time
import cv2
import sounddevice as sd
from threading import Thread, Lock
from src.daemon.config import Config
from src.inference.run_inference import should_trigger_alarm
from src.utils import get_next_run_dir, save_result

class WatchClient:
    def __init__(self, config: Config):
        self.config = config
        self.sock = None
        self.infile = None
        self.outfile = None
        
        self.lock = Lock()
        self.latest_vision = None
        self.vision_time = 0
        self.latest_audio = None
        self.audio_time = 0
        
        self.alarm_state = False
        self.running = True

    def _connect(self):
        while self.running:
            try:
                if self.sock:
                    self.sock.close()
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(str(self.config.socket_path))
                self.infile = self.sock.makefile('r', encoding='utf-8')
                self.outfile = self.sock.makefile('w', encoding='utf-8')
                print("Connected to server.")
                break
            except Exception as e:
                print(f"Reconnect failed: {e}. Retrying in 2s...")
                time.sleep(2)

    def _send_request(self, req: dict) -> dict:
        while self.running:
            try:
                if not self.sock or not self.outfile or not self.infile:
                    self._connect()
                self.outfile.write(json.dumps(req) + "\n")
                self.outfile.flush()
                resp_line = self.infile.readline()
                if not resp_line:
                    raise ConnectionError("Socket closed by server.")
                return json.loads(resp_line)
            except Exception as e:
                print(f"Send error: {e}. Reconnecting...")
                self._connect()
        return {}

    def vision_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            b64 = base64.b64encode(buf).decode('utf-8')
            
            resp = self._send_request({"type": "vision", "jpeg_b64": b64})
            if resp and resp.get("ok"):
                with self.lock:
                    self.latest_vision = resp["result"]
                    self.vision_time = time.time()
                self._check_alarm()
            
            time.sleep(0.5)
        cap.release()

    def _find_usb_mic(self):
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
                return i, int(dev['default_samplerate'])
        return None, 44100

    def audio_loop(self):
        device_idx, sample_rate = self._find_usb_mic()
        if device_idx is not None:
            print(f"Using USB Mic [idx {device_idx}] at {sample_rate}Hz")
        else:
            print("WARNING: No USB Mic found. Falling back to default.")

        duration = 2.0
        while self.running:
            try:
                if device_idx is not None:
                    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32", device=device_idx)
                else:
                    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
                sd.wait()
                recording = recording.flatten()
                
                # Noise Gate: Calculate RMS and drop packet if it's just static
                rms = (recording**2).mean()**0.5
                if rms < 0.02:
                    print(f"Audio ignored (Noise Gate): RMS {rms:.4f} < 0.02")
                    continue
                
                b64 = base64.b64encode(recording.tobytes()).decode('utf-8')
                resp = self._send_request({"type": "audio", "pcm_b64": b64, "sample_rate": sample_rate})
                
                if resp and resp.get("ok"):
                    with self.lock:
                        self.latest_audio = resp["result"]
                        self.audio_time = time.time()
                    self._check_alarm()
                
                # Sleep exactly 3s as required by duty cycle
                for _ in range(30):
                    if not self.running: break
                    time.sleep(0.1)
            except Exception as e:
                print(f"Audio loop error: {e}")
                time.sleep(3)

    def _trigger_gpio(self, alarm: bool):
        if alarm != self.alarm_state:
            print(f"GPIO BUZZER PIN {self.config.gpio_buzzer_pin} -> {'HIGH' if alarm else 'LOW'}")
            self.alarm_state = alarm

    def _check_alarm(self):
        with self.lock:
            now = time.time()
            v_fresh = (now - self.vision_time) <= self.config.staleness_window
            a_fresh = (now - self.audio_time) <= self.config.staleness_window
            
            if v_fresh and a_fresh and self.latest_vision and self.latest_audio:
                alarm = should_trigger_alarm(
                    self.latest_vision, 
                    self.latest_audio, 
                    self.config.vision_threshold, 
                    self.config.cry_threshold
                )
                self._trigger_gpio(alarm)
                
                # Mọi lần đều trigger alarm? Không, chỉ lưu log khi alarm trạng thái thay đổi hoặc có sự kiện đặc biệt
                if alarm:
                    log_data = {
                        "timestamp": now,
                        "vision": self.latest_vision,
                        "audio": self.latest_audio,
                        "alarm_triggered": alarm
                    }
                    save_result(get_next_run_dir(), "daemon_log.json", json.dumps(log_data, indent=2))
            else:
                self._trigger_gpio(False)

    def run(self):
        vt = Thread(target=self.vision_loop, daemon=True)
        at = Thread(target=self.audio_loop, daemon=True)
        vt.start()
        at.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False

if __name__ == "__main__":
    WatchClient(Config()).run()
