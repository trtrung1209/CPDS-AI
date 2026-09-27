import cv2
import json
import base64
import socket
import time
import threading
import numpy as np
import sounddevice as sd
from flask import Flask, Response

app = Flask(__name__)

# === SHARED STATE ===
latest_frame = None
vision_label = ""
vision_color = (80, 80, 80)  # Xam mac dinh
audio_label = "Mic: Waiting..."
audio_color = (80, 80, 80)
SOCKET_PATH = "/tmp/cpds_inference.sock"

# Socket ket noi lau dai (Persistent Connection)
_sock = None
_infile = None
_outfile = None
_sock_lock = threading.Lock()


def _get_connection():
    """Tao hoac tai su dung ket noi toi AI Server."""
    global _sock, _infile, _outfile
    if _sock is not None:
        return _sock, _infile, _outfile
    try:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.settimeout(10)
        _sock.connect(SOCKET_PATH)
        _infile = _sock.makefile('r', encoding='utf-8')
        _outfile = _sock.makefile('w', encoding='utf-8')
        return _sock, _infile, _outfile
    except Exception:
        _sock = None
        _infile = None
        _outfile = None
        return None, None, None


def _close_connection():
    global _sock, _infile, _outfile
    try:
        if _sock:
            _sock.close()
    except Exception:
        pass
    _sock = None
    _infile = None
    _outfile = None


def send_to_server(req: dict) -> dict:
    """Gui request toi AI Server qua persistent Unix Socket."""
    with _sock_lock:
        sock, infile, outfile = _get_connection()
        if not sock:
            return None
        try:
            outfile.write(json.dumps(req) + "\n")
            outfile.flush()
            resp_line = infile.readline()
            if not resp_line:
                _close_connection()
                return None
            return json.loads(resp_line)
        except Exception:
            _close_connection()
            return None


def find_usb_mic():
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
            return i, int(dev['default_samplerate'])
    return None, 44100


def camera_thread():
    """Luong Camera: Chup anh lien tuc, KHONG CHO AI."""
    global latest_frame
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # === THANH TREN: Vision ===
        cv2.rectangle(frame, (0, 0), (w, 36), vision_color, -1)
        cv2.putText(frame, vision_label if vision_label else "Vision: Waiting...",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # === THANH DUOI: Audio ===
        cv2.rectangle(frame, (0, h - 36), (w, h), audio_color, -1)
        cv2.putText(frame, audio_label,
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 105, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc:
            latest_frame = buffer.tobytes()

        time.sleep(0.05)  # ~20 FPS cho muot


def vision_ai_thread():
    """Luong AI Vision: Cu moi 1 giay gui 1 frame cho AI (TACH BIET voi Camera)."""
    global vision_label, vision_color
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        ret_enc, buf = cv2.imencode('.jpg', frame)
        if not ret_enc:
            time.sleep(1)
            continue

        b64 = base64.b64encode(buf).decode('utf-8')
        resp = send_to_server({"type": "vision", "jpeg_b64": b64})

        if resp and resp.get("ok"):
            result = resp["result"]
            cls = result.get("class", "None")
            conf = result.get("confidence", 0.0)
            is_child = result.get("child_detected", False)

            if cls != "None":
                if is_child:
                    vision_label = f"[!] CHILD DETECTED: {conf:.0%}"
                    vision_color = (0, 0, 200)  # Do
                else:
                    vision_label = f"[OK] {cls}: {conf:.0%}"
                    vision_color = (0, 160, 0)  # Xanh
            else:
                vision_label = "No detection"
                vision_color = (80, 80, 80)
        else:
            vision_label = "AI Server Offline"
            vision_color = (60, 60, 60)

        time.sleep(1)  # Chi gui 1 frame/giay de Pi 4 khong qua tai


def audio_ai_thread():
    """Luong AI Audio: Thu am va gui cho AI."""
    global audio_label, audio_color

    device_idx, native_sr = find_usb_mic()
    if device_idx is not None:
        print(f"[Audio] USB Mic found: idx={device_idx}, sr={native_sr}Hz")
    else:
        print("[Audio] No USB Mic found!")
        audio_label = "No USB Mic"
        return

    duration = 2.0
    while True:
        try:
            recording = sd.rec(
                int(duration * native_sr),
                samplerate=native_sr,
                channels=1, dtype='float32',
                device=device_idx
            )
            sd.wait()
            recording = recording.flatten()

            max_val = np.max(np.abs(recording))
            if max_val > 1e-4:
                recording = (recording / max_val) * 0.95

            b64 = base64.b64encode(recording.tobytes()).decode('utf-8')
            resp = send_to_server({
                "type": "audio",
                "pcm_b64": b64,
                "sample_rate": native_sr
            })

            if resp and resp.get("ok"):
                result = resp["result"]
                cry_conf = result.get("confidence", 0.0)
                is_crying = result.get("is_crying", False)
                if is_crying:
                    audio_label = f"[!] CRYING DETECTED ({cry_conf:.0%})"
                    audio_color = (0, 0, 200)
                else:
                    audio_label = f"[OK] No cry ({cry_conf:.0%})"
                    audio_color = (80, 80, 80)
            else:
                audio_label = "AI Audio Offline"
                audio_color = (60, 60, 60)

            time.sleep(1)

        except Exception as e:
            print(f"[Audio] Error: {e}")
            audio_label = "Mic Error"
            audio_color = (0, 0, 150)
            time.sleep(3)


def generate_stream():
    while True:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')


@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CPDS-AI Dashboard</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e0e0e0;
      font-family:'Segoe UI',Arial,sans-serif;min-height:100vh;
      display:flex;flex-direction:column;align-items:center;justify-content:center}
    h1{font-size:1.6rem;font-weight:600;margin:20px 0 10px;
      background:linear-gradient(90deg,#00d2ff,#3a7bd5);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .sub{font-size:.85rem;color:#888;margin-bottom:16px}
    .vc{border:3px solid rgba(255,255,255,.1);border-radius:14px;overflow:hidden;
      box-shadow:0 0 40px rgba(0,210,255,.15);max-width:95vw}
    .vc img{display:block;width:640px;max-width:95vw;height:auto}
    .legend{display:flex;gap:20px;margin-top:12px;font-size:.8rem}
    .legend span{display:flex;align-items:center;gap:5px}
    .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
    .dr{background:#e44}.dg{background:#4c4}.dd{background:#888}
    .ft{margin-top:18px;font-size:.75rem;color:#555}
  </style>
</head>
<body>
  <h1>CPDS-AI Live Monitor</h1>
  <p class="sub">Child Presence Detection System — Raspberry Pi 4</p>
  <div class="vc"><img src="/video_feed" alt="Live"/></div>
  <div class="legend">
    <span><span class="dot dr"></span>Child / Crying</span>
    <span><span class="dot dg"></span>Adult / No cry</span>
    <span><span class="dot dd"></span>AI Offline</span>
  </div>
  <p class="ft">USB Camera + MI 305 Mic | YOLOv8 + Audio ONNX</p>
</body></html>"""


@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # 3 luong doc lap: Camera (muot), Vision AI (cham), Audio AI (cham)
    for name, fn in [("Camera", camera_thread), ("Vision AI", vision_ai_thread), ("Audio AI", audio_ai_thread)]:
        print(f"Starting {name} thread...")
        threading.Thread(target=fn, daemon=True).start()

    print("\n=== Web Dashboard ===")
    print("Mo Chrome tren Laptop, go: http://<IP-PI-4>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
