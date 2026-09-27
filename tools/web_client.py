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
latest_raw_frame = None
latest_display_frame = None
frame_lock = threading.Lock()

# Vision AI results (bao gom bounding boxes)
vision_label = "Vision: Starting..."
vision_color = (80, 80, 80)
vision_boxes = []  # Danh sach cac bounding box tu AI

# Audio AI results
audio_label = "Mic: Starting..."
audio_color = (80, 80, 80)

SOCKET_PATH = "/tmp/cpds_inference.sock"

_sock = None
_infile = None
_outfile = None
_sock_lock = threading.Lock()


def _get_connection():
    global _sock, _infile, _outfile
    if _sock is not None:
        return _sock, _infile, _outfile
    try:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.settimeout(10)
        _sock.connect(SOCKET_PATH)
        _infile = _sock.makefile('r', encoding='utf-8')
        _outfile = _sock.makefile('w', encoding='utf-8')
        print("[Socket] Connected to AI Server")
        return _sock, _infile, _outfile
    except Exception:
        _sock = None
        _infile = None
        _outfile = None
        return None, None, None


def _close_connection():
    global _sock, _infile, _outfile
    try:
        if _sock: _sock.close()
    except Exception:
        pass
    _sock = None
    _infile = None
    _outfile = None


def send_to_server(req: dict) -> dict:
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


# ===========================================================
#  THREAD 1: Camera Capture
# ===========================================================
def camera_capture_thread():
    global latest_raw_frame
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Thu reset focus

    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    print("[Camera] Capture started")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_raw_frame = frame.copy()
        time.sleep(0.04)


# ===========================================================
#  THREAD 2: Display (ve bounding boxes + overlay)
# ===========================================================
def display_thread():
    global latest_display_frame
    while True:
        with frame_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # === VE BOUNDING BOXES ===
        for box in vision_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cls = box["class"]
            conf = box["confidence"]
            is_child = box.get("is_child", False)

            # Mau do cho tre em, xanh la cho nguoi lon
            color = (0, 0, 255) if is_child else (0, 220, 0)
            thickness = 3 if is_child else 2

            # Ve khung chu nhat
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Ve nhan (label) phia tren khung
            label = f"{'CHILD' if is_child else cls} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # === THANH TREN: Vision status ===
        cv2.rectangle(frame, (0, 0), (w, 32), vision_color, -1)
        cv2.putText(frame, vision_label, (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # === THANH DUOI: Audio status ===
        cv2.rectangle(frame, (0, h - 32), (w, h), audio_color, -1)
        cv2.putText(frame, audio_label, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 100, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        ret_enc, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret_enc:
            latest_display_frame = buf.tobytes()

        time.sleep(0.04)


# ===========================================================
#  THREAD 3: Vision AI
# ===========================================================
def vision_ai_thread():
    global vision_label, vision_color, vision_boxes
    print("[Vision AI] Thread started")

    while True:
        with frame_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

        if frame is None:
            time.sleep(0.5)
            continue

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(1)
            continue

        b64 = base64.b64encode(buf).decode('utf-8')
        resp = send_to_server({"type": "vision", "jpeg_b64": b64})

        if resp and resp.get("ok"):
            result = resp["result"]
            cls = result.get("class", "None")
            conf = result.get("confidence", 0.0)
            is_child = result.get("child_detected", False)
            boxes = result.get("boxes", [])

            # Cap nhat bounding boxes
            vision_boxes = boxes

            if cls != "None":
                n_det = len(boxes)
                if is_child:
                    vision_label = f"[!] CHILD: {conf:.0%} ({n_det} det)"
                    vision_color = (0, 0, 200)
                else:
                    vision_label = f"[OK] {cls}: {conf:.0%} ({n_det} det)"
                    vision_color = (0, 140, 0)
            else:
                vision_boxes = []
                vision_label = "No person detected"
                vision_color = (80, 80, 80)
        else:
            vision_label = "AI Server Offline"
            vision_color = (60, 60, 60)

        time.sleep(1.5)


# ===========================================================
#  THREAD 4: Audio AI
# ===========================================================
def audio_ai_thread():
    global audio_label, audio_color

    device_idx, native_sr = find_usb_mic()
    if device_idx is not None:
        print(f"[Audio] USB Mic: idx={device_idx}, sr={native_sr}Hz")
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
            print(f"[Audio] Raw peak amplitude: {max_val:.6f}")
            
            if max_val < 1e-4:
                audio_label = f"[!] Mic silent (amp={max_val:.6f})"
                audio_color = (0, 0, 150)
                time.sleep(1)
                continue

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
                probs = result.get("probabilities", {})
                noise_conf = probs.get("noise", 0.0)
                print(f"[Audio] AI result: cry={cry_conf:.2%}, noise={noise_conf:.2%}, peak={max_val:.4f}")

                if is_crying:
                    audio_label = f"[!] CRYING {cry_conf:.0%} | noise {noise_conf:.0%}"
                    audio_color = (0, 0, 200)
                else:
                    audio_label = f"[OK] cry {cry_conf:.0%} | noise {noise_conf:.0%}"
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


# ===========================================================
#  Flask
# ===========================================================
def generate_stream():
    while True:
        if latest_display_frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_display_frame + b'\r\n')


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
    threads = [
        ("Camera Capture", camera_capture_thread),
        ("Display Overlay", display_thread),
        ("Vision AI", vision_ai_thread),
        ("Audio AI", audio_ai_thread),
    ]
    for name, fn in threads:
        print(f"Starting {name} thread...")
        threading.Thread(target=fn, daemon=True).start()

    print("\n=== Web Dashboard ===")
    print("Mo Chrome: http://<IP-PI-4>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
