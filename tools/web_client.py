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

latest_frame = None
ai_vision_status = "⏳ Waiting..."
ai_audio_status = "🎙️ Waiting..."
audio_result_text = ""
audio_is_crying = False
SOCKET_PATH = "/tmp/cpds_inference.sock"


def find_usb_mic():
    """Tìm thiết bị USB Microphone trong danh sách."""
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and 'usb' in dev['name'].lower():
            return i, int(dev['default_samplerate'])
    return None, 44100


def send_to_server(req: dict) -> dict:
    """Gửi request tới AI Server qua Unix Socket."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(SOCKET_PATH)
        infile = sock.makefile('r', encoding='utf-8')
        outfile = sock.makefile('w', encoding='utf-8')

        outfile.write(json.dumps(req) + "\n")
        outfile.flush()

        resp_line = infile.readline()
        sock.close()

        if resp_line:
            return json.loads(resp_line)
    except Exception:
        pass
    return None


def draw_overlay(frame, ai_result):
    """Vẽ cảnh báo Vision + Audio lên hình ảnh."""
    h, w = frame.shape[:2]

    # === THANH TRÊN: Vision ===
    if ai_result and ai_result.get("class", "None") != "None":
        class_name = ai_result.get("class", "None")
        confidence = ai_result.get("confidence", 0.0)
        is_child = ai_result.get("child_detected", False)

        color = (0, 0, 255) if is_child else (0, 200, 0)
        label = f"{'CHILD' if is_child else class_name}: {confidence:.0%}"

        cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
        cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(frame, ai_vision_status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    # === THANH DƯỚI: Audio ===
    if audio_is_crying:
        audio_color = (0, 0, 255)  # Đỏ = Đang khóc
    else:
        audio_color = (80, 80, 80)

    cv2.rectangle(frame, (0, h - 40), (w, h), audio_color, -1)
    cv2.putText(frame, audio_result_text if audio_result_text else ai_audio_status,
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Timestamp góc dưới phải
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 110, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return frame


def camera_thread():
    """Luồng Camera: Chụp ảnh → Gửi AI → Vẽ kết quả."""
    global latest_frame, ai_vision_status
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Đọc bỏ vài frame đầu để Camera tự lấy sáng
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Gửi ảnh cho AI
        ai_result = None
        ret_enc, buf = cv2.imencode('.jpg', frame)
        if ret_enc:
            b64 = base64.b64encode(buf).decode('utf-8')
            resp = send_to_server({"type": "vision", "jpeg_b64": b64})
            if resp and resp.get("ok"):
                ai_result = resp["result"]
                ai_vision_status = "🟢 AI Vision OK"
            else:
                ai_vision_status = "🔴 AI Server Offline"

        # Vẽ kết quả lên ảnh
        frame = draw_overlay(frame, ai_result)

        # Nén và cập nhật frame cho Web
        ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret_enc:
            latest_frame = buffer.tobytes()

        time.sleep(0.2)


def audio_thread():
    """Luồng Audio: Thu âm → Gửi AI → Cập nhật trạng thái."""
    global ai_audio_status, audio_result_text, audio_is_crying

    device_idx, native_sr = find_usb_mic()
    if device_idx is not None:
        print(f"🎙️ USB Mic found: [idx {device_idx}] at {native_sr}Hz")
    else:
        print("⚠️ No USB Mic found. Audio disabled.")
        ai_audio_status = "❌ No USB Mic"
        return

    duration = 2.0
    while True:
        try:
            recording = sd.rec(
                int(duration * native_sr),
                samplerate=native_sr,
                channels=1,
                dtype='float32',
                device=device_idx
            )
            sd.wait()
            recording = recording.flatten()

            # Chuẩn hóa
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

                audio_is_crying = is_crying
                if is_crying:
                    audio_result_text = f"🔊 CRYING DETECTED! ({cry_conf:.0%})"
                else:
                    audio_result_text = f"🔇 No cry ({cry_conf:.0%})"
                ai_audio_status = "🟢 AI Audio OK"
            else:
                ai_audio_status = "🔴 AI Audio Offline"

            # Nghỉ 1 giây giữa các lần thu
            time.sleep(1)

        except Exception as e:
            print(f"Audio error: {e}")
            ai_audio_status = f"❌ Mic Error"
            time.sleep(3)


def generate_video_stream():
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')


@app.route('/')
def index():
    html = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CPDS-AI Dashboard</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      color: #e0e0e0;
      font-family: 'Segoe UI', Arial, sans-serif;
      min-height: 100vh;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    h1 {
      font-size: 1.6rem; font-weight: 600; margin: 20px 0 10px;
      background: linear-gradient(90deg, #00d2ff, #3a7bd5);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 16px; }
    .video-container {
      border: 3px solid rgba(255,255,255,0.1);
      border-radius: 14px; overflow: hidden;
      box-shadow: 0 0 40px rgba(0, 210, 255, 0.15);
      max-width: 95vw;
    }
    .video-container img { display: block; width: 640px; max-width: 95vw; height: auto; }
    .footer { margin-top: 18px; font-size: 0.75rem; color: #555; }
    .legend {
      display: flex; gap: 20px; margin-top: 12px; font-size: 0.8rem;
    }
    .legend span { display: flex; align-items: center; gap: 5px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .dot-red { background: #ff4444; }
    .dot-green { background: #44ff44; }
    .dot-gray { background: #888; }
  </style>
</head>
<body>
  <h1>CPDS-AI Live Monitor</h1>
  <p class="subtitle">Child Presence Detection System &mdash; Raspberry Pi 4</p>
  <div class="video-container">
    <img src="/video_feed" alt="Live Camera Feed" />
  </div>
  <div class="legend">
    <span><span class="dot dot-red"></span> Child / Crying detected</span>
    <span><span class="dot dot-green"></span> Adult / No cry</span>
    <span><span class="dot dot-gray"></span> AI Offline</span>
  </div>
  <p class="footer">Streaming from Pi 4 &bull; USB Camera + MI 305 Mic &bull; YOLOv8 + Audio ONNX</p>
</body>
</html>"""
    return html


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("Starting Camera Thread...")
    ct = threading.Thread(target=camera_thread, daemon=True)
    ct.start()

    print("Starting Audio Thread...")
    at = threading.Thread(target=audio_thread, daemon=True)
    at.start()

    print("\n🌐 Web Dashboard is running!")
    print("Mở Chrome trên Laptop, gõ địa chỉ Pi 4:")
    print("http://<IP-CỦA-PI-4>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
