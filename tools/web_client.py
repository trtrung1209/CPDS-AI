import cv2
import json
import base64
import socket
import time
import threading
from flask import Flask, Response

app = Flask(__name__)

# Biến toàn cục để lưu trữ frame mới nhất (đã vẽ ô vuông)
latest_frame = None
# Đường dẫn tới não bộ AI đang chạy ngầm
SOCKET_PATH = "/tmp/cpds_inference.sock"

def get_ai_prediction(frame):
    """Gửi frame cho Server AI qua Unix Socket và nhận kết quả."""
    try:
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret: return None
        b64 = base64.b64encode(buf).decode('utf-8')
        
        # Kết nối tới Server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        infile = sock.makefile('r', encoding='utf-8')
        outfile = sock.makefile('w', encoding='utf-8')
        
        req = {"type": "vision", "jpeg_b64": b64}
        outfile.write(json.dumps(req) + "\n")
        outfile.flush()
        
        resp_line = infile.readline()
        sock.close()
        
        if resp_line:
            return json.loads(resp_line).get("result", {})
    except Exception as e:
        print(f"Socket Error: {e}")
    return None

def draw_boxes(frame, ai_result):
    """Vẽ cảnh báo lên hình ảnh."""
    if not ai_result:
        return frame
        
    class_name = ai_result.get("class", "None")
    confidence = ai_result.get("confidence", 0.0)
    
    color = (0, 255, 0) # Xanh lá (An toàn)
    if ai_result.get("child_detected"):
        color = (0, 0, 255) # Đỏ (Báo động có trẻ em)
        
    text = f"{class_name}: {confidence:.2f}"
    
    # Vẽ một ô cảnh báo góc trên bên trái
    cv2.rectangle(frame, (10, 10), (300, 50), color, -1)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def camera_thread():
    global latest_frame
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        # Thu nhỏ ảnh lại chút cho stream qua mạng mượt hơn
        frame = cv2.resize(frame, (640, 480))
        
        # Nhờ AI phán đoán
        ai_result = get_ai_prediction(frame)
        
        # Vẽ thông tin lên ảnh
        frame = draw_boxes(frame, ai_result)
        
        # Cập nhật frame mới nhất để Web lấy
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame = buffer.tobytes()
            
        # Không bắt AI làm việc quá sức (giới hạn 5 FPS)
        time.sleep(0.2)

def generate_video_stream():
    """Hàm phát (stream) hình ảnh liên tục cho trình duyệt Web."""
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

@app.route('/')
def index():
    html = """
    <html>
      <head>
        <title>CPDS-AI Web Dashboard</title>
        <style>
          body { background-color: #1a1a1a; color: white; font-family: Arial; text-align: center; }
          img { border: 5px solid #333; border-radius: 10px; max-width: 100%; }
        </style>
      </head>
      <body>
        <h2>Live Camera & AI Vision Dashboard</h2>
        <img src="/video_feed" />
      </body>
    </html>
    """
    return html

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting Web Camera Thread...")
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    
    print("\n🌐 Web Dashboard is running!")
    print("Mở Google Chrome trên Laptop và gõ địa chỉ mạng của con Pi 4:")
    print("http://<IP-CỦA-PI-4>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
