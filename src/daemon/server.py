import socket
import json
import base64
import traceback
from pathlib import Path
from src.daemon.config import Config
from src.daemon.backend import ONNXBackend

class InferenceServer:
    def __init__(self, config: Config):
        self.config = config
        self.socket_path = Path(config.socket_path)
        if config.backend_type == "onnx":
            self.backend = ONNXBackend(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend_type}")

    def run(self):
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
            server_sock.bind(str(self.socket_path))
            server_sock.listen()
            print(f"Listening on {self.socket_path}")
            
            while True:
                try:
                    conn, _ = server_sock.accept()
                    self._handle_client(conn)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Accept error: {e}")

    def _handle_client(self, conn: socket.socket):
        with conn:
            try:
                with conn.makefile('r', encoding='utf-8') as infile, \
                     conn.makefile('w', encoding='utf-8') as outfile:
                    for line in infile:
                        if not line.strip(): continue
                        response = self._process_request(line)
                        outfile.write(json.dumps(response) + "\n")
                        outfile.flush()
            except Exception as e:
                print(f"Connection handler error: {e}")

    def _process_request(self, line: str) -> dict:
        try:
            req = json.loads(line)
            req_type = req.get("type")
            if req_type == "vision":
                img_data = base64.b64decode(req["jpeg_b64"])
                result = self.backend.predict_vision(img_data)
                return {"ok": True, "result": result}
            elif req_type == "audio":
                audio_data = base64.b64decode(req["pcm_b64"])
                sr = req.get("sample_rate", 16000)
                result = self.backend.predict_audio(audio_data, sr)
                return {"ok": True, "result": result}
            else:
                return {"ok": False, "error": f"Unknown request type: {req_type}"}
        except Exception as e:
            # traceback.print_exc()
            return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    InferenceServer(Config()).run()
