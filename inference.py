import cv2
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Config
MODEL_PATH = "yolo26n.pt"
VIDEO_SOURCE = "commercial.mp4"
CONF_THRESH = 0.5

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# Shared state
output_frame = None
lock = threading.Lock()

# Inference thread
def detect():
    global output_frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.open(VIDEO_SOURCE)
            continue

        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        annotated = results[0].plot()

        with lock:
            output_frame = annotated.copy()

# HTTP stream handler
class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with lock:
                if output_frame is None:
                    continue
                _, jpeg = cv2.imencode(".jpg", output_frame)
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
            except BrokenPipeError:
                break

# Main
threading.Thread(target=detect, daemon=True).start()

print("Stream running at http://127.0.0.1:8080")
print("Press Ctrl+C to stop.")

server = HTTPServer(("0.0.0.0", 8080), StreamHandler)
server.serve_forever()
