import cv2
import numpy as np
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "yolo26n.onnx"
VIDEO_SOURCE = "commercial.mp4"
INPUT_SIZE   = (640, 640)
CONF_THRESH  = 0.5

# COCO class names
CLASSES      = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ── Build ONNX Runtime session with TensorRT EP ───────────────────────────────
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 1 << 30,          # 1GB workspace
            "trt_fp16_enable": True,                    # FP16 for faster inference on Jetson
            "trt_engine_cache_enable": True,            # Cache TRT engine to disk
            "trt_engine_cache_path": "./trt_engine",    # Where to save the cache
        },
    ),
    "CUDAExecutionProvider",    # Fallback if TensorRT fails
    "CPUExecutionProvider",     # Final fallback
]

print("Loading model...")
session = ort.InferenceSession(MODEL_PATH, providers=providers)
print(f"Using provider: {session.get_providers()}")

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ── Shared state ──────────────────────────────────────────────────────────────
output_frame = None
lock = threading.Lock()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0       # Normalise to [0, 1]
    img = np.transpose(img, (2, 0, 1))         # HWC → CHW
    img = np.expand_dims(img, axis=0)          # Add batch dimension
    return img

# ── Postprocessing ────────────────────────────────────────────────────────────
def postprocess(output, orig_shape):
    predictions = output[0][0]
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / INPUT_SIZE[0]
    scale_y = orig_h / INPUT_SIZE[1]

    detections = []
    for pred in predictions:
        x1, y1, x2, y2, score, class_id = pred
        if score < CONF_THRESH:
            continue
        detections.append({
            "box":        (int(x1 * scale_x), int(y1 * scale_y),
                           int(x2 * scale_x), int(y2 * scale_y)),
            "score":      float(score),
            "class_id":   int(class_id),
            "class_name": CLASSES[int(class_id)],
        })
    return detections

# ── Draw detections ───────────────────────────────────────────────────────────
def draw(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["score"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ── Inference thread ──────────────────────────────────────────────────────────
def detect():
    global output_frame
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.open(VIDEO_SOURCE)
            continue

        inp        = preprocess(frame)
        output     = session.run([output_name], {input_name: inp})
        detections = postprocess(output, frame.shape)
        annotated  = draw(frame, detections)

        with lock:
            output_frame = annotated.copy()

# ── HTTP stream handler ───────────────────────────────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────────────────────
threading.Thread(target=detect, daemon=True).start()

print("Stream running at http://192.168.50.226:8080")
print("Press Ctrl+C to stop.")

server = HTTPServer(("0.0.0.0", 8080), StreamHandler)
server.serve_forever()
