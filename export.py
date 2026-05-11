from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model

# Export the model
model.export(format="onnx", imgsz=640, half=True)
