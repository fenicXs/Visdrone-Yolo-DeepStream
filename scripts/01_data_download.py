from ultralytics import YOLO

# This will trigger VisDrone download/convert if not present, because VisDrone.yaml includes download logic.
YOLO("yolov8n.pt").train(data="configs/dataset/visdrone.yaml", epochs=1, imgsz=640)
