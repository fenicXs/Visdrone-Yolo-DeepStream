.PHONY: help install train-n train-s eval-n eval-s

help:
	@echo "Targets:"
	@echo "  install   - pip install -r requirements.txt"
	@echo "  train-n   - train YOLOv8n @ 640"
	@echo "  train-s   - train YOLOv8s @ 640"
	@echo "  eval-n    - eval YOLOv8n run"
	@echo "  eval-s    - eval YOLOv8s run"

install:
	pip install -U pip
	pip install -r requirements.txt

train-n:
	python scripts/02_train.py --config configs/train/yolo8n_640.yaml

train-s:
	python scripts/02_train.py --config configs/train/yolo8s_640.yaml

eval-n:
	python scripts/03_eval.py --weights runs/visdrone/yolov8n_640/weights/best.pt --train-config configs/train/yolo8n_640.yaml

eval-s:
	python scripts/03_eval.py --weights runs/visdrone/yolov8s_640/weights/best.pt --train-config configs/train/yolo8s_640.yaml
