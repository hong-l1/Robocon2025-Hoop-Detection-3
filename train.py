from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('yolov10s.pt')  # 请根据实际路径调整
    results = model.train(
        data="./basketball-1/data.yaml",
        epochs=100,
        imgsz=640,
        workers=2,
        device=0,  # 如果有GPU，改为 device=0
    )