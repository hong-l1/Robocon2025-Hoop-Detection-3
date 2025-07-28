from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载上次训练保存的模型（通常是runs/train/exp/weights/last.pt）
    model = YOLO('./runs/detect/train3/weights/last.pt')  # 请根据实际路径调整
    results = model.train(
        data="./Robocon2025-Hoop-Detection-3/data.yaml",
        epochs=200,
        imgsz=640,
        workers=2,
        device=0,  # 如果有GPU，改为 device=0
        resume=True  # 关键参数：从上次的训练继续
    )