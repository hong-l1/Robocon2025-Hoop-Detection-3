from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的最佳模型
    model = YOLO('./runs/detect/train3/weights/best.pt')  # 根据实际路径调整

    # 在测试集上推理并评估
    results = model.val(
        data='./Robocon2025-Hoop-Detection-3/data.yaml',
        split='test',  # 使用测试集
        imgsz=1024,
        device=0  # 使用 GPU
    )

    # 可视化测试结果（显示并保存边界框）
    model.predict(
        source='./Robocon2025-Hoop-Detection-3/test/images',  # 测试集图像路径
        conf=0.3,  # 降低置信度阈值，确保更多目标显示
        save=True,  # 保存带边界框的结果图像
        show=True,  # 实时显示带边界框的图像
        save_txt=False,  # 不保存坐标文本文件
        device=0,
        project='runs/test_results',  # 结果保存的主目录
        name='detections',  # 结果保存的子目录
        imgsz=1024
    )

# from ultralytics import YOLO
# import cv2
# import numpy as np
#
#
# def main():
#     # 加载YOLO模型
#     model = YOLO('./runs/detect/train3/weights/best.pt')
#
#     # 视频路径
#     video_path = 'dataset/test2.mp4'
#
#     # 打开视频流
#     cap = cv2.VideoCapture(video_path)
#     assert cap.isOpened(), f"无法打开视频源: {video_path}"
#
#     # 定义边界框的颜色和线宽（增强可视化）
#     box_color = (0, 255, 0)  # 绿色
#     box_thickness = 2
#     text_color = (255, 255, 255)  # 白色文字
#     text_bg_color = (0, 0, 255)  # 红色背景
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 模型推理
#         results = model.predict(
#             source=frame,
#             conf=0.3,  # 降低置信度阈值，确保更多目标被检测到
#             imgsz=640,
#             device=0,
#             verbose=False
#         )
#
#         # 获取检测结果
#         detections = results[0].boxes
#
#         # 手动绘制边界框和标签（更直观展示）
#         for box in detections:
#             # 获取边界框坐标
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # 左上角和右下角坐标
#
#             # 获取类别和置信度
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])
#             class_name = model.names[cls]
#
#             # 绘制边界框
#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
#
#             # 绘制标签（带背景）
#             label = f"{class_name}: {conf:.2f}"
#             text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#             cv2.rectangle(frame, (x1, y1 - text_size[1] - 5),
#                           (x1 + text_size[0], y1), text_bg_color, -1)
#             cv2.putText(frame, label, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
#
#         # 显示结果
#         cv2.imshow('YOLO 边界框可视化', frame)
#
#         # 按'q'退出
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     # 释放资源
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
