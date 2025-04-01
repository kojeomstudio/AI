from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
img = cv2.imread("test_screen.png")
results = model(img)

for box in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls_id = box
    label = results[0].names[int(cls_id)]
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    print(f"[{label}] at ({cx}, {cy}) conf={conf:.2f}")
