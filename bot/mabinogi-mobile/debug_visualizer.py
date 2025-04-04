import cv2
import numpy as np
from ultralytics import YOLO
from capture import get_game_window_image
import sys
import os
import json

def get_file_path(in_origin):
    """실행 환경에 따라 상대 경로 처리"""
    if getattr(sys, 'frozen', False):  # PyInstaller 실행 환경
        dir = os.path.dirname(sys.executable)
    else:  # 개발 중
        dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, str(in_origin))

def visualize_prediction(model, screen_img, window_name="YOLO 디버그"):
    results = model.predict(screen_img, conf=0.5, verbose=True)
    result = results[0]

    print(f"predict result num : {len(results)}")

    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = result.names[int(cls)]
        cv2.rectangle(screen_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(screen_img, f"{label} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 화면에 보여주기
    cv2.imshow(window_name, screen_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = load_config(get_file_path("./config/config.json"))

    model = YOLO(get_file_path("ml/training_output/vein_model/weights/best.pt"))
    screen_img = get_game_window_image(config["window_title"])  # 실제 창 제목

    if screen_img is None:
        print("게임 창을 찾을 수 없습니다.")
        return

    if len(screen_img.shape) == 2 or screen_img.shape[2] == 1:
        screen_color = cv2.cvtColor(screen_img, cv2.COLOR_GRAY2BGR)
    else:
        screen_color = screen_img.copy()

    visualize_prediction(model, screen_color)

if __name__ == "__main__":
    main()
