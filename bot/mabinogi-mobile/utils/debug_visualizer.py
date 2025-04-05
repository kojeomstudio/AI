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
    else:
        dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, str(in_origin))

def visualize_prediction(model, screen_img, window_name="YOLO 디버그"):
    results = model.predict(screen_img, conf=0.5, verbose=False)
    result = results[0]

    print(f"[INFO] 총 {len(result.boxes)}개의 객체 탐지됨")

    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        class_id = int(cls)
        label = result.names[class_id]

        # 중심 좌표 계산
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 콘솔 출력
        print(f" - 클래스: {label} | 신뢰도: {conf:.2f} | 위치: ({cx}, {cy})")

        # 바운딩 박스 그리기
        cv2.rectangle(screen_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 텍스트 정보 그리기
        text = f"{label} {conf:.2f}"
        cv2.putText(screen_img, text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 중심점 시각화
        cv2.circle(screen_img, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(screen_img, f"({cx},{cy})", (cx + 5, cy),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

    # 결과 시각화
    cv2.imshow(window_name, screen_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = load_config(get_file_path("../config/config.json"))

    model = YOLO(get_file_path("../ml/training_output/vein_model/weights/best.pt"))
    screen_img = get_game_window_image(config["window_title"])  # 게임 창 스크린샷

    if screen_img is None:
        print("[ERROR] 게임 창을 찾을 수 없습니다.")
        return

    if len(screen_img.shape) == 2 or screen_img.shape[2] == 1:
        screen_color = cv2.cvtColor(screen_img, cv2.COLOR_GRAY2BGR)
    else:
        screen_color = screen_img.copy()

    visualize_prediction(model, screen_color)

if __name__ == "__main__":
    main()
