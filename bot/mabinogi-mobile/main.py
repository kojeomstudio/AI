import time
import json
import sys
import os

from ultralytics import YOLO

from ui.vein import CoalNode, IronNode
from logger_helper import get_logger
from capture import *

logger = get_logger()
config = None

def get_file_path(in_origin):
    """실행 환경에 따라 상대 경로 처리"""
    if getattr(sys, 'frozen', False):  # PyInstaller 실행 환경
        dir = os.path.dirname(sys.executable)
    else:  # 개발 중
        dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, str(in_origin))

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main_loop(model, elements, tick=0.5):
    try:
        while True:
            screen_np = get_game_window_image(config["window_title"])

            results = model.predict(screen_np, conf=0.5, verbose=False)

            for element in elements:
                matched, pos = element.match(results)
                if matched:
                    element.action(pos)
                    time.sleep(2)
                    break

            time.sleep(tick)
    except KeyboardInterrupt:
        logger.info("[EXIT] 매크로 종료됨")

if __name__ == "__main__":
    config = load_config(get_file_path("./config/config.json"))
    
    # 학습된 YOLO 모델 경로
    model_path = get_file_path("ml/training_output/vein_model/weights/best.pt")
    model = YOLO(model_path)

    # 클래스 ID에 따라 요소 정의 (YOLO 모델의 class 순서와 매핑)
    elements = [
        CoalNode("석탄 광맥", class_id=0),
        IronNode("철 광맥", class_id=1)
    ]

    logger.info("[START] YOLO 매크로 실행 중...")
    main_loop(model, elements, tick=config.get("tick", 1.0))
