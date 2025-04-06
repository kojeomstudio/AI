import time
import json
import sys
import os

from ultralytics import YOLO

from ui.vein import *
from ui.action import *
from ui.base.element import *
from logger_helper import get_logger
from utils.capture import *

logger = get_logger()
config = None

def get_file_path(in_origin):
    """실행 환경에 따라 상대 경로 처리"""
    base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, str(in_origin))

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_elements(results, elements):
    """모든 요소와 YOLO 결과 매칭"""
    matched = {}
    for element in elements:
        is_match, pos = element.match(results)
        if is_match:
            matched[element.ui_type] = (element, pos)
    return matched

def process_logic(matched):
    """매칭된 요소 기반 동작 처리"""
    # COMPASS 또는 WORKING 상태면 아무것도 하지 않음
    if UIElementType.COMPASS in matched or UIElementType.WORKING in matched:
        logger.debug("대기 상태 또는 작업 중 상태이므로 동작하지 않음")
        return False

    # 채굴 조건: 채굴 UI + 광맥 중 하나
    if UIElementType.MINING in matched:
        if UIElementType.COAL_VEIN in matched or UIElementType.IRON_VEIN in matched:
            element, pos = matched[UIElementType.MINING]
            logger.info("→ 채굴 조건 만족, 채굴 실행")
            element.action(pos)
            return True
        else:
            logger.debug("채굴 UI 감지됨, 그러나 광맥 없음")

    # 벌채 조건: 벌채 UI (추후 나무 노드 존재 여부도 체크 가능)
    elif UIElementType.FELLING in matched:
        # TODO: 나무 노드(TreeNode 등) 존재할 때만 실행하도록 조건 강화 예정
        element, pos = matched[UIElementType.FELLING]
        logger.info("→ 벌채 조건 만족 (조건 검증 생략), 벌채 실행")
        element.action(pos)
        return True

    return False

def main_loop(model, elements, tick=0.5):
    try:
        while True:
            screen_np = get_game_window_image(config["window_title"])
            results = model.predict(screen_np, conf=0.5, verbose=False)
            matched = match_elements(results, elements)

            if process_logic(matched):
                time.sleep(2)
            else:
                logger.debug("처리된 액션 없음")

            time.sleep(tick)

    except KeyboardInterrupt:
        logger.info("[EXIT] 매크로 종료됨")

if __name__ == "__main__":
    config = load_config(get_file_path("./config/config.json"))

    model_path = get_file_path("ml/training_output/mabi_model/weights/best.pt")
    model = YOLO(model_path)

    elements = [
        CoalNode(UIElementType.COAL_VEIN, class_id=0),
        IronNode(UIElementType.IRON_VEIN, class_id=1),
        UI_Felling(UIElementType.FELLING, class_id=2),
        UI_Mining(UIElementType.MINING, class_id=3),
        UI_Compass(UIElementType.COMPASS, class_id=4),
        UI_Working(UIElementType.WORKING, class_id=5),
    ]

    logger.info("[START] YOLO 매크로 실행 중...")
    main_loop(model, elements, tick=config.get("tick", 1.0))
